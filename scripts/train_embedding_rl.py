"""Per-step PPO training for the dynamic embedding priority model (option B).

Trains MapEncoder + PriorityDecoder end-to-end with a state-value critic on
fresh random maps, driving the real Simulator under paper-mode resolution.
Each iteration collects a batch of episodes, computes GAE(lambda) advantages,
and does a clipped multi-epoch PPO update; it periodically benchmarks the greedy
policy against MST. Saves a checkpoint loadable by ``src.priority.model.load_model``
(arch="embedding"), so ``scripts/evaluate.py --ckpt ...`` picks it up
automatically. (``--algo a2c`` falls back to the single-episode A2C step.)
"""
import sys, os, argparse, shlex
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.envs.grid import maze, random_forest
from src.envs.simulator import ORACLES
from src.priority.model import build_model, load_model
from src.priority.model_embedding import EmbeddingPriorityModel, InferenceTimer
from src.train.reward import StepRewardWeights
from src.train.rl_embedding import (make_critic, train_embedding_episode,
                                    train_embedding_ppo_step)
from src.eval.benchmark import (make_eval_maps, make_instances, evaluate,
                                evaluate_embedding, baseline_provider, print_report,
                                print_inference_timing)
from src.envs.grid import sample_start_goals


def _one_map(kind, size, rng):
    if kind == "forest":
        return random_forest(size, size, int(size * size * 0.07), rng=rng)
    if kind == "wide":
        return maze(size, size, corridor=2, braid=0.25, rng=rng)
    return maze(size, size, corridor=1, braid=0.25, rng=rng)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=None, help="embedding checkpoint to warm-start")
    ap.add_argument("--out", default="runs/rl_embedding.pt")
    ap.add_argument("--algo", choices=["ppo", "a2c"], default="ppo")
    ap.add_argument("--iters", type=int, default=1000,
                    help="PPO updates (each collects --batch_episodes episodes)")
    ap.add_argument("--batch_episodes", type=int, default=8,
                    help="episodes collected per PPO update")
    ap.add_argument("--epochs", type=int, default=4, help="PPO epochs per batch")
    ap.add_argument("--clip", type=float, default=0.2, help="PPO clip epsilon")
    ap.add_argument("--target_kl", type=float, default=0.03,
                    help="stop PPO epochs early once mean |approx_kl| exceeds this "
                         "(trust-region guard; set <=0 to disable)")
    ap.add_argument("--lam", type=float, default=0.95, help="GAE lambda")
    ap.add_argument("--size", type=int, default=17)
    ap.add_argument("--n_agents", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--sigma", type=float, default=0.5, help="initial exploration std")
    ap.add_argument("--sigma_final", type=float, default=0.1,
                    help="exploration std at the last iter (linear anneal)")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--value_coef", type=float, default=0.5)
    ap.add_argument("--max_steps", type=int, default=256)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--enc_depth", type=int, default=4)
    ap.add_argument("--dec_depth", type=int, default=2)
    ap.add_argument("--history", type=int, default=8,
                    help="occupancy-history frames fed to the decoder")
    ap.add_argument("--hist_dim", type=int, default=64, help="history encoder dim")
    ap.add_argument("--hist_depth", type=int, default=2,
                    help="history encoder transformer layers")
    ap.add_argument("--oracle", choices=ORACLES, default="paper")
    ap.add_argument("--reward_weights", default=None,
                    help="YAML of per-step reward weights (progress/time_penalty/success/reached)")
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--eval_maps_per_kind", type=int, default=6)
    ap.add_argument("--eval_inst_per_map", type=int, default=3)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # Echo the exact launch command to the log for reproducibility. sys.orig_argv
    # preserves the interpreter and its flags (e.g. -u), so a copy-paste reruns
    # the same training. (Falls back to sys.argv on Python < 3.10.)
    _argv = getattr(sys, "orig_argv", None) or ([sys.executable] + sys.argv)
    print(f"command: {shlex.join(_argv)}", flush=True)

    if args.target_kl <= 0:
        args.target_kl = None  # disable the early-stop guard
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    weights = (StepRewardWeights.load(args.reward_weights) if args.reward_weights
               else StepRewardWeights())

    if args.init:
        model = load_model(args.init, device=args.device)
        assert isinstance(model, EmbeddingPriorityModel), "--init must be an embedding ckpt"
    else:
        model = build_model("embedding", dim=args.dim, enc_depth=args.enc_depth,
                            dec_depth=args.dec_depth, history=args.history,
                            hist_dim=args.hist_dim, hist_depth=args.hist_depth).to(args.device)
    critic = make_critic(model).to(args.device)
    opt = torch.optim.Adam(list(model.parameters()) + list(critic.parameters()),
                           lr=args.lr)

    eval_inst = make_instances(
        make_eval_maps(n_per_kind=args.eval_maps_per_kind, size=args.size),
        n_agents=args.n_agents, n_inst=args.eval_inst_per_map)

    def save(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"arch": "embedding", "config": model.config,
                    "model": model.state_dict(), "critic": critic.state_dict()},
                   path)

    best_path = os.path.splitext(args.out)[0] + "_best.pt"

    def do_eval(tag):
        """Report MST vs embedding (+ enc/dec latency); return mean success."""
        print_report(f"MST baseline @{tag}", evaluate(baseline_provider, eval_inst,
                                                      oracle=args.oracle))
        timer = InferenceTimer()
        rep = evaluate_embedding(model, eval_inst, oracle=args.oracle,
                                 device=args.device, timer=timer)
        print_report(f"Embedding @{tag}", rep)
        print_inference_timing(f"Embedding @{tag}", timer)
        return float(np.mean([rep[k]["success_rate"] for k in rep]))

    kinds = ["forest", "wide", "narrow"]
    running = []
    best_score = -1.0
    for it in range(1, args.iters + 1):
        # linear sigma anneal: broad exploration early, refine late.
        frac = (it - 1) / max(args.iters - 1, 1)
        sigma = args.sigma + frac * (args.sigma_final - args.sigma)

        if args.algo == "ppo":
            instances = []
            for b in range(args.batch_episodes):
                g = _one_map(kinds[(it + b) % 3], args.size, rng)
                s, gl = sample_start_goals(g, args.n_agents, rng=rng, min_sep=4)
                instances.append((g, s, gl))
            st = train_embedding_ppo_step(
                model, critic, instances, opt, sigma=sigma, gamma=args.gamma,
                lam=args.lam, weights=weights, clip=args.clip,
                value_coef=args.value_coef, epochs=args.epochs,
                target_kl=args.target_kl, max_steps=args.max_steps,
                oracle=args.oracle, device=args.device, rng=rng)
            running.append(st["reward"])
            if it % args.log_every == 0:
                print(f"[{it:5d}] sigma={sigma:.2f} reward={np.mean(running[-args.log_every:]):+.3f} "
                      f"succ={st['success_rate']*100:4.0f}% reached={st['frac_reached']*100:4.0f}% "
                      f"pi_loss={st['policy_loss']:+.3f} v_loss={st['value_loss']:.2f} "
                      f"kl={st['approx_kl']:+.3f} clip={st['clip_frac']:.2f} "
                      f"ep={st['epochs_run']}/{args.epochs}", flush=True)
        else:
            g = _one_map(kinds[it % 3], args.size, rng)
            s, gl = sample_start_goals(g, args.n_agents, rng=rng, min_sep=4)
            st = train_embedding_episode(
                model, critic, g, s, gl, opt, sigma=sigma, gamma=args.gamma,
                weights=weights, value_coef=args.value_coef,
                max_steps=args.max_steps, oracle=args.oracle, device=args.device,
                rng=rng)
            running.append(st["reward"])
            if it % args.log_every == 0:
                print(f"[{it:5d}] sigma={sigma:.2f} reward={np.mean(running[-args.log_every:]):+.3f} "
                      f"v_loss={st['value_loss']:.2f} succ={st['success']} "
                      f"reached={st['n_reached']}/{args.n_agents}", flush=True)

        if it % args.eval_every == 0:
            score = do_eval(it)
            save(args.out)
            if score > best_score:
                best_score = score
                save(best_path)
                print(f"  new best embedding success={score*100:.1f}% -> {best_path}",
                      flush=True)

    score = do_eval("final")
    save(args.out)
    if score > best_score:
        best_score = score
        save(best_path)
    print(f"saved -> {args.out}  (best embedding success={best_score*100:.1f}% -> {best_path})",
          flush=True)


if __name__ == "__main__":
    main()
