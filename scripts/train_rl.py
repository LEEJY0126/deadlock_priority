"""RL fine-tuning loop. Optionally warm-starts from the imitation checkpoint,
then improves the priority field on fresh random maps via group-baseline
REINFORCE, periodically benchmarking against the MST baseline.
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import multiprocessing as mp

import numpy as np
import torch
from src.envs.grid import maze, random_forest
from src.priority import features as features_mod
from src.priority import model as model_mod
from src.priority.model import build_model, predict_field
from src.train.rl import rl_step
from src.train.reward import RewardWeights
from src.eval.benchmark import (make_eval_maps, make_instances, evaluate,
                                baseline_provider, print_report)
from src.utils.experiment import Experiment


def _one_map(kind, size, rng):
    if kind == "forest":
        return random_forest(size, size, int(size * size * 0.07), rng=rng)
    if kind == "wide":
        return maze(size, size, corridor=2, rng=rng)
    return maze(size, size, corridor=1, rng=rng)


def sample_maps(B, size, rng):
    """Balanced across map kinds so no type is forgotten during RL."""
    kinds = ["forest", "wide", "narrow"]
    return [_one_map(kinds[b % 3], size, rng) for b in range(B)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=None, help="imitation checkpoint to warm-start")
    ap.add_argument("--out", default="runs/rl.pt")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--batch_maps", type=int, default=4)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--anchor_w", type=float, default=0.5)
    ap.add_argument("--size", type=int, default=21)
    ap.add_argument("--n_agents", type=int, default=8)
    ap.add_argument("--eval_every", type=int, default=25)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--arch", choices=["unet", "transformer"], default="unet",
                    help="model architecture for cold start (ignored if --init sets it)")
    ap.add_argument("--no_pool", action="store_true",
                    help="ablation: full-res CNN (ignored if --init sets the arch)")
    ap.add_argument("--dim", type=int, default=128, help="transformer token dim (cold start)")
    ap.add_argument("--depth", type=int, default=4, help="transformer layers (cold start)")
    ap.add_argument("--heads", type=int, default=4, help="transformer heads (cold start)")
    ap.add_argument("--dropout", type=float, default=0.0, help="transformer dropout (cold start)")
    ap.add_argument("--workers", type=int, default=0,
                    help="parallel rollout workers (0/1 = serial; cpu engine only)")
    ap.add_argument("--engine", choices=["cpu", "vec"], default="cpu",
                    help="reward rollouts: cpu=exact PIBT, vec=GPU-batched approx")
    ap.add_argument("--oracle", choices=["beta", "paper"], default="beta",
                    help="PIBT deadlock-resolution mode for the RL reward rollouts "
                         "(beta=legacy boost, paper=right-hand rule + livelock; "
                         "cpu engine only -- vec uses its own approximate solver)")
    ap.add_argument("--reward_weights", default="reward_weight.yaml",
                    help="YAML of reward shaping weights")
    args = ap.parse_args()

    weights = RewardWeights.load(args.reward_weights)
    exp = Experiment("train_rl", config={
        "total_iters": args.iters,
        "sigma": args.sigma,
        "learning_rate": args.lr,
        "n_agents": args.n_agents,
        "batch_maps": args.batch_maps,
        "K": args.K,
        "anchor_w": args.anchor_w,
        "engine": args.engine,
        "oracle": args.oracle,
        "init": args.init,
        "arch": args.arch,
        "reward_weights": weights.to_dict(),
    })
    exp.save_yaml("reward_weight.yaml", weights.to_dict())  # snapshot for reproducibility
    exp.snapshot(model_mod.__file__, "model.py")
    exp.snapshot(features_mod.__file__, "features.py")
    exp.log(f"reward weights: {weights.to_dict()}")

    dev = args.device
    arch, no_pool = args.arch, args.no_pool
    config = ({} if args.arch == "unet" else
              dict(dim=args.dim, depth=args.depth, heads=args.heads, dropout=args.dropout))
    anchor = None
    if args.init and os.path.exists(args.init):
        ckpt = torch.load(args.init, map_location=dev)
        # inherit the full architecture from the init checkpoint
        arch = ckpt.get("arch", "unet")
        no_pool = ckpt.get("no_pool", False)
        config = ckpt.get("config", {})
        sd = ckpt["model"]
        model = build_model(arch, no_pool=no_pool, **config).to(dev)
        model.load_state_dict(sd)
        exp.log(f"warm-started from {args.init} (arch={arch}, no_pool={no_pool})")
        if args.anchor_w > 0:
            anchor = build_model(arch, no_pool=no_pool, **config).to(dev)
            anchor.load_state_dict(sd)
            anchor.eval()
            for p in anchor.parameters():
                p.requires_grad_(False)
    else:
        model = build_model(arch, no_pool=no_pool, **config).to(dev)
    if arch == "transformer":  # snapshot the resolved arch (may come from --init)
        from src.priority import model_transformer as mt_mod
        exp.snapshot(mt_mod.__file__, "model_transformer.py")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(0)

    eval_maps = make_eval_maps(n_per_kind=8)
    eval_inst = make_instances(eval_maps, n_agents=args.n_agents, n_inst=3)

    def bench(tag, step=None):
        # best.pt selection + logged metrics use the same deadlock-resolution
        # mode as the training rollouts (--oracle), so the chosen iterate is best
        # under the dynamics it was trained on.
        provider = lambda g: predict_field(model, g, device=dev)
        report = evaluate(provider, eval_inst, yield_mode=args.oracle)
        print_report(f"{tag}", report)
        if step is not None:
            for kind, r in report.items():
                exp.scalar(f"eval_success/{kind}", r["success_rate"], step)
                exp.scalar(f"eval_makespan/{kind}", r["makespan"], step)
                exp.scalar(f"eval_flowtime/{kind}", r["flowtime"], step)
        return report

    def mean_success(report):
        """Headline metric for best-checkpoint selection: mean success over kinds."""
        return sum(r["success_rate"] for r in report.values()) / len(report)

    print_report("MST baseline", evaluate(baseline_provider, eval_inst, yield_mode=args.oracle))
    init_report = bench("learned @ init", step=0)

    # Rollouts are pure-CPU NumPy and independent across (map, sample); a spawn
    # pool parallelizes them without touching the parent's CUDA context.
    pool = None
    if args.workers and args.workers > 1 and args.engine == "cpu":
        pool = mp.get_context("spawn").Pool(args.workers)
        exp.log(f"using {args.workers} rollout workers")
    if args.engine == "vec":
        exp.log("using GPU-vectorized rollout engine (approximate solver)")

    def save(name, mirror=False):
        ckpt = {"model": model.state_dict(), "arch": arch,
                "no_pool": no_pool, "config": config}
        torch.save(ckpt, exp.path(name))
        if mirror:               # only the best checkpoint is mirrored to --out
            torch.save(ckpt, args.out)

    try:
        ema = None
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        # best-by-success tracking: seed from the init eval so best.pt / --out are
        # always populated, then overwrite whenever an eval improves mean success.
        best_success = mean_success(init_report)
        save("best.pt", mirror=True)
        exp.log(f"init best success {best_success:.1f}% -> best.pt and {args.out}")
        for it in range(1, args.iters + 1):
            maps = sample_maps(args.batch_maps, args.size, rng)
            stats = rl_step(model, maps, opt, dev, K=args.K, sigma=args.sigma,
                            n_agents=args.n_agents, rng=rng,
                            anchor=anchor, anchor_w=args.anchor_w, pool=pool,
                            engine=args.engine, weights=weights, yield_mode=args.oracle)
            ema = stats["reward"] if ema is None else 0.95 * ema + 0.05 * stats["reward"]
            exp.scalar("reward/step", stats["reward"], it)
            exp.scalar("reward/ema", ema, it)
            exp.scalar("loss", stats["loss"], it)
            if it % 10 == 0:
                exp.log(f"it {it:4d}  loss {stats['loss']:+.3f}  "
                        f"reward {stats['reward']:+.3f}  ema {ema:+.3f}")
            if it % args.eval_every == 0:
                report = bench(f"learned @ it{it}", step=it)
                save("checkpoint.pt")  # latest state, for resume/inspection
                ms = mean_success(report)
                exp.scalar("eval_success/mean", ms, it)
                if ms > best_success:
                    best_success = ms
                    save("best.pt", mirror=True)
                    exp.log(f"  new best success {ms:.1f}% @ it{it} -> best.pt and {args.out}")
        save("final.pt")  # end-of-training state (run dir only; --out holds best)
        exp.log(f"saved final -> {exp.path('final.pt')}; "
                f"best ({best_success:.1f}%) -> {exp.path('best.pt')} and {args.out}")
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        exp.close()


if __name__ == "__main__":
    main()
