"""RL fine-tuning loop. Optionally warm-starts from the imitation checkpoint,
then improves the priority field on fresh random maps via group-baseline
REINFORCE, periodically benchmarking against the MST baseline.
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from src.envs.grid import maze, random_forest
from src.priority.model import PriorityUNet, predict_field
from src.train.rl import rl_step
from src.eval.benchmark import (make_eval_maps, make_instances, evaluate,
                                baseline_provider, print_report)


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
    args = ap.parse_args()

    dev = args.device
    model = PriorityUNet().to(dev)
    anchor = None
    if args.init and os.path.exists(args.init):
        sd = torch.load(args.init, map_location=dev)["model"]
        model.load_state_dict(sd)
        print(f"warm-started from {args.init}")
        if args.anchor_w > 0:
            anchor = PriorityUNet().to(dev)
            anchor.load_state_dict(sd)
            anchor.eval()
            for p in anchor.parameters():
                p.requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(0)

    eval_maps = make_eval_maps(n_per_kind=8)
    eval_inst = make_instances(eval_maps, n_agents=args.n_agents, n_inst=3)

    def bench(tag):
        provider = lambda g: predict_field(model, g, device=dev)
        print_report(f"{tag}", evaluate(provider, eval_inst))

    print_report("MST baseline", evaluate(baseline_provider, eval_inst))
    bench("learned @ init")

    ema = None
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for it in range(1, args.iters + 1):
        maps = sample_maps(args.batch_maps, args.size, rng)
        stats = rl_step(model, maps, opt, dev, K=args.K, sigma=args.sigma,
                        n_agents=args.n_agents, rng=rng,
                        anchor=anchor, anchor_w=args.anchor_w)
        ema = stats["reward"] if ema is None else 0.95 * ema + 0.05 * stats["reward"]
        if it % 10 == 0:
            print(f"it {it:4d}  loss {stats['loss']:+.3f}  reward {stats['reward']:+.3f}  ema {ema:+.3f}")
        if it % args.eval_every == 0:
            bench(f"learned @ it{it}")
            torch.save({"model": model.state_dict()}, args.out)
    torch.save({"model": model.state_dict()}, args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
