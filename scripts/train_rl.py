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
    ap.add_argument("--no_pool", action="store_true",
                    help="ablation: full-res CNN (ignored if --init sets the arch)")
    ap.add_argument("--workers", type=int, default=0,
                    help="parallel rollout workers (0/1 = serial)")
    args = ap.parse_args()

    dev = args.device
    no_pool = args.no_pool
    anchor = None
    if args.init and os.path.exists(args.init):
        ckpt = torch.load(args.init, map_location=dev)
        no_pool = ckpt.get("no_pool", False)  # inherit arch from the init checkpoint
        sd = ckpt["model"]
        model = PriorityUNet(pool=not no_pool).to(dev)
        model.load_state_dict(sd)
        print(f"warm-started from {args.init} (no_pool={no_pool})")
        if args.anchor_w > 0:
            anchor = PriorityUNet(pool=not no_pool).to(dev)
            anchor.load_state_dict(sd)
            anchor.eval()
            for p in anchor.parameters():
                p.requires_grad_(False)
    else:
        model = PriorityUNet(pool=not no_pool).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(0)

    eval_maps = make_eval_maps(n_per_kind=8)
    eval_inst = make_instances(eval_maps, n_agents=args.n_agents, n_inst=3)

    def bench(tag):
        provider = lambda g: predict_field(model, g, device=dev)
        print_report(f"{tag}", evaluate(provider, eval_inst))

    print_report("MST baseline", evaluate(baseline_provider, eval_inst))
    bench("learned @ init")

    # Rollouts are pure-CPU NumPy and independent across (map, sample); a spawn
    # pool parallelizes them without touching the parent's CUDA context.
    pool = None
    if args.workers and args.workers > 1:
        pool = mp.get_context("spawn").Pool(args.workers)
        print(f"using {args.workers} rollout workers")

    try:
        ema = None
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        for it in range(1, args.iters + 1):
            maps = sample_maps(args.batch_maps, args.size, rng)
            stats = rl_step(model, maps, opt, dev, K=args.K, sigma=args.sigma,
                            n_agents=args.n_agents, rng=rng,
                            anchor=anchor, anchor_w=args.anchor_w, pool=pool)
            ema = stats["reward"] if ema is None else 0.95 * ema + 0.05 * stats["reward"]
            if it % 10 == 0:
                print(f"it {it:4d}  loss {stats['loss']:+.3f}  reward {stats['reward']:+.3f}  ema {ema:+.3f}")
            if it % args.eval_every == 0:
                bench(f"learned @ it{it}")
                torch.save({"model": model.state_dict(), "no_pool": no_pool}, args.out)
        torch.save({"model": model.state_dict(), "no_pool": no_pool}, args.out)
        print(f"saved -> {args.out}")
    finally:
        if pool is not None:
            pool.close()
            pool.join()


if __name__ == "__main__":
    main()
