"""Throughput benchmark: GPU vectorized envs vs CPU PIBT simulator.

Builds E identical episodes (same maps/instances/fields) and runs them both ways,
reporting episodes/second. Also reports a sanity check on an open map (agents
should reach goals) since the vectorized stepper is a non-backtracking
approximation of PIBT -- see src/envs/vec_sim.py.

  python scripts/bench_vec.py --device cuda
"""
import sys, os, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.envs.grid import maze, random_forest, sample_start_goals
from src.envs.simulator import Simulator
from src.envs.vec_sim import build_batch, VecSim
from src.priority.mst_baseline import mst_priority_field


def make_entries(n, size, n_agents, rng):
    kinds = ["forest", "wide", "narrow"]
    entries = []
    for i in range(n):
        k = kinds[i % 3]
        if k == "forest":
            g = random_forest(size, size, int(size * size * 0.07), rng=rng)
        else:
            g = maze(size, size, corridor=2 if k == "wide" else 1, rng=rng)
        s, gl = sample_start_goals(g, n_agents, rng=rng, min_sep=4)
        entries.append((g, s, gl, mst_priority_field(g)))
    return entries


def cpu_run(entries, max_steps):
    res = []
    for g, s, gl, fld in entries:
        sim = Simulator(g, s, gl, max_steps=max_steps)
        res.append(sim.run(fld))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--size", type=int, default=21)
    ap.add_argument("--n_agents", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=256)
    ap.add_argument("--counts", default="32,128,512,2048")
    ap.add_argument("--cpu_max", type=int, default=512,
                    help="skip the CPU run above this E (eps/s is ~constant)")
    args = ap.parse_args()

    counts = [int(x) for x in args.counts.split(",")]
    print(f"device={args.device} size={args.size} agents={args.n_agents} "
          f"max_steps={args.max_steps}\n")

    # --- sanity: open map, agents should (mostly) reach goals ---
    rng = np.random.default_rng(0)
    g = random_forest(args.size, args.size, 4, rng=rng)
    ents = [(g, sample_start_goals(g, args.n_agents, rng=rng, min_sep=4)[0],
             sample_start_goals(g, args.n_agents, rng=rng, min_sep=4)[1],
             mst_priority_field(g)) for _ in range(64)]
    vb = build_batch(ents, device=args.device)
    vr = VecSim(vb, max_steps=args.max_steps).run()
    cr = cpu_run(ents, args.max_steps)
    print(f"sanity (open map, 64 eps): "
          f"vec success={vr['success'].mean():.0%}  cpu success={np.mean([r.success for r in cr]):.0%}\n")

    print(f"{'E':>6} {'CPU eps/s':>12} {'GPU eps/s':>12} {'GPU build/s':>12} {'speedup':>9}")
    for E in counts:
        rng = np.random.default_rng(123)
        entries = make_entries(E, args.size, args.n_agents, rng)

        if E <= args.cpu_max:
            t0 = time.perf_counter()
            cpu_run(entries, args.max_steps)
            cpu_t = time.perf_counter() - t0
            cpu_eps = f"{E/cpu_t:.1f}"
        else:
            cpu_t, cpu_eps = None, "  (skip)"

        t0 = time.perf_counter()
        batch = build_batch(entries, device=args.device)
        if args.device == "cuda":
            torch.cuda.synchronize()
        build_t = time.perf_counter() - t0

        t0 = time.perf_counter()
        VecSim(batch, max_steps=args.max_steps).run()
        if args.device == "cuda":
            torch.cuda.synchronize()
        gpu_t = time.perf_counter() - t0

        speed = f"{cpu_t/gpu_t:.1f}x" if cpu_t else "    -"
        print(f"{E:>6} {cpu_eps:>12} {E/gpu_t:>12.1f} {E/build_t:>12.1f} {speed:>9}",
              flush=True)


if __name__ == "__main__":
    main()
