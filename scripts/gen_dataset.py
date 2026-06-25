"""Generate an imitation dataset: per map, search the oracle for the best
map-level priority field and cache (occupancy, label field).

Fields are map-only (no goal conditioning) so each map has a single target,
matching the paper's static position-priority tree.
"""
import sys, os, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tqdm import tqdm
from src.envs.grid import maze, random_forest
from src.train.oracle import best_field


def make_map(kind, size, rng):
    if kind == "forest":
        return random_forest(size, size, n_obstacles=int(size * size * 0.07), rng=rng)
    if kind == "wide":
        return maze(size, size, corridor=2, rng=rng)
    if kind == "narrow":
        return maze(size, size, corridor=1, rng=rng)
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/imitation.npz")
    ap.add_argument("--n_maps", type=int, default=120)
    ap.add_argument("--size", type=int, default=21)
    ap.add_argument("--n_agents", type=int, default=8)
    ap.add_argument("--n_samples", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--oracle", choices=["beta", "paper"], default="beta",
                    help="PIBT deadlock-resolution mode the oracle uses to score "
                         "candidate fields (beta=legacy boost, paper=right-hand rule)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    kinds = ["forest", "wide", "narrow"]
    occs, labels, kinds_log = [], [], []
    t0 = time.time()
    for m in tqdm(range(args.n_maps)):
        kind = kinds[m % len(kinds)]
        gmap = make_map(kind, args.size, rng)
        fld, info = best_field(gmap, n_agents=args.n_agents,
                               n_samples=args.n_samples, seed=int(rng.integers(1 << 30)),
                               yield_mode=args.oracle)
        occs.append(gmap.occ)
        labels.append(fld)
        kinds_log.append(kind)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out,
                        occ=np.stack(occs).astype(np.uint8),
                        label=np.stack(labels).astype(np.float32),
                        kind=np.array(kinds_log))
    print(f"saved {len(occs)} maps to {args.out} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
