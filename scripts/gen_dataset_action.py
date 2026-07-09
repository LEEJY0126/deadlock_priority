"""Generate an imitation dataset for the action-map policy.

Rolls out an expert that never collides -- the trained priority model (default
``runs/rl_embedding_best.pt``) driving PIBT, or the MST baseline -- on fresh random
maps, and caches each episode's ``(occupancy, position log)``. Because PIBT
guarantees a legal joint move every step, every recorded transition is a valid
collision-free demonstration; per-step, per-agent action labels are reconstructed
from the position log at train time (see ``train_imitation_action.py``).
"""
import sys, os, argparse, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from tqdm import tqdm

from src.envs.grid import maze, random_forest, sample_start_goals
from src.envs.simulator import ORACLES
from src.priority.model import load_model
from src.train.imitation_action import expert_positions_log


def make_map(kind, size, rng):
    if kind == "forest":
        return random_forest(size, size, n_obstacles=int(size * size * 0.07), rng=rng)
    if kind == "wide":
        return maze(size, size, corridor=2, braid=0.25, rng=rng)
    if kind == "narrow":
        return maze(size, size, corridor=1, braid=0.25, rng=rng)
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/imitation_action.npz")
    ap.add_argument("--expert", default="runs/rl_embedding_best.pt",
                    help="priority checkpoint to drive PIBT (any arch); "
                         "'' or 'mst' uses the MST baseline field")
    ap.add_argument("--n_maps", type=int, default=120)
    ap.add_argument("--n_inst", type=int, default=4, help="start/goal sets per map")
    ap.add_argument("--size", type=int, default=21)
    ap.add_argument("--n_agents", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=256)
    ap.add_argument("--oracle", choices=ORACLES, default="paper",
                    help="PIBT resolution mode for the expert rollouts")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    _argv = getattr(sys, "orig_argv", None) or ([sys.executable] + sys.argv)
    print(f"command: {' '.join(_argv)}", flush=True)

    expert = None
    if args.expert and args.expert.lower() != "mst":
        expert = load_model(args.expert, device=args.device)
        print(f"expert: {args.expert} ({type(expert).__name__})", flush=True)
    else:
        print("expert: MST baseline field", flush=True)

    rng = np.random.default_rng(args.seed)
    kinds = ["forest", "wide", "narrow"]
    occs, positions, kinds_log = [], [], []
    n_steps = 0
    t0 = time.time()
    for m in tqdm(range(args.n_maps)):
        kind = kinds[m % len(kinds)]
        gmap = make_map(kind, args.size, rng)
        for _ in range(args.n_inst):
            starts, goals = sample_start_goals(gmap, args.n_agents, rng=rng, min_sep=4)
            log = expert_positions_log(expert, gmap, starts, goals,
                                       oracle=args.oracle, max_steps=args.max_steps,
                                       device=args.device, rng=rng)
            if len(log) < 2:
                continue  # degenerate (all agents already on goal): no transitions
            occs.append(gmap.occ.astype(np.uint8))
            # store as a plain nested list so numpy pickles it as one object entry
            positions.append([[(int(r), int(c)) for (r, c) in step] for step in log])
            kinds_log.append(kind)
            n_steps += len(log) - 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    meta = json.dumps(vars(args))
    pos_arr = np.empty(len(positions), dtype=object)
    pos_arr[:] = positions
    np.savez_compressed(args.out,
                        occ=np.stack(occs).astype(np.uint8),
                        positions=pos_arr,
                        kind=np.array(kinds_log),
                        meta=np.array(meta))
    print(f"saved {len(occs)} episodes ({n_steps} transitions) to {args.out} "
          f"in {time.time()-t0:.1f}s", flush=True)
    print(f"  config: {meta}", flush=True)


if __name__ == "__main__":
    main()
