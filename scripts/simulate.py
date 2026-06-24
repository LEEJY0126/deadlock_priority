"""Animated side-by-side episode: MST priority vs learned priority.

Runs the *same* start/goal instance under both fields and animates the agents,
so you can watch how the learned priority changes who-yields-to-whom and the
resulting trajectories. The priority field is drawn as the background (viridis),
obstacles in grey, agents as colored dots with fading trails, goals as stars.

  python scripts/simulate.py --ckpt runs/rl.pt --map narrow --out runs/sim.gif
  python scripts/simulate.py --ckpt runs/rl.pt --map narrow --live   # window
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib

from src.envs.grid import maze, random_forest, sample_start_goals
from src.envs.simulator import Simulator
from src.priority.mst_baseline import mst_priority_field
from src.priority.model import load_model, predict_field


def make_map(kind, size, rng):
    if kind == "forest":
        return random_forest(size, size, int(size * size * 0.07), rng=rng)
    return maze(size, size, corridor=2 if kind == "wide" else 1, rng=rng)


def field_background(field, occ, raw=False):
    """Field as an image with obstacles as NaN (drawn grey).

    raw=False: per-map z-score (good contrast for comparing patterns).
    raw=True : the actual priority values.
    """
    free = occ == 0
    v = field.astype(float).copy()
    if not raw and free.sum():
        v = (v - v[free].mean()) / (v[free].std() + 1e-6)
    v[~free] = np.nan
    return v


def draw_raw_map(ax, field, occ, fontsize):
    """Static raw-priority-map subplot: raw values as color + per-cell labels."""
    ax.imshow(field_background(field, occ, raw=True), cmap="viridis")
    H, W = occ.shape
    for r in range(H):
        for c in range(W):
            if occ[r, c] != 0:
                continue
            v = field[r, c]
            txt = f"{v:.0f}" if abs(v - round(v)) < 0.05 else f"{v:.1f}"
            ax.text(c, r, txt, ha="center", va="center", fontsize=fontsize, color="w")
    ax.set_xticks([]); ax.set_yticks([])


def run(g, starts, goals, field, max_steps):
    sim = Simulator(g, starts, goals, max_steps=max_steps, log_positions=True)
    res = sim.run(field)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/rl.pt", help="learned checkpoint")
    ap.add_argument("--map", choices=["forest", "wide", "narrow"], default="narrow")
    ap.add_argument("--size", type=int, default=21)
    ap.add_argument("--n_agents", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/sim.gif")
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--trail", type=int, default=8, help="trail length in steps (0=off)")
    ap.add_argument("--raw", action="store_true",
                    help="show raw priority values + colorbar (default: per-map z-score)")
    ap.add_argument("--live", action="store_true", help="show a window instead of saving")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if not args.live:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    rng = np.random.default_rng(args.seed)
    g = make_map(args.map, args.size, rng)
    starts, goals = sample_start_goals(g, args.n_agents, rng=rng, min_sep=4)

    panels = [("MST priority", mst_priority_field(g))]
    if os.path.exists(args.ckpt):
        model = load_model(args.ckpt, device=args.device)
        panels.append(("learned priority", predict_field(model, g, device=args.device)))
    else:
        print(f"[warn] {args.ckpt} not found -- showing MST only")

    # run the episode under each field
    results = [(name, fld, run(g, starts, goals, fld, args.max_steps))
               for name, fld in panels]
    T = max(len(r.positions_log) for _, _, r in results)

    colors = plt.cm.hsv(np.linspace(0, 1, args.n_agents, endpoint=False))
    ncol = len(results)
    # with --raw, a top row shows the static raw priority map; the simulation
    # always animates in the bottom row.
    nrow = 2 if args.raw else 1
    fig, axgrid = plt.subplots(nrow, ncol, figsize=(6 * ncol, 6.2 * nrow), squeeze=False)
    sim_axes = axgrid[-1]

    if args.raw:
        fontsize = max(2.5, 90.0 / args.size)
        for col, (name, fld, res) in enumerate(results):
            draw_raw_map(axgrid[0][col], fld, g.occ, fontsize)
            axgrid[0][col].set_title(f"{name}: raw priority map")

    scatters, trails = [], []
    for ax, (name, fld, res) in zip(sim_axes, results):
        ax.imshow(field_background(fld, g.occ, raw=False), cmap="viridis")
        # goals as stars
        gy = [gl[0] for gl in goals]
        gx = [gl[1] for gl in goals]
        ax.scatter(gx, gy, marker="*", s=220, c=colors, edgecolors="k", linewidths=0.6, zorder=4)
        # per-agent fading trail lines
        tl = [ax.plot([], [], "-", color=colors[i], lw=1.6, alpha=0.6, zorder=3)[0]
              for i in range(args.n_agents)]
        trails.append(tl)
        sc = ax.scatter([p[1] for p in starts], [p[0] for p in starts],
                        s=130, c=colors, edgecolors="k", linewidths=0.8, zorder=5)
        scatters.append(sc)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(name)

    fig.suptitle("", fontsize=13)
    axes = sim_axes

    def positions_at(res, t):
        log = res.positions_log
        return log[min(t, len(log) - 1)]

    def update(t):
        for ax, sc, tl, (name, fld, res) in zip(axes, scatters, trails, results):
            pos = positions_at(res, t)
            sc.set_offsets([[p[1], p[0]] for p in pos])
            if args.trail > 0:
                lo = max(0, t - args.trail)
                hist = [positions_at(res, k) for k in range(lo, t + 1)]
                for i in range(args.n_agents):
                    tl[i].set_data([h[i][1] for h in hist], [h[i][0] for h in hist])
            done = "solved" if res.success and t >= res.makespan else f"{res.n_reached}/{args.n_agents} home"
            ax.set_title(f"{name}  ·  step {min(t, len(res.positions_log)-1)}  ·  {done}")
        fig.suptitle(f"{args.map} maze · {args.n_agents} agents", fontsize=13)
        return scatters

    anim = FuncAnimation(fig, update, frames=T, interval=1000 / args.fps, blit=False)

    if args.live:
        plt.show()
    else:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        anim.save(args.out, writer=PillowWriter(fps=args.fps))
        print(f"saved {args.out}  ({T} frames)")
        for name, _, res in results:
            print(f"  {name:18s} success={res.success}  makespan={res.makespan}  "
                  f"flowtime={res.flowtime}")


if __name__ == "__main__":
    main()
