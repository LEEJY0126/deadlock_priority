"""Animated side-by-side episode: MST priority vs the *dynamic* embedding priority.

Like `simulate.py`, but for the embedding model whose priority field is
recomputed every step from the live occupancy. The MST panel's background is a
static field; the embedding panel's background **animates** — you watch the
priority field morph as agents move and the model re-decides who-yields-to-whom.

  python scripts/simulate_embedding.py --ckpt runs/rl_embedding_best.pt --map narrow
  python scripts/simulate_embedding.py --ckpt runs/rl_embedding_best.pt --map narrow --raw --live
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib

from src.envs.grid import maze, random_forest, sample_start_goals
from src.envs.simulator import Simulator, oracle_kwargs, ORACLES
from src.priority.mst_baseline import mst_priority_field
from src.priority.model import load_model, predict_field
from src.priority.model_embedding import EmbeddingPriorityModel, embedding_field_fn


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


def _fmt(v):
    return f"{v:.0f}" if abs(v - round(v)) < 0.05 else f"{v:.1f}"


def make_raw_panel(ax, field, occ, fontsize):
    """Raw-priority-map subplot: raw values as color + per-cell labels.

    Returns (imshow, {(r,c): text}) so a *dynamic* field can be updated in place.
    """
    im = ax.imshow(field_background(field, occ, raw=True), cmap="viridis")
    texts = {}
    H, W = occ.shape
    for r in range(H):
        for c in range(W):
            if occ[r, c] != 0:
                continue
            texts[(r, c)] = ax.text(c, r, _fmt(field[r, c]), ha="center",
                                    va="center", fontsize=fontsize, color="w")
    ax.set_xticks([]); ax.set_yticks([])
    return im, texts


def update_raw_panel(im, texts, field, occ):
    im.set_data(field_background(field, occ, raw=True))
    for (r, c), t in texts.items():
        t.set_text(_fmt(field[r, c]))


def run_static(g, starts, goals, field, max_steps, oracle):
    sim = Simulator(g, starts, goals, max_steps=max_steps, log_positions=True,
                    **oracle_kwargs(oracle))
    return sim.run(field), None


def run_dynamic(g, starts, goals, model, max_steps, oracle, device):
    """Run the embedding model, recording the per-step field the sim consumed.

    `fields_log[t]` is the field computed from `positions_log[t]` (call/step are
    aligned 1:1), so it is the field 'in effect' while the agents sit at step t.
    """
    fields_log = []
    base = embedding_field_fn(model, g, goals, device=device)

    def rec(positions):
        f = base(positions)
        fields_log.append(f.copy())
        return f

    sim = Simulator(g, starts, goals, max_steps=max_steps, log_positions=True,
                    **oracle_kwargs(oracle))
    res = sim.run(field_fn=rec)
    return res, fields_log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/rl_embedding_best.pt",
                    help="embedding checkpoint (falls back to static if not embedding)")
    ap.add_argument("--map", choices=["forest", "wide", "narrow"], default="narrow")
    ap.add_argument("--size", type=int, default=21)
    ap.add_argument("--n_agents", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/sim_embedding.gif")
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--trail", type=int, default=8, help="trail length in steps (0=off)")
    ap.add_argument("--raw", action="store_true",
                    help="add a top row of raw-priority maps (values labeled; the "
                         "embedding row animates)")
    ap.add_argument("--live", action="store_true", help="show a window instead of saving")
    ap.add_argument("--oracle", choices=ORACLES, default="paper",
                    help="PIBT resolution mode for the simulated episodes: paper "
                         "(right-hand rule + livelock), beta (legacy boost), or "
                         "goal-livelock (paper + goal-livelock retreat)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if not args.live:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    rng = np.random.default_rng(args.seed)
    g = make_map(args.map, args.size, rng)
    starts, goals = sample_start_goals(g, args.n_agents, rng=rng, min_sep=4)

    # Panel 0: static MST. Panel 1: the checkpoint (dynamic if embedding).
    mst_res, _ = run_static(g, starts, goals, mst_priority_field(g), args.max_steps, args.oracle)
    panels = [{"name": "MST priority", "dynamic": False,
               "field": mst_priority_field(g), "fields_log": None, "res": mst_res}]

    if os.path.exists(args.ckpt):
        model = load_model(args.ckpt, device=args.device)
        if isinstance(model, EmbeddingPriorityModel):
            res, flog = run_dynamic(g, starts, goals, model, args.max_steps,
                                    args.oracle, args.device)
            panels.append({"name": "embedding priority (dynamic)", "dynamic": True,
                           "field": flog[0], "fields_log": flog, "res": res})
        else:  # non-embedding ckpt -> static learned field, like simulate.py
            fld = predict_field(model, g, device=args.device)
            res, _ = run_static(g, starts, goals, fld, args.max_steps, args.oracle)
            panels.append({"name": "learned priority", "dynamic": False,
                           "field": fld, "fields_log": None, "res": res})
    else:
        print(f"[warn] {args.ckpt} not found -- showing MST only")

    T = max(len(p["res"].positions_log) for p in panels)
    colors = plt.cm.hsv(np.linspace(0, 1, args.n_agents, endpoint=False))
    ncol = len(panels)
    nrow = 2 if args.raw else 1
    fig, axgrid = plt.subplots(nrow, ncol, figsize=(6 * ncol, 6.2 * nrow), squeeze=False)
    sim_axes = axgrid[-1]

    # optional top row: raw-priority maps (embedding panel keeps handles to animate)
    if args.raw:
        fontsize = max(2.5, 90.0 / args.size)
        for col, p in enumerate(panels):
            im, texts = make_raw_panel(axgrid[0][col], p["field"], g.occ, fontsize)
            p["raw_im"], p["raw_texts"] = im, texts
            axgrid[0][col].set_title(f"{p['name']}: raw priority map")

    def frame_field(p, t):
        """The field to draw for panel `p` at frame `t` (clamped for dynamic)."""
        if not p["dynamic"]:
            return p["field"]
        flog = p["fields_log"]
        return flog[min(t, len(flog) - 1)]

    # bottom row: the animated simulation. z-scored background (fixed clim so the
    # dynamic panel does not flicker frame-to-frame), goals as stars, agents+trails.
    for ax, p in zip(sim_axes, panels):
        p["sim_im"] = ax.imshow(field_background(frame_field(p, 0), g.occ),
                                cmap="viridis", vmin=-2.5, vmax=2.5)
        ax.scatter([gl[1] for gl in goals], [gl[0] for gl in goals], marker="*",
                   s=220, c=colors, edgecolors="k", linewidths=0.6, zorder=4)
        p["trails"] = [ax.plot([], [], "-", color=colors[i], lw=1.6, alpha=0.6,
                               zorder=3)[0] for i in range(args.n_agents)]
        p["scatter"] = ax.scatter([q[1] for q in starts], [q[0] for q in starts],
                                  s=130, c=colors, edgecolors="k", linewidths=0.8, zorder=5)
        p["ax"] = ax
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(p["name"])

    def positions_at(res, t):
        log = res.positions_log
        return log[min(t, len(log) - 1)]

    def update(t):
        for p in panels:
            res = p["res"]
            pos = positions_at(res, t)
            p["scatter"].set_offsets([[q[1], q[0]] for q in pos])
            if p["dynamic"]:  # animate the priority-field background
                p["sim_im"].set_data(field_background(frame_field(p, t), g.occ))
                if args.raw:
                    update_raw_panel(p["raw_im"], p["raw_texts"], frame_field(p, t), g.occ)
            if args.trail > 0:
                lo = max(0, t - args.trail)
                hist = [positions_at(res, k) for k in range(lo, t + 1)]
                for i in range(args.n_agents):
                    p["trails"][i].set_data([h[i][1] for h in hist], [h[i][0] for h in hist])
            done = ("solved" if res.success and t >= res.makespan
                    else f"{res.n_reached}/{args.n_agents} home")
            step = min(t, len(res.positions_log) - 1)
            p["ax"].set_title(f"{p['name']}  ·  step {step}  ·  {done}")
        fig.suptitle(f"{args.map} maze · {args.n_agents} agents", fontsize=13)
        return [p["scatter"] for p in panels]

    anim = FuncAnimation(fig, update, frames=T, interval=1000 / args.fps, blit=False)

    if args.live:
        plt.show()
    else:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        anim.save(args.out, writer=PillowWriter(fps=args.fps))
        print(f"saved {args.out}  ({T} frames)")
        for p in panels:
            r = p["res"]
            print(f"  {p['name']:28s} success={r.success}  makespan={r.makespan}  "
                  f"flowtime={r.flowtime}")


if __name__ == "__main__":
    main()
