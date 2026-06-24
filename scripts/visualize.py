"""Render the original map next to MST vs learned priority fields.

For each sample map the figure shows three columns: the raw obstacle map, the
MST priority field, and the learned priority field. Priority cells are annotated
with their (raw) priority value so you can read absolute levels, while the cell
color is per-map normalized so the MST and learned patterns are visually
comparable despite living on different scales.

Run after training: python scripts/visualize.py --ckpt runs/rl.pt
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.envs.grid import maze, random_forest
from src.priority.mst_baseline import mst_priority_field
from src.priority.model import load_model, predict_field


def norm(f, occ):
    """Per-map z-score over free cells; obstacles -> NaN (render blank)."""
    free = occ == 0
    v = f.copy().astype(float)
    if free.sum():
        m, s = v[free].mean(), v[free].std() + 1e-6
        v = (v - m) / s
    v[~free] = np.nan
    return v


def fmt(v):
    """Compact label: integer if near-integer, else one decimal."""
    return f"{v:.0f}" if abs(v - round(v)) < 0.05 else f"{v:.1f}"


def annotate(ax, raw, normed, occ, fontsize):
    """Write the raw priority value in each free cell."""
    H, W = occ.shape
    for y in range(H):
        for x in range(W):
            if occ[y, x] != 0:
                continue
            # dark (low) background -> white text, bright (high) -> black text
            color = "white" if (np.nan_to_num(normed[y, x]) < 0.25) else "black"
            ax.text(x, y, fmt(raw[y, x]), ha="center", va="center",
                    fontsize=fontsize, color=color)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/rl.pt")
    ap.add_argument("--out", default="runs/fields.png")
    ap.add_argument("--size", type=int, default=21, help="grid side length")
    ap.add_argument("--no_numbers", action="store_true", help="skip per-cell priority labels")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    S = args.size
    rng = np.random.default_rng(7)
    maps = [("forest", random_forest(S, S, max(1, S * S // 15), rng=rng)),
            ("wide", maze(S, S, corridor=2, rng=rng)),
            ("narrow", maze(S, S, corridor=1, rng=rng))]

    model = None
    if os.path.exists(args.ckpt):
        model = load_model(args.ckpt, device=args.device)

    annotate_on = not args.no_numbers
    fontsize = max(2.5, 90.0 / S)  # shrink labels as the grid grows

    fig, axes = plt.subplots(len(maps), 3, figsize=(15, 5 * len(maps)))
    if len(maps) == 1:
        axes = axes[None, :]
    for i, (name, g) in enumerate(maps):
        # column 0: original obstacle map (obstacles dark, free white)
        axes[i, 0].imshow(g.occ, cmap="gray_r", vmin=0, vmax=1)
        axes[i, 0].set_title(f"{name}: map")

        # column 1: MST priority field
        mst_raw = mst_priority_field(g)
        mst_n = norm(mst_raw, g.occ)
        axes[i, 1].imshow(mst_n, cmap="viridis")
        axes[i, 1].set_title(f"{name}: MST priority")
        if annotate_on:
            annotate(axes[i, 1], mst_raw, mst_n, g.occ, fontsize)

        # column 2: learned priority field
        if model is not None:
            learned_raw = predict_field(model, g, device=args.device)
            learned_n = norm(learned_raw, g.occ)
            axes[i, 2].imshow(learned_n, cmap="viridis")
            axes[i, 2].set_title(f"{name}: learned priority")
            if annotate_on:
                annotate(axes[i, 2], learned_raw, learned_n, g.occ, fontsize)
        axes[i, 2].set_title(axes[i, 2].get_title() or f"{name}: learned priority")

        for c in range(3):
            axes[i, c].axis("off")
    plt.tight_layout()
    plt.savefig(args.out, dpi=130)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
