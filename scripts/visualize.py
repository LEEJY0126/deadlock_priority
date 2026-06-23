"""Render MST vs learned priority fields side by side for a few maps.

Saves a PNG so you can see *what* the network learned -- e.g. whether it raises
priority along corridor approaches or reshapes junction orderings relative to the
MST tree. Run after training: python scripts/visualize.py --ckpt runs/rl.pt
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
from src.priority.model import PriorityUNet, predict_field


def norm(f, occ):
    free = occ == 0
    v = f.copy().astype(float)
    if free.sum():
        m, s = v[free].mean(), v[free].std() + 1e-6
        v = (v - m) / s
    v[~free] = np.nan
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/rl.pt")
    ap.add_argument("--out", default="runs/fields.png")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    rng = np.random.default_rng(7)
    maps = [("forest", random_forest(21, 21, 30, rng=rng)),
            ("wide", maze(21, 21, corridor=2, rng=rng)),
            ("narrow", maze(21, 21, corridor=1, rng=rng))]

    model = None
    if os.path.exists(args.ckpt):
        model = PriorityUNet().to(args.device)
        model.load_state_dict(torch.load(args.ckpt, map_location=args.device)["model"])

    fig, axes = plt.subplots(len(maps), 2, figsize=(8, 12))
    for i, (name, g) in enumerate(maps):
        mst = norm(mst_priority_field(g), g.occ)
        axes[i, 0].imshow(mst, cmap="viridis")
        axes[i, 0].set_title(f"{name}: MST priority")
        axes[i, 0].axis("off")
        if model is not None:
            learned = norm(predict_field(model, g, device=args.device), g.occ)
            axes[i, 1].imshow(learned, cmap="viridis")
            axes[i, 1].set_title(f"{name}: learned priority")
        axes[i, 1].axis("off")
    plt.tight_layout()
    plt.savefig(args.out, dpi=110)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
