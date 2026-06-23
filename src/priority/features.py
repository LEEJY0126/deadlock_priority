"""Input feature maps for the learned priority field.

All channels are derived from *globally known* information only (the static map
and the set of goals, which the paper assumes every agent knows). Crucially we
do NOT use live agent positions, so the resulting field is identical for every
agent and can be computed offline -- preserving the communication-free property.
"""
from __future__ import annotations

import numpy as np

from ..envs.grid import GridMap

N_CHANNELS = 5


def build_features(gmap: GridMap, goals=None) -> np.ndarray:
    """Return (C, H, W) float32 feature stack.

    Channels:
      0 free mask (1 = free)
      1 clearance, normalized
      2 goal heatmap (count of goals per cell, blurred), normalized
      3 row coordinate, normalized to [0,1]
      4 col coordinate, normalized to [0,1]
    """
    H, W = gmap.H, gmap.W
    occ = gmap.occ
    free = (occ == 0).astype(np.float32)

    clr = gmap.clearance().astype(np.float32)
    clr = clr / (clr.max() + 1e-6)

    goal_hm = np.zeros((H, W), dtype=np.float32)
    if goals:
        for (r, c) in goals:
            goal_hm[r, c] += 1.0
        goal_hm = _blur(goal_hm)
        goal_hm = goal_hm / (goal_hm.max() + 1e-6)

    rr = np.linspace(0, 1, H, dtype=np.float32)[:, None] * np.ones((1, W), np.float32)
    cc = np.linspace(0, 1, W, dtype=np.float32)[None, :] * np.ones((H, 1), np.float32)

    feats = np.stack([free, clr * free, goal_hm * free, rr * free, cc * free], axis=0)
    return feats


def _blur(x, k=1):
    """Cheap box blur so a goal influences nearby cells."""
    out = x.copy()
    for _ in range(2):
        p = np.pad(out, k, mode="constant")
        acc = np.zeros_like(out)
        for dr in range(-k, k + 1):
            for dc in range(-k, k + 1):
                acc += p[k + dr:k + dr + out.shape[0], k + dc:k + dc + out.shape[1]]
        out = acc / ((2 * k + 1) ** 2)
    return out
