"""Input feature maps for the learned priority field.

All channels are derived from the *static map* only. Crucially we do NOT use live
agent positions, so the resulting field is identical for every agent and can be
computed offline -- preserving the communication-free property. Goal positions
are also excluded (the goal-heatmap channel was removed): the map embedding is
purely structural, and agents still navigate to their goals via the simulator's
per-agent goal-distance fields.
"""
from __future__ import annotations

import numpy as np

from ..envs.grid import GridMap

N_CHANNELS = 4


def build_features(gmap: GridMap, goals=None) -> np.ndarray:
    """Return (C, H, W) float32 feature stack.

    Channels:
      0 free mask (1 = free)
      1 clearance, normalized
      2 row coordinate, normalized to [0,1]
      3 col coordinate, normalized to [0,1]

    ``goals`` is accepted for call-site compatibility but **no longer used** --
    the goal-heatmap channel was removed, so the priority field no longer sees
    goal positions.
    """
    H, W = gmap.H, gmap.W
    occ = gmap.occ
    free = (occ == 0).astype(np.float32)

    clr = gmap.clearance().astype(np.float32)
    clr = clr / (clr.max() + 1e-6)

    rr = np.linspace(0, 1, H, dtype=np.float32)[:, None] * np.ones((1, W), np.float32)
    cc = np.linspace(0, 1, W, dtype=np.float32)[None, :] * np.ones((H, 1), np.float32)

    feats = np.stack([free, clr * free, rr * free, cc * free], axis=0)
    return feats
