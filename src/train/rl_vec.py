"""GPU-vectorized reward evaluation for RL (GPU_vectorized branch).

Drop-in alternative to the per-(b,k) CPU `episode_reward` loop in rl.py: batches
all B*K*n_samples rollouts into one VecSim call. Returns the same (B,K) reward
matrix using the same reward formula, so the REINFORCE math is unchanged.

NOTE: VecSim is the non-backtracking approximation of PIBT (see vec_sim.py), so
training with engine="vec" optimizes the field for a *different* solver than the
PIBT-based evaluate.py. Always validate transfer back to PIBT.
"""
from __future__ import annotations

import numpy as np

from ..envs.vec_sim import build_batch, VecSim
from .reward import DEFAULT_WEIGHTS


def vec_rewards(maps, samples_list, fields_np, n_agents, max_steps, device,
                weights=DEFAULT_WEIGHTS):
    """Mean reward over samples for each (map, field).

    maps:         list[GridMap], length B
    samples_list: list over b of list of (starts, goals) -- the shared instances
    fields_np:    list over b of (K, H, W) positive priority fields
    returns:      (B, K) reward matrix
    """
    B = len(maps)
    K = fields_np[0].shape[0]
    entries, layout = [], []
    for b in range(B):
        for k in range(K):
            for (starts, goals) in samples_list[b]:
                entries.append((maps[b], starts, goals, fields_np[b][k]))
                layout.append((b, k))

    out = VecSim(build_batch(entries, device=device), max_steps=max_steps).run()
    r = weights.episode(out["success"].astype(np.float64),
                        out["makespan"], out["flowtime"], n_agents, max_steps)

    rbk = np.zeros((B, K))
    cnt = np.zeros((B, K))
    for i, (b, k) in enumerate(layout):
        rbk[b, k] += r[i]
        cnt[b, k] += 1
    return rbk / np.maximum(cnt, 1)
