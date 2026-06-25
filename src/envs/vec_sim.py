"""GPU-vectorized batched episode simulator (experimental, GPU_vectorized branch).

Runs E independent episodes in lockstep as torch tensors so the whole batch
steps on the GPU at once. This is the "vectorized env" approach (Isaac-Gym/Brax
style) for testing whether GPU batching beats CPU multiprocessing for our
rollouts.

IMPORTANT -- this is NOT PIBT. PIBT's priority *inheritance with backtracking* is
recursive and data-dependent, which does not vectorize cleanly. Here we use a
GPU-friendly approximation: a single-pass, priority-ordered reservation with a
head-on-swap fix-up. It respects the priority field's ordering and resolves
vertex conflicts exactly, but drops PIBT's inheritance/backtracking (so its
reachability guarantee). It is intended for throughput benchmarking and as a
fast, approximate rollout -- not as a drop-in replacement for the CPU solver.

Agent priority mirrors simulator.py: normalized field value at the current cell,
a small index tie-break, a stuck-time boost (beta), and yielding once arrived.
"""
from __future__ import annotations

import numpy as np
import torch

from .grid import GridMap, MOVES

INF = 1e9


def _neighbor_tables(occ: np.ndarray):
    """Geometry + per-cell move validity for one map.

    Returns nbr_idx (HW,5) flattened target cell per move (self if out of
    bounds) and nbr_valid (HW,5) whether that move lands on a free cell. Move 0
    is 'stay'.
    """
    H, W = occ.shape
    HW = H * W
    nbr_idx = np.zeros((HW, 5), dtype=np.int64)
    nbr_valid = np.zeros((HW, 5), dtype=bool)
    free = occ == 0
    for n in range(HW):
        r, c = divmod(n, W)
        if not free[r, c]:
            nbr_idx[n] = n
            continue
        for m, (dr, dc) in enumerate(MOVES):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and free[nr, nc]:
                nbr_idx[n, m] = nr * W + nc
                nbr_valid[n, m] = True
            else:
                nbr_idx[n, m] = n  # invalid -> point at self, marked invalid
    return nbr_idx, nbr_valid


def build_batch(entries, device="cuda"):
    """Stack a list of (gmap, starts, goals, field) into batched GPU tensors.

    All maps must share H, W (same grid size). starts/goals are lists of (r,c).
    `field` is an (H, W) priority field (raw; normalized here like simulator.py).

    Neighbor tables (per map) and BFS distance fields (per map+goals) are cached
    by object identity, so RL batches where K fields share an instance pay the
    CPU-side BFS/neighbor cost once per instance instead of once per env.
    """
    E = len(entries)
    g0 = entries[0][0]
    H, W = g0.H, g0.W
    HW = H * W
    A = len(entries[0][1])

    pos = np.zeros((E, A), np.int64)
    goal = np.zeros((E, A), np.int64)
    field = np.zeros((E, HW), np.float32)
    free = np.zeros((E, HW), bool)
    nbr_idx = np.zeros((E, HW, 5), np.int64)
    nbr_valid = np.zeros((E, HW, 5), bool)
    distA = np.full((E, A, HW), INF, np.float32)

    nbr_cache: dict = {}   # id(gmap) -> (ni, nv, free_row)
    dist_cache: dict = {}  # (id(gmap), goals_key) -> (A, HW) distances
    for e, (gmap, starts, goals, fld) in enumerate(entries):
        assert gmap.H == H and gmap.W == W, "all maps must share grid size"
        assert len(starts) == A and len(goals) == A
        key = id(gmap)
        if key not in nbr_cache:
            ni, nv = _neighbor_tables(gmap.occ)
            nbr_cache[key] = (ni, nv, (gmap.occ == 0).reshape(-1))
        ni, nv, fr = nbr_cache[key]
        nbr_idx[e], nbr_valid[e], free[e] = ni, nv, fr
        # normalize field over free cells (matches Simulator.run)
        vals = fld[gmap.occ == 0]
        f = (fld - vals.mean()) / (vals.std() + 1e-6)
        field[e] = (f.reshape(-1) * fr).astype(np.float32)
        gkey = (key, tuple(goals))
        if gkey not in dist_cache:
            dist_cache[gkey] = np.stack(
                [gmap.bfs_dist(gl).reshape(-1).astype(np.float32) for gl in goals])
        distA[e] = dist_cache[gkey]
        for a, (s, gl) in enumerate(zip(starts, goals)):
            pos[e, a] = s[0] * W + s[1]
            goal[e, a] = gl[0] * W + gl[1]

    t = lambda x: torch.from_numpy(x).to(device)
    return {
        "H": H, "W": W, "HW": HW, "A": A, "E": E,
        "pos": t(pos), "goal": t(goal), "field": t(field),
        "nbr_idx": t(nbr_idx), "nbr_valid": t(nbr_valid), "distA": t(distA),
        "device": device,
    }


class VecSim:
    """Batched stepper. One instance owns one batch of E episodes."""

    def __init__(self, batch, max_steps=256, alpha=0.3, beta=0.3):
        self.b = batch
        self.max_steps = max_steps
        self.alpha = alpha
        self.beta = beta

    def _sorted_candidates(self, pos):
        """For each (env, agent) return move-target cells sorted by goal distance.

        Returns cand (E,A,5) cell idx and valid (E,A,5), both reordered so the
        lowest-distance (best) candidate is first. Invalid moves sort last.
        """
        b = self.b
        E, A, HW = b["E"], b["A"], b["HW"]
        ar = torch.arange(E, device=b["device"])[:, None]
        # cells reachable from each agent's current cell (E,A,5)
        cand = b["nbr_idx"][ar, pos]
        valid = b["nbr_valid"][ar, pos]                               # (E,A,5)
        # distance-to-own-goal at each candidate cell
        d = b["distA"].gather(2, cand)                                 # (E,A,5)
        d = torch.where(valid, d, torch.full_like(d, INF))
        order = torch.argsort(d, dim=2)                               # best first
        cand = torch.gather(cand, 2, order)
        valid = torch.gather(valid, 2, order)
        return cand, valid

    def run(self):
        b = self.b
        dev = b["device"]
        E, A, HW = b["E"], b["A"], b["HW"]
        pos = b["pos"].clone()
        goal = b["goal"]
        ar = torch.arange(E, device=dev)

        arrival = torch.full((E, A), self.max_steps, dtype=torch.long, device=dev)
        done_env = torch.zeros(E, dtype=torch.bool, device=dev)
        makespan = torch.full((E,), self.max_steps, dtype=torch.long, device=dev)
        stuck = torch.zeros(E, A, device=dev)
        prev_g = b["distA"].gather(2, pos[:, :, None]).squeeze(2)      # (E,A)

        for t in range(1, self.max_steps + 1):
            arrived = pos == goal
            base = torch.gather(b["field"], 1, pos)                    # (E,A)
            idx_tb = torch.arange(A, device=dev).float()[None] / A
            prio = base + self.alpha * idx_tb + self.beta * stuck
            prio = torch.where(arrived, torch.full_like(prio, -INF), prio)
            # unique priority for deterministic arbitration (higher wins)
            prio_u = prio * 1e4 - torch.arange(A, device=dev).float()[None]

            cand, valid = self._sorted_candidates(pos)                # (E,A,5)
            order = torch.argsort(-prio_u, dim=1)                     # agents by prio desc

            reserved = torch.zeros(E, HW, dtype=torch.bool, device=dev)
            target = pos.clone()
            # process one priority rank at a time, vectorized across all envs
            for slot in range(A):
                a = order[:, slot]                                    # (E,)
                cE = cand[ar, a]                                      # (E,5) candidate cells
                vE = valid[ar, a]                                     # (E,5)
                taken = reserved.gather(1, cE)                        # (E,5)
                avail = vE & ~taken
                has = avail.any(dim=1)
                # first available column (argmax on bool picks first True)
                first = torch.argmax(avail.to(torch.int8), dim=1)     # (E,)
                cell = torch.where(has, cE[ar, first], pos[ar, a])    # fallback: stay
                target[ar, a] = cell
                reserved[ar, cell] = True

            # head-on swap fix: if a->cur[b] and b->cur[a], lower-prio stays
            tgt_i = target[:, :, None].expand(E, A, A)
            pos_j = pos[:, None, :].expand(E, A, A)
            swap = (tgt_i == pos_j) & (target[:, None, :].expand(E, A, A) ==
                                       pos[:, :, None].expand(E, A, A))
            eye = torch.eye(A, dtype=torch.bool, device=dev)[None]
            swap = swap & ~eye
            # for each agent, does it swap with a higher-prio partner?
            higher = prio_u[:, None, :] > prio_u[:, :, None]          # (E,A,A) j higher than i
            loses = (swap & higher).any(dim=2)                        # (E,A)
            target = torch.where(loses, pos, target)

            pos = target
            g = b["distA"].gather(2, pos[:, :, None]).squeeze(2)
            progressed = g < prev_g
            stuck = torch.where(arrived | progressed, torch.zeros_like(stuck), stuck + 1)
            prev_g = g

            now_arr = pos == goal
            newly = now_arr & (arrival == self.max_steps)
            arrival = torch.where(newly, torch.full_like(arrival, t), arrival)

            all_arr = now_arr.all(dim=1)
            just_done = all_arr & ~done_env
            makespan = torch.where(just_done, torch.full_like(makespan, t), makespan)
            done_env = done_env | all_arr
            if bool(done_env.all()):
                break

        success = done_env
        flowtime = arrival.sum(dim=1)
        return {
            "success": success.cpu().numpy(),
            "makespan": makespan.cpu().numpy(),
            "flowtime": flowtime.cpu().numpy(),
            "n_reached": (pos == goal).sum(dim=1).cpu().numpy(),
        }
