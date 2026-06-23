"""PIBT: Priority Inheritance with Backtracking (Okumura et al., 2022).

Single-step prioritized MAPF. Given each agent's current cell, goal-distance
heuristic, and a *priority value*, compute the next configuration with no vertex
or swap conflicts. Higher priority value => plans first / inherits.

This is the MAPF solver the paper feeds with the position-based priority `rho_i`
(runMAPF, line 10 of Alg. 3). We make the priority an explicit input so we can
swap in either the MST baseline field or a learned field.
"""
from __future__ import annotations

import numpy as np

from .grid import GridMap


class PIBT:
    def __init__(self, gmap: GridMap, goal_dist: list[np.ndarray]):
        """`goal_dist[i]` is the BFS distance field to agent i's goal."""
        self.gmap = gmap
        self.goal_dist = goal_dist
        self.n = len(goal_dist)

    def _candidates(self, i, cur):
        """Next-cell candidates for agent i, sorted by distance-to-goal (asc)."""
        r, c = cur
        cands = self.gmap.neighbors(cur) + [cur]
        dist = self.goal_dist[i]
        # small deterministic tie-break on coordinates keeps runs reproducible
        cands.sort(key=lambda u: (dist[u[0], u[1]], u[0], u[1]))
        return cands

    def step(self, positions, priorities, rng=None):
        """Advance one timestep.

        positions: list[(r,c)] current cell per agent.
        priorities: array[n] float; higher plans first (ties broken by index).
        Returns list[(r,c)] next cell per agent (conflict-free).
        """
        n = self.n
        cur = list(positions)
        occupied_now = {cur[i]: i for i in range(n)}
        occupied_next: dict[tuple, int] = {}
        nxt = [None] * n

        def func_pibt(i, caller=None):
            # caller.now is forbidden to avoid head-on swaps
            forbidden = cur[caller] if caller is not None else None
            for u in self._candidates(i, cur[i]):
                if u in occupied_next:
                    continue
                if u == forbidden:
                    continue
                k = occupied_now.get(u)
                if k == i:
                    k = None  # staying in place is fine
                occupied_next[u] = i
                nxt[i] = u
                if k is not None and nxt[k] is None:
                    if not func_pibt(k, caller=i):
                        # displaced agent couldn't move; revert this choice
                        del occupied_next[u]
                        nxt[i] = None
                        continue
                return True
            # fallback: forced to stay (may still conflict if its cell is taken,
            # but as last resort we keep it put and let the caller backtrack)
            if cur[i] not in occupied_next:
                occupied_next[cur[i]] = i
                nxt[i] = cur[i]
                return True
            nxt[i] = cur[i]
            return False

        order = sorted(range(n), key=lambda i: (-priorities[i], i))
        for i in order:
            if nxt[i] is None:
                func_pibt(i)
        return [nxt[i] if nxt[i] is not None else cur[i] for i in range(n)]
