"""Grid world: occupancy map, 4-connected graph, BFS distances, and map generators.

A cell is free (0) or obstacle (1). Agents occupy free cells and move on the
4-connected lattice (plus a wait/stay action). This is the discrete abstraction
the paper's subgoal planner runs on (grid space G = (V, E)).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import numpy as np

# 4-connected moves plus stay. Order matters only as a stable default.
MOVES = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass
class GridMap:
    """Static occupancy grid. occ[r, c] == 1 means obstacle."""

    occ: np.ndarray  # (H, W) uint8

    @property
    def H(self) -> int:
        return self.occ.shape[0]

    @property
    def W(self) -> int:
        return self.occ.shape[1]

    def free(self, rc) -> bool:
        r, c = rc
        return 0 <= r < self.H and 0 <= c < self.W and self.occ[r, c] == 0

    def free_cells(self):
        rs, cs = np.where(self.occ == 0)
        return list(zip(rs.tolist(), cs.tolist()))

    def neighbors(self, rc):
        """Free neighbors reachable by a single 4-connected step (excludes stay)."""
        r, c = rc
        out = []
        for dr, dc in MOVES[1:]:
            nr, nc = r + dr, c + dc
            if self.free((nr, nc)):
                out.append((nr, nc))
        return out

    def bfs_dist(self, goal) -> np.ndarray:
        """Shortest-path distance (in steps) from every free cell to `goal`.

        Unreachable / obstacle cells are set to a large finite value so they can
        be used directly as a planning heuristic without inf-arithmetic.
        """
        INF = self.H * self.W + 1
        dist = np.full((self.H, self.W), INF, dtype=np.int32)
        gr, gc = goal
        dist[gr, gc] = 0
        q = deque([goal])
        while q:
            r, c = q.popleft()
            d = dist[r, c]
            for dr, dc in MOVES[1:]:
                nr, nc = r + dr, c + dc
                if self.free((nr, nc)) and dist[nr, nc] > d + 1:
                    dist[nr, nc] = d + 1
                    q.append((nr, nc))
        return dist

    def clearance(self) -> np.ndarray:
        """L1 distance from each free cell to the nearest obstacle/boundary.

        Larger == more open. Used both by the MST baseline (root selection) and
        as a model input feature. Boundary counts as obstacle.
        """
        INF = self.H * self.W + 1
        dist = np.full((self.H, self.W), INF, dtype=np.int32)
        q = deque()
        # Seed with obstacle cells and a virtual border ring (treated as walls).
        for r in range(self.H):
            for c in range(self.W):
                if self.occ[r, c] == 1:
                    dist[r, c] = 0
                    q.append((r, c))
                elif r == 0 or c == 0 or r == self.H - 1 or c == self.W - 1:
                    # border free cell is 1 step from the outside wall
                    dist[r, c] = 1
                    q.append((r, c))
        while q:
            r, c = q.popleft()
            for dr, dc in MOVES[1:]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.H and 0 <= nc < self.W and dist[nr, nc] > dist[r, c] + 1:
                    dist[nr, nc] = dist[r, c] + 1
                    q.append((nr, nc))
        dist[self.occ == 1] = 0
        return dist


# --------------------------------------------------------------------------- #
# Map generators
# --------------------------------------------------------------------------- #
def random_forest(H=24, W=24, n_obstacles=40, max_block=2, rng=None) -> GridMap:
    """Scatter rectangular obstacle blocks (cf. paper's random-forest scenes)."""
    rng = rng or np.random.default_rng()
    occ = np.zeros((H, W), dtype=np.uint8)
    for _ in range(n_obstacles):
        r = rng.integers(1, H - 1)
        c = rng.integers(1, W - 1)
        bh = rng.integers(1, max_block + 1)
        bw = rng.integers(1, max_block + 1)
        occ[r:min(r + bh, H - 1), c:min(c + bw, W - 1)] = 1
    return GridMap(occ)


def maze(H=25, W=25, corridor=1, braid=0.25, rng=None) -> GridMap:
    """Recursive-division maze with corridors `corridor` cells wide.

    corridor=1 -> narrow maze (one agent at a time), larger -> wider corridors,
    matching the paper's narrow/wide maze distinction.

    `braid` in [0,1] removes that fraction of interior walls to create loops and
    passing places. A perfect (tree) maze leaves no room for two agents to pass
    in a 1-wide corridor; braiding makes congested mazes hard-but-solvable.
    """
    rng = rng or np.random.default_rng()
    occ = np.zeros((H, W), dtype=np.uint8)
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = 1

    def divide(r0, c0, r1, c1):
        h = r1 - r0
        w = c1 - c0
        if h < 2 * corridor + 1 or w < 2 * corridor + 1:
            return
        horizontal = h > w if h != w else rng.random() < 0.5
        if horizontal:
            # wall row at an even offset, passage at an odd offset
            wr = rng.integers(r0 + corridor, r1 - corridor)
            occ[wr, c0:c1] = 1
            pc = rng.integers(c0, c1)
            occ[wr, pc:pc + corridor] = 0
            divide(r0, c0, wr, c1)
            divide(wr + 1, c0, r1, c1)
        else:
            wc = rng.integers(c0 + corridor, c1 - corridor)
            occ[r0:r1, wc] = 1
            pr = rng.integers(r0, r1)
            occ[pr:pr + corridor, wc] = 0
            divide(r0, c0, r1, wc)
            divide(r0, wc + 1, r1, c1)

    divide(1, 1, H - 1, W - 1)

    # Braiding: remove interior walls to introduce loops / alcoves.
    if braid > 0:
        wall_cells = [(r, c) for r in range(2, H - 2) for c in range(2, W - 2)
                      if occ[r, c] == 1]
        rng.shuffle(wall_cells)
        n_remove = int(braid * len(wall_cells))
        removed = 0
        for (r, c) in wall_cells:
            if removed >= n_remove:
                break
            # only remove if it joins free space without opening a 2x2+ room
            free_nb = sum(occ[r + dr, c + dc] == 0 for dr, dc in MOVES[1:])
            if free_nb >= 2:
                occ[r, c] = 0
                removed += 1
    return GridMap(occ)


def sample_start_goals(gmap: GridMap, n_agents: int, rng=None, min_sep=3):
    """Sample distinct, reachable start/goal pairs.

    Goals are mutually distinct and starts are mutually distinct (PIBT/grid
    abstraction requires one agent per cell). We retry to keep start != goal and
    a minimum start-goal separation so episodes are non-trivial.
    """
    rng = rng or np.random.default_rng()
    cells = gmap.free_cells()
    if len(cells) < 2 * n_agents:
        raise ValueError("Not enough free cells for requested agents")
    idx = rng.permutation(len(cells))
    cells = [cells[i] for i in idx]
    starts, goals, used = [], [], set()
    pool = deque(cells)
    while len(starts) < n_agents and len(pool) >= 2:
        s = pool.popleft()
        if s in used:
            continue
        # find a goal far enough away
        g = None
        for cand in list(pool):
            if cand not in used and abs(cand[0] - s[0]) + abs(cand[1] - s[1]) >= min_sep:
                g = cand
                break
        if g is None:
            continue
        pool.remove(g)
        starts.append(s)
        goals.append(g)
        used.add(s)
        used.add(g)
    if len(starts) < n_agents:
        raise RuntimeError("Could not sample enough start/goal pairs; relax min_sep")
    return starts, goals
