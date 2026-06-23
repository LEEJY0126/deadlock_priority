"""Position-priority field via MST (the paper's heuristic, Sec. IV-C.1, Eq. 12).

Build a spanning tree of the free grid rooted in a wide-open cell, then assign a
*position priority* to every node: equal to the parent's priority if the two lie
in a common 4-cycle (a fully-free 2x2 block => "wide area"), otherwise parent+1
("narrow passage"). The resulting field is identical for every agent given the
known map, which is what makes it usable without communication.

Returned as a dense (H, W) float array; obstacle cells are 0.
"""
from __future__ import annotations

from collections import deque
import numpy as np

from ..envs.grid import GridMap, MOVES


def _edge_in_4cycle(occ: np.ndarray, a, b) -> bool:
    """True if adjacent free cells a, b sit in a fully-free 2x2 block."""
    H, W = occ.shape
    (r1, c1), (r2, c2) = a, b
    if r1 == r2:  # horizontal edge, columns c1,c2
        c = min(c1, c2)
        r = r1
        for r0 in (r - 1, r):  # two candidate 2x2 squares share this edge
            rows = (r0, r0 + 1)
            cols = (c, c + 1)
            if all(0 <= rr < H and 0 <= cc < W and occ[rr, cc] == 0
                   for rr in rows for cc in cols):
                return True
    else:  # vertical edge, rows r1,r2
        r = min(r1, r2)
        c = c1
        for c0 in (c - 1, c):
            rows = (r, r + 1)
            cols = (c0, c0 + 1)
            if all(0 <= rr < H and 0 <= cc < W and occ[rr, cc] == 0
                   for rr in rows for cc in cols):
                return True
    return False


def select_root(gmap: GridMap):
    """Pick the most open free cell (max clearance) as the tree root."""
    clr = gmap.clearance()
    clr = clr.astype(np.int32)
    clr[gmap.occ == 1] = -1
    idx = int(np.argmax(clr))
    return (idx // gmap.W, idx % gmap.W)


def mst_priority_field(gmap: GridMap, root=None) -> np.ndarray:
    """Compute the position-priority field (Eq. 12)."""
    occ = gmap.occ
    H, W = occ.shape
    if root is None:
        root = select_root(gmap)

    # BFS spanning tree from root (unit weights => BFS tree is an MST).
    parent = {}
    rho = np.zeros((H, W), dtype=np.float32)
    visited = np.zeros((H, W), dtype=bool)
    rho[root] = 1.0
    visited[root] = True
    q = deque([root])
    while q:
        v = q.popleft()
        for dr, dc in MOVES[1:]:
            u = (v[0] + dr, v[1] + dc)
            if not gmap.free(u) or visited[u]:
                continue
            visited[u] = True
            parent[u] = v
            rho[u] = rho[v] if _edge_in_4cycle(occ, v, u) else rho[v] + 1.0
            q.append(u)
    return rho
