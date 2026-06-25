"""Search oracle for imitation labels.

For a given map we score a bank of candidate priority fields by simulating PIBT
over several start/goal samples, and keep the best one. The candidate bank is
built from MST fields rooted at different cells (different roots induce different
corridor orderings) plus a couple of geometry-based fields. The winning field is
the imitation target the CNN learns to reproduce -- a cheap stand-in for the
intractable "optimal position-priority field".
"""
from __future__ import annotations

import numpy as np

from ..envs.grid import GridMap, sample_start_goals
from ..envs.simulator import Simulator
from ..priority.mst_baseline import mst_priority_field


def candidate_roots(gmap: GridMap, k=8, rng=None):
    """Top-k clearance cells plus a few random free cells, as MST roots."""
    rng = rng or np.random.default_rng()
    clr = gmap.clearance().astype(np.float32)
    clr[gmap.occ == 1] = -1
    flat = np.argsort(clr.ravel())[::-1]
    roots = []
    for idx in flat[: k]:
        roots.append((idx // gmap.W, idx % gmap.W))
    free = gmap.free_cells()
    for _ in range(k):
        roots.append(free[rng.integers(len(free))])
    # de-dup preserving order
    seen, out = set(), []
    for r in roots:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def candidate_fields(gmap: GridMap, rng=None):
    """Yield (name, field) candidates."""
    fields = []
    for r in candidate_roots(gmap, rng=rng):
        fields.append((f"mst@{r}", mst_priority_field(gmap, root=r)))
    # geometry baselines
    clr = gmap.clearance().astype(np.float32) * (gmap.occ == 0)
    fields.append(("clearance", clr))
    fields.append(("neg_clearance", (clr.max() - clr) * (gmap.occ == 0)))
    return fields


def score_field(gmap, samples, field, max_steps=400, alpha=0.3):
    """Mean (success_rate, flowtime) of `field` over fixed start/goal samples."""
    succ, flow = 0, 0.0
    for starts, goals in samples:
        # training-label search pinned to the legacy beta engine (matches how the
        # shipped checkpoints were produced); eval/benchmark use paper-yield.
        sim = Simulator(gmap, starts, goals, max_steps=max_steps, alpha=alpha,
                        yield_mode="beta")
        res = sim.run(field)
        succ += res.success
        flow += res.flowtime
    n = len(samples)
    return succ / n, flow / n


def best_field(gmap: GridMap, n_agents=8, n_samples=4, seed=0, max_steps=400):
    """Return (best_field, info) for one map by searching the candidate bank."""
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_samples):
        s, g = sample_start_goals(gmap, n_agents, rng=rng, min_sep=4)
        samples.append((s, g))

    best = None
    results = []
    for name, fld in candidate_fields(gmap, rng=rng):
        sr, ft = score_field(gmap, samples, fld, max_steps=max_steps)
        results.append((name, sr, ft))
        key = (sr, -ft)  # maximise success, then minimise flowtime
        if best is None or key > best[0]:
            best = (key, name, fld)
    info = {"name": best[1], "success": best[0][0], "flowtime": -best[0][1],
            "all": results}
    return best[2].astype(np.float32), info
