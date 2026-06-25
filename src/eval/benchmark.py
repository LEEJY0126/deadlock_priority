"""Held-out benchmark: compare priority fields on fixed maps + start/goal sets.

A "field provider" is a callable gmap -> (H,W) field. We evaluate MST baseline
vs a learned model on identical instances so the comparison is apples-to-apples.
"""
from __future__ import annotations

from collections import defaultdict
import numpy as np

from ..envs.grid import maze, random_forest, sample_start_goals
from ..envs.simulator import Simulator
from ..priority.mst_baseline import mst_priority_field


def make_eval_maps(n_per_kind=10, size=21, seed=12345):
    rng = np.random.default_rng(seed)
    maps = []
    for kind in ("forest", "wide", "narrow"):
        for _ in range(n_per_kind):
            if kind == "forest":
                g = random_forest(size, size, n_obstacles=int(size * size * 0.07), rng=rng)
            elif kind == "wide":
                g = maze(size, size, corridor=2, rng=rng)
            else:
                g = maze(size, size, corridor=1, rng=rng)
            maps.append((kind, g))
    return maps


def make_instances(maps, n_agents=8, n_inst=4, seed=999):
    """Fixed start/goal instances per map, shared across methods."""
    rng = np.random.default_rng(seed)
    inst = []
    for kind, g in maps:
        sg = [sample_start_goals(g, n_agents, rng=rng, min_sep=4) for _ in range(n_inst)]
        inst.append((kind, g, sg))
    return inst


def evaluate(field_provider, instances, max_steps=400, yield_mode="paper"):
    """Return per-kind aggregate metrics for a field provider.

    ``yield_mode`` selects the PIBT deadlock-resolution behavior for the rollouts
    (``"paper"`` = right-hand rule + livelock; ``"beta"`` = legacy boost)."""
    agg = defaultdict(lambda: {"succ": 0, "n": 0, "makespan": [], "flowtime": []})
    for kind, g, sg in instances:
        field = field_provider(g)
        for starts, goals in sg:
            res = Simulator(g, starts, goals, max_steps=max_steps,
                            yield_mode=yield_mode).run(field)
            a = agg[kind]
            a["n"] += 1
            a["succ"] += res.success
            if res.success:
                a["makespan"].append(res.makespan)
                a["flowtime"].append(res.flowtime)
    out = {}
    for kind, a in agg.items():
        out[kind] = {
            "success_rate": a["succ"] / a["n"],
            "makespan": float(np.mean(a["makespan"])) if a["makespan"] else float("nan"),
            "flowtime": float(np.mean(a["flowtime"])) if a["flowtime"] else float("nan"),
            "n": a["n"],
        }
    return out


def baseline_provider(gmap):
    return mst_priority_field(gmap)


def print_report(name, report):
    print(f"== {name} ==")
    for kind in ("forest", "wide", "narrow"):
        if kind in report:
            r = report[kind]
            print(f"  {kind:12s} success={r['success_rate']*100:5.1f}%  "
                  f"makespan={r['makespan']:6.1f}  flowtime={r['flowtime']:7.1f}  (n={r['n']})")
