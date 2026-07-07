"""Held-out benchmark: compare priority fields on fixed maps + start/goal sets.

A "field provider" is a callable gmap -> (H,W) field. We evaluate MST baseline
vs a learned model on identical instances so the comparison is apples-to-apples.
"""
from __future__ import annotations

from collections import defaultdict
import numpy as np

from ..envs.grid import maze, random_forest, sample_start_goals
from ..envs.pibt import PIBT
from ..envs.simulator import EpisodeResult, Simulator, oracle_kwargs
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


def _new_agg():
    return defaultdict(lambda: {"succ": 0, "n": 0, "makespan": [], "flowtime": []})


def _record(agg, kind, res):
    a = agg[kind]
    a["n"] += 1
    a["succ"] += res.success
    if res.success:
        a["makespan"].append(res.makespan)
        a["flowtime"].append(res.flowtime)


def _finalize(agg):
    out = {}
    for kind, a in agg.items():
        out[kind] = {
            "success_rate": a["succ"] / a["n"],
            "makespan": float(np.mean(a["makespan"])) if a["makespan"] else float("nan"),
            "flowtime": float(np.mean(a["flowtime"])) if a["flowtime"] else float("nan"),
            "n": a["n"],
        }
    return out


def evaluate(field_provider, instances, max_steps=400, oracle="paper"):
    """Return per-kind aggregate metrics for a field provider.

    ``oracle`` selects the PIBT resolution behavior for the rollouts: ``"beta"``
    (legacy boost), ``"paper"`` (right-hand rule + livelock), or
    ``"goal-livelock"`` (paper + the opt-in goal-livelock retreat)."""
    agg = _new_agg()
    for kind, g, sg in instances:
        field = field_provider(g)
        for starts, goals in sg:
            res = Simulator(g, starts, goals, max_steps=max_steps,
                            **oracle_kwargs(oracle)).run(field)
            _record(agg, kind, res)
    return _finalize(agg)


def evaluate_embedding(model, instances, max_steps=400, oracle="paper",
                       device="cpu"):
    """Per-kind metrics for the dynamic embedding priority model.

    Unlike :func:`evaluate` (one static field per map), the embedding field is
    goal-conditioned and recomputed each step from live occupancy, so a fresh
    ``field_fn`` is built per (map, start/goal) instance and passed to
    ``Simulator.run(field_fn=...)``. Same instances, metrics, and oracle as
    :func:`evaluate`, so it is directly comparable to the MST/learned reports."""
    from ..priority.model_embedding import embedding_field_fn  # lazy: torch dep
    agg = _new_agg()
    for kind, g, sg in instances:
        for starts, goals in sg:
            field_fn = embedding_field_fn(model, g, goals, device=device)
            res = Simulator(g, starts, goals, max_steps=max_steps,
                            **oracle_kwargs(oracle)).run(field_fn=field_fn)
            _record(agg, kind, res)
    return _finalize(agg)


def run_elapsed_pibt(gmap, starts, goals, max_steps=400, stall_limit=None, rng=None):
    """Faithful PIBT baseline using mapf-IR's dynamic elapsed-time priority.

    Reproduces Kei18/mapf-IR ``pibt.cpp``: each step, agents are ordered by the
    lexicographic key ``(elapsed, init_d, tie_breaker)`` taken **highest-first**
    (mapf-IR pops a max-heap over that comparator, so the agent longest away from
    its goal plans first -- anti-starvation):

      * ``elapsed`` -- timesteps since the agent was last at its goal (reset to 0
        when its next cell is the goal, else +1), so it grows while travelling.
      * ``init_d``  -- initial start->goal BFS distance, fixed per episode.
      * ``tie_breaker`` -- a random float drawn once per agent.

    Unlike :class:`Simulator`, there is **no** position-priority field and **no**
    goal-livelock / right-hand-rule resolution -- this is plain PIBT, the
    canonical baseline to A/B against the MST and learned field providers. The
    priority is dynamic (per-agent, per-step) so it cannot be expressed as a
    static field and does not fit :func:`evaluate`'s provider interface.

    Returns an :class:`EpisodeResult` with the same metric semantics as
    ``Simulator.run`` (makespan, flowtime, success).
    """
    n = len(starts)
    goal_dist = [gmap.bfs_dist(g) for g in goals]
    pibt = PIBT(gmap, goal_dist)
    rng = np.random.default_rng(0) if rng is None else rng
    init_d = np.array([goal_dist[i][starts[i]] for i in range(n)], dtype=np.float64)
    tie = rng.random(n)                       # mapf-IR: fixed per agent at creation
    elapsed = np.zeros(n, dtype=np.float64)   # mapf-IR: all 0 on the first step
    pos = list(starts)
    arrival = [None] * n
    stall_limit = stall_limit or max(20, 4 * (gmap.H + gmap.W))

    def remaining():
        return sum(int(goal_dist[i][pos[i]]) for i in range(n))

    best_remaining, since_improve, makespan, t = remaining(), 0, None, 0
    for t in range(1, max_steps + 1):
        # (elapsed, init_d, tie) DESC == mapf-IR plan order (max-heap top first).
        order = sorted(range(n), key=lambda i: (elapsed[i], init_d[i], tie[i]),
                       reverse=True)
        # PIBT.step sorts by (-prio, index); unique ranks reproduce `order` exactly
        # (so the index tie-break never fires and the random `tie` decides ties).
        prio = np.empty(n, dtype=np.float64)
        for rank, i in enumerate(order):
            prio[i] = -rank
        pos = pibt.step(pos, prio)
        for i in range(n):                    # elapsed update (pibt.cpp): reset at goal
            elapsed[i] = 0.0 if pos[i] == goals[i] else elapsed[i] + 1.0
            if pos[i] == goals[i] and arrival[i] is None:
                arrival[i] = t
        if all(pos[i] == goals[i] for i in range(n)):
            makespan = t
            break
        rem = remaining()
        if rem < best_remaining:
            best_remaining, since_improve = rem, 0
        else:
            since_improve += 1
        if since_improve >= stall_limit:
            break  # deadlock / livelock: no progress for too long

    success = makespan is not None
    n_reached = sum(pos[i] == goals[i] for i in range(n))
    flowtime = sum(a if a is not None else max_steps for a in arrival)
    return EpisodeResult(
        success=success,
        makespan=makespan if success else max_steps,
        flowtime=flowtime,
        n_reached=n_reached,
        steps=makespan if success else t,
        deadlocked=not success,
    )


def evaluate_elapsed(instances, max_steps=400, seed=2024):
    """Per-kind metrics for the mapf-IR elapsed-time PIBT baseline (no field).

    Mirrors :func:`evaluate` but drives :func:`run_elapsed_pibt`. A single seeded
    RNG supplies each episode's random tie-breakers so the benchmark stays
    deterministic across runs (like the seeded eval instances)."""
    agg = _new_agg()
    rng = np.random.default_rng(seed)
    for kind, g, sg in instances:
        for starts, goals in sg:
            res = run_elapsed_pibt(g, starts, goals, max_steps=max_steps, rng=rng)
            _record(agg, kind, res)
    return _finalize(agg)


def baseline_provider(gmap):
    return mst_priority_field(gmap)


def print_report(name, report):
    print(f"== {name} ==")
    for kind in ("forest", "wide", "narrow"):
        if kind in report:
            r = report[kind]
            print(f"  {kind:12s} success={r['success_rate']*100:5.1f}%  "
                  f"makespan={r['makespan']:6.1f}  flowtime={r['flowtime']:7.1f}  (n={r['n']})")
