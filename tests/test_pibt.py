"""Correctness checks: PIBT must never produce vertex or swap conflicts."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.envs.grid import maze, random_forest, sample_start_goals
from src.envs.simulator import Simulator
from src.priority.mst_baseline import mst_priority_field


def _check_step_valid(prev, nxt, gmap):
    # one cell per agent (no vertex conflict)
    assert len(set(nxt)) == len(nxt), f"vertex conflict: {nxt}"
    # each move is stay or a single free step
    for a, b in zip(prev, nxt):
        assert b == a or b in gmap.neighbors(a), f"illegal move {a}->{b}"
    # no head-on swap
    pos = {a: i for i, a in enumerate(prev)}
    for i, (a, b) in enumerate(zip(prev, nxt)):
        if b in pos and b != a:
            j = pos[b]
            assert nxt[j] != a, f"swap conflict between {i} and {j}"


def test_pibt_conflict_free():
    rng = np.random.default_rng(0)
    for seed in range(8):
        g = (maze(21, 21, corridor=1, rng=np.random.default_rng(seed)) if seed % 2
             else random_forest(20, 20, 25, rng=np.random.default_rng(seed)))
        starts, goals = sample_start_goals(g, 8, rng=rng, min_sep=3)
        field = mst_priority_field(g)
        sim = Simulator(g, starts, goals, max_steps=120, log_positions=True)
        res = sim.run(field)
        log = res.positions_log
        for t in range(len(log) - 1):
            _check_step_valid(log[t], log[t + 1], g)
    print("PIBT conflict-free over all sampled episodes: OK")


if __name__ == "__main__":
    test_pibt_conflict_free()
