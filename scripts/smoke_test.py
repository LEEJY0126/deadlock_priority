"""End-to-end smoke test: build maps, run the MST-baseline priority through PIBT."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.envs.grid import maze, random_forest, sample_start_goals
from src.envs.simulator import Simulator
from src.priority.mst_baseline import mst_priority_field, select_root


def trial(name, gmap, n_agents, seed):
    rng = np.random.default_rng(seed)
    starts, goals = sample_start_goals(gmap, n_agents, rng=rng, min_sep=4)
    field = mst_priority_field(gmap)
    sim = Simulator(gmap, starts, goals, max_steps=400)
    res = sim.run(field)
    print(f"  {name:16s} agents={n_agents:2d} "
          f"success={res.success!s:5s} makespan={res.makespan:3d} "
          f"flowtime={res.flowtime:4d} reached={res.n_reached}/{n_agents} "
          f"deadlock={res.deadlocked}")
    return res


def main():
    print("MST baseline smoke test")
    print(f"  prio field range demo (forest):")
    f = random_forest(20, 20, n_obstacles=25, rng=np.random.default_rng(0))
    fld = mst_priority_field(f)
    print(f"    root={select_root(f)} priority min/max="
          f"{fld[f.occ==0].min():.0f}/{fld[f.occ==0].max():.0f}")

    ok = 0
    total = 0
    for seed in range(5):
        forest = random_forest(20, 20, n_obstacles=25, rng=np.random.default_rng(seed))
        wmaze = maze(21, 21, corridor=2, rng=np.random.default_rng(seed))
        nmaze = maze(21, 21, corridor=1, rng=np.random.default_rng(seed))
        for name, g in [("forest", forest), ("wide-maze", wmaze), ("narrow-maze", nmaze)]:
            try:
                r = trial(f"{name}", g, n_agents=8, seed=seed)
                ok += r.success
                total += 1
            except Exception as e:
                print(f"  {name}: ERROR {e}")
    print(f"\n  baseline success: {ok}/{total}")


if __name__ == "__main__":
    main()
