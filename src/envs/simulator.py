"""Episode runner: drives PIBT with a position-priority field and reports metrics.

The priority field (H, W) is the only thing that differs between the MST baseline
and a learned model. Agent priorities are assembled from the field exactly as in
the paper (Eq. 13): priority at the current node, a small position-based
tie-break that damps oscillation, and zero priority once an agent is at its goal.

Deadlock resolution (``yield_mode``):

- ``"paper"`` (default): the paper's explicit yield (Alg. 3 lines 11-12). When a
  lower-priority agent is stuck and blocked by a higher-priority neighbour
  (grid-adapted livelock condition, Eq. 18), its subgoal is reassigned to the
  lowest-priority adjacent node and it backs out to make room.
- ``"beta"`` (legacy): a PIBT anti-starvation boost that instead raises a stuck
  agent's priority so it pushes through. Kept for the training engine and A/B.

Either way the mechanism is applied identically to every method, so the priority
field remains the only variable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .grid import GridMap
from .pibt import PIBT


@dataclass
class EpisodeResult:
    success: bool
    makespan: int          # step at which all agents were simultaneously at goal
    flowtime: int          # sum of first-arrival times (unreached = max_steps)
    n_reached: int         # agents at goal at the final step
    steps: int             # steps actually simulated
    deadlocked: bool
    positions_log: list = field(default_factory=list)


class Simulator:
    def __init__(self, gmap: GridMap, starts, goals, max_steps=256,
                 alpha=0.3, beta=0.3, stall_limit=None, log_positions=False,
                 yield_mode="paper", yield_patience=1):
        assert len(starts) == len(goals)
        self.gmap = gmap
        self.starts = list(starts)
        self.goals = list(goals)
        self.n = len(starts)
        self.max_steps = max_steps
        self.alpha = alpha  # tie-break weight, in (0, 0.5) per the paper
        # yield_mode selects the deadlock-resolution mechanism (see module docs).
        assert yield_mode in ("paper", "beta")
        self.yield_mode = yield_mode
        # yield_patience: stuck steps before a blocked lower-priority agent backs
        # out (paper mode). Eq. 18 triggers on oscillation; a small patience
        # avoids yielding before PIBT's own backtracking gets a chance.
        self.yield_patience = yield_patience
        # beta: PIBT dynamic anti-starvation boost per stuck step (legacy mode).
        # Raises a stuck agent's priority so it pushes through, restoring PIBT
        # reachability in narrow corridors. Ignored when yield_mode == "paper".
        self.beta = beta  # default chosen as a gentle nudge (see run() scaling)
        self.stall_limit = stall_limit or max(20, 4 * (gmap.H + gmap.W))
        self.log_positions = log_positions
        self.goal_dist = [gmap.bfs_dist(g) for g in goals]
        self.pibt = PIBT(gmap, self.goal_dist)

    def _agent_priorities(self, pos, prev_base, arrived, stuck):
        """Assemble per-agent priority from the position-priority field (Eq. 13).

        In legacy ``beta`` mode a dynamic anti-starvation boost proportional to
        stuck time is added; in ``paper`` mode the boost is omitted (deadlocks are
        resolved by the explicit yield instead)."""
        n = self.n
        prio = np.zeros(n, dtype=np.float64)
        base = np.array([self.field[p[0], p[1]] for p in pos], dtype=np.float64)
        for i in range(n):
            if arrived[i]:
                prio[i] = 0.0  # reached goal -> yield
                continue
            # tie-break: penalise returning to a lower-priority node (anti-oscillation)
            di = (i / n) if base[i] >= prev_base[i] else (1.0 + i / n)
            prio[i] = base[i] + self.alpha * di
            if self.yield_mode == "beta":
                prio[i] += self.beta * stuck[i]
        return prio, base

    def _yielders(self, pos, prio, arrived, stuck):
        """Grid-adapted livelock detection (paper Eq. 18 / Alg. 3 line 11).

        An agent yields when it is (a) not at its goal, (b) stuck for at least
        ``yield_patience`` steps (no progress / oscillation proxy, Eq. 18a-b),
        and (c) adjacent to a higher-priority agent that is still en route
        (Eq. 18c-d). Such agents back out to their lowest-priority neighbour.
        Returns a boolean array over agents."""
        n = self.n
        yld = np.zeros(n, dtype=bool)
        posset = {pos[j]: j for j in range(n)}
        for i in range(n):
            if arrived[i] or stuck[i] < self.yield_patience:
                continue
            for u in self.gmap.neighbors(pos[i]):
                j = posset.get(u)
                if j is None or arrived[j]:
                    continue
                if prio[j] > prio[i]:  # blocked by a higher-priority neighbour
                    yld[i] = True
                    break
        return yld

    def run(self, priority_field: np.ndarray, rng=None) -> EpisodeResult:
        # Normalize the field to unit spread over free cells so the tie-break
        # (alpha) and stuck-boost (beta) act in comparable units regardless of
        # whether the field is the integer MST field or a learned softplus field.
        free = self.gmap.occ == 0
        vals = priority_field[free]
        std = vals.std() + 1e-6
        self.field = (priority_field - vals.mean()) / std * free
        pos = list(self.starts)
        prev_base = np.array([self.field[p[0], p[1]] for p in pos], dtype=np.float64)
        arrival = [None] * self.n
        log = [list(pos)] if self.log_positions else []

        def remaining():
            return sum(int(self.goal_dist[i][pos[i]]) for i in range(self.n))

        best_remaining = remaining()
        since_improve = 0
        makespan = None
        stuck = np.zeros(self.n, dtype=np.float64)
        prev_gdist = np.array([self.goal_dist[i][pos[i]] for i in range(self.n)])

        for t in range(1, self.max_steps + 1):
            arrived = [pos[i] == self.goals[i] for i in range(self.n)]
            prio, base = self._agent_priorities(pos, prev_base, arrived, stuck)
            # paper yield: stuck, blocked lower-priority agents route to their
            # lowest-priority neighbour (minimise the field) instead of the goal.
            cost = None
            if self.yield_mode == "paper":
                yld = self._yielders(pos, prio, arrived, stuck)
                if yld.any():
                    cost = [self.field if yld[i] else None for i in range(self.n)]
            pos = self.pibt.step(pos, prio, rng=rng, cost=cost)
            prev_base = np.array([self.field[p[0], p[1]] for p in pos])
            # update per-agent stuck counter (reset on progress toward own goal)
            gdist = np.array([self.goal_dist[i][pos[i]] for i in range(self.n)])
            for i in range(self.n):
                if arrived[i] or gdist[i] < prev_gdist[i]:
                    stuck[i] = 0.0
                else:
                    stuck[i] += 1.0
            prev_gdist = gdist
            if self.log_positions:
                log.append(list(pos))

            for i in range(self.n):
                if pos[i] == self.goals[i] and arrival[i] is None:
                    arrival[i] = t

            if all(pos[i] == self.goals[i] for i in range(self.n)):
                makespan = t
                break

            rem = remaining()
            if rem < best_remaining:
                best_remaining = rem
                since_improve = 0
            else:
                since_improve += 1
            if since_improve >= self.stall_limit:
                break  # deadlock / livelock: no progress for too long

        success = makespan is not None
        steps = makespan if success else t
        n_reached = sum(pos[i] == self.goals[i] for i in range(self.n))
        flowtime = sum(a if a is not None else self.max_steps for a in arrival)
        return EpisodeResult(
            success=success,
            makespan=makespan if success else self.max_steps,
            flowtime=flowtime,
            n_reached=n_reached,
            steps=steps,
            deadlocked=not success,
            positions_log=log,
        )
