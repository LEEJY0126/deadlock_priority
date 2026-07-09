"""Direct, collision-aware execution for the action-map policy (no PIBT).

The priority pipeline (:mod:`src.envs.simulator`) hands a priority field to PIBT,
which *guarantees* collision-free moves. The action-map policy instead emits a
per-agent move directly, so collisions become possible and are the model's
responsibility. This module applies those moves and detects collisions.

A collision at a step is any of:

  * **wall**   -- an agent's next cell is an obstacle or out of bounds.
  * **vertex** -- two agents' next cells coincide.
  * **swap**   -- two adjacent agents exchange cells in the same step
                  (i -> j's cell while j -> i's cell).

Plain *following* -- moving into a cell another agent vacates the same step -- is
allowed (no vertex conflict at the next configuration, no swap). When a collision
occurs the episode terminates immediately and is marked failed (the penalty is
applied by the RL reward, not here). Moves are **not** masked to legal ones: the
policy must learn to avoid walls and other agents.

Action channel order is fixed to ``[UP, DOWN, LEFT, RIGHT, STAY]`` (matching
:mod:`src.priority.model_action`); :data:`ACTION_MOVES` holds the deltas.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# [UP, DOWN, LEFT, RIGHT, STAY] -- fixed order shared with model_action.
ACTION_MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1), (0, 0))
ACTION_NAMES = ("UP", "DOWN", "LEFT", "RIGHT", "STAY")


@dataclass
class ActionEpisodeResult:
    success: bool
    makespan: int          # step at which all agents were simultaneously at goal
    flowtime: int          # sum of first-arrival times (unreached = max_steps)
    n_reached: int         # agents at goal at the final step
    steps: int             # steps actually simulated (== action_fn calls)
    collided: bool         # episode ended on a collision
    collision_step: int    # 1-based step of the collision (0 if none)
    collision_reason: str  # "wall" | "vertex" | "swap" | "" (none)
    positions_log: list = field(default_factory=list)


def intended_cells(positions, actions):
    """Apply one action per agent -> list of intended next cells (r, c).

    Cells are **not** clamped or validated here; wall / OOB moves are surfaced by
    :func:`find_collision`. ``actions`` is any indexable of action indices.
    """
    out = []
    for (r, c), a in zip(positions, actions):
        dr, dc = ACTION_MOVES[int(a)]
        out.append((r + dr, c + dc))
    return out


def find_collision(gmap, positions, actions, nxt=None):
    """Return the collision reason for one joint move, or ``None`` if legal.

    ``nxt`` may be precomputed :func:`intended_cells`; otherwise it is derived.
    Checks, in order, wall/OOB, vertex (shared next cell), then swap.
    """
    nxt = nxt if nxt is not None else intended_cells(positions, actions)
    n = len(positions)
    # wall / out-of-bounds
    for cell in nxt:
        if not gmap.free(cell):
            return "wall"
    # vertex: two agents want the same next cell
    seen = {}
    for i, cell in enumerate(nxt):
        if cell in seen:
            return "vertex"
        seen[cell] = i
    # swap: adjacent agents exchange cells (i -> pos[j], j -> pos[i])
    pos_index = {p: i for i, p in enumerate(positions)}
    for i in range(n):
        j = pos_index.get(nxt[i])
        if j is not None and j != i and nxt[j] == positions[i]:
            return "swap"
    return None


def run_action_episode(gmap, starts, goals, action_fn, max_steps=256):
    """Run one episode by applying per-agent actions directly (no PIBT).

    ``action_fn(positions, t) -> iterable[N]`` returns an action index per agent
    for the current step (``t`` is 0-based). Each call is one recorded step. On a
    collision the episode stops immediately (failed); otherwise the joint move is
    applied. Arrived agents stay in play and remain collidable (success = all
    agents simultaneously on goals). Returns an :class:`ActionEpisodeResult` whose
    ``positions_log`` has ``steps + 1`` entries (start config first); on a
    collision the final entry repeats the pre-collision config (no move applied),
    so per-step reward assembly stays aligned with the action calls.
    """
    n = len(starts)
    assert len(goals) == n
    pos = list(starts)
    log = [list(pos)]
    arrival = [None] * n
    makespan = None
    collided = False
    collision_step = 0
    collision_reason = ""
    t = 0
    for t in range(1, max_steps + 1):
        actions = action_fn(pos, t - 1)
        nxt = intended_cells(pos, actions)
        reason = find_collision(gmap, pos, actions, nxt=nxt)
        if reason is not None:
            collided = True
            collision_step = t
            collision_reason = reason
            log.append(list(pos))   # no move applied; keep log aligned
            break
        pos = nxt
        log.append(list(pos))
        for i in range(n):
            if pos[i] == goals[i] and arrival[i] is None:
                arrival[i] = t
        if all(pos[i] == goals[i] for i in range(n)):
            makespan = t
            break

    success = makespan is not None
    n_reached = sum(pos[i] == goals[i] for i in range(n))
    flowtime = sum(a if a is not None else max_steps for a in arrival)
    steps = makespan if success else t
    return ActionEpisodeResult(
        success=success,
        makespan=makespan if success else max_steps,
        flowtime=flowtime,
        n_reached=n_reached,
        steps=steps,
        collided=collided,
        collision_step=collision_step,
        collision_reason=collision_reason,
        positions_log=log,
    )


__all__ = ["ACTION_MOVES", "ACTION_NAMES", "ActionEpisodeResult",
           "intended_cells", "find_collision", "run_action_episode"]
