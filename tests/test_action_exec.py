"""Collision detection + direct execution for the action-map policy."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.envs.grid import GridMap
from src.envs.action_exec import (ACTION_MOVES, ACTION_NAMES, find_collision,
                                  intended_cells, run_action_episode)

# Action index constants for readability.
UP, DOWN, LEFT, RIGHT, STAY = 0, 1, 2, 3, 4


def open_room(H=7, W=7):
    """A walled border around an all-free interior."""
    occ = np.ones((H, W), dtype=np.int8)
    occ[1:-1, 1:-1] = 0
    return GridMap(occ)


def test_action_move_order_matches_names():
    assert ACTION_NAMES == ("UP", "DOWN", "LEFT", "RIGHT", "STAY")
    assert ACTION_MOVES[UP] == (-1, 0) and ACTION_MOVES[DOWN] == (1, 0)
    assert ACTION_MOVES[LEFT] == (0, -1) and ACTION_MOVES[RIGHT] == (0, 1)
    assert ACTION_MOVES[STAY] == (0, 0)


def test_intended_cells():
    g = open_room()
    nxt = intended_cells([(3, 3), (2, 2)], [RIGHT, DOWN])
    assert nxt == [(3, 4), (3, 2)]


def test_wall_collision():
    g = open_room()
    # (1,1) is free; moving UP lands on the border wall row 0.
    assert find_collision(g, [(1, 1)], [UP]) == "wall"
    # STAY stays on a free cell -> legal.
    assert find_collision(g, [(1, 1)], [STAY]) is None


def test_vertex_collision():
    g = open_room()
    # both target (3,3)
    assert find_collision(g, [(3, 2), (3, 4)], [RIGHT, LEFT]) == "vertex"


def test_swap_collision():
    g = open_room()
    # adjacent horizontal pair exchange cells
    assert find_collision(g, [(3, 3), (3, 4)], [RIGHT, LEFT]) == "swap"


def test_following_is_allowed():
    g = open_room()
    # a chain moving in the same direction: each enters the cell ahead as the
    # occupant vacates it -- no vertex conflict, no swap.
    pos = [(3, 1), (3, 2), (3, 3)]
    assert find_collision(g, pos, [RIGHT, RIGHT, RIGHT]) is None


def test_all_stay_is_legal():
    g = open_room()
    pos = [(2, 2), (3, 3), (4, 4)]
    assert find_collision(g, pos, [STAY, STAY, STAY]) is None


def test_run_episode_success():
    g = open_room()
    starts = [(1, 1)]
    goals = [(1, 3)]
    # scripted: move RIGHT twice then STAY
    plan = iter([[RIGHT], [RIGHT], [STAY], [STAY]])

    def action_fn(pos, t):
        return next(plan)

    res = run_action_episode(g, starts, goals, action_fn, max_steps=10)
    assert res.success and not res.collided
    assert res.makespan == 2 and res.n_reached == 1
    assert len(res.positions_log) == res.steps + 1


def test_run_episode_terminates_on_collision():
    g = open_room()
    starts = [(3, 3), (3, 4)]
    goals = [(3, 5), (3, 2)]

    def action_fn(pos, t):
        return [RIGHT, LEFT]  # immediate swap

    res = run_action_episode(g, starts, goals, action_fn, max_steps=10)
    assert res.collided and not res.success
    assert res.collision_reason == "swap" and res.collision_step == 1
    # final log entry repeats the pre-collision config (no move applied)
    assert res.positions_log[-1] == list(starts)
    assert len(res.positions_log) == res.steps + 1
