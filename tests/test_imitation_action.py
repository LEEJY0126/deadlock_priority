"""Behavioral-cloning helpers for the action-map policy."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch

from src.envs.action_exec import ACTION_MOVES
from src.envs.grid import GridMap, maze, sample_start_goals
from src.priority.model import build_model
from src.train.imitation_action import (action_from_delta, actions_from_log,
                                        expert_positions_log, il_episode_loss)

UP, DOWN, LEFT, RIGHT, STAY = 0, 1, 2, 3, 4


def _model(history=4):
    return build_model("embedding_action", dim=32, enc_depth=1, dec_depth=1,
                       heads=2, history=history, hist_dim=16, hist_depth=1)


def test_action_from_delta_all_moves():
    for i, d in enumerate(ACTION_MOVES):
        assert action_from_delta(d) == i
    assert action_from_delta((-1, 0)) == UP and action_from_delta((0, 1)) == RIGHT
    assert action_from_delta((0, 0)) == STAY


def test_action_from_delta_rejects_non_unit():
    for bad in [(2, 0), (1, 1), (-1, 1)]:
        with pytest.raises(ValueError):
            action_from_delta(bad)


def test_actions_from_log_recovers_moves():
    # 2 agents: one goes RIGHT then STAY, the other goes UP then DOWN
    log = [[(3, 3), (5, 5)],
           [(3, 4), (4, 5)],
           [(3, 4), (5, 5)]]
    labels = actions_from_log(log)
    assert len(labels) == 2
    assert labels[0].tolist() == [RIGHT, UP]
    assert labels[1].tolist() == [STAY, DOWN]


def test_il_episode_loss_shapes_and_backward():
    g = maze(11, 11, corridor=1, braid=0.25, rng=np.random.default_rng(0))
    starts, goals = sample_start_goals(g, 4, rng=np.random.default_rng(0), min_sep=3)
    log = expert_positions_log(None, g, starts, goals, max_steps=30)  # MST expert
    assert len(log) >= 2
    labels = actions_from_log(log)

    model = _model()
    loss, nc, nt = il_episode_loss(model, g.occ, log, device="cpu", labels=labels)
    assert torch.isfinite(loss)
    assert nt == sum(len(step) for step in labels)      # steps x agents
    assert 0 <= nc <= nt

    before = [p.detach().clone() for p in model.parameters()]
    loss.backward()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    opt.step()
    after = [p.detach().clone() for p in model.parameters()]
    assert any(not torch.equal(a, b) for a, b in zip(after, before))


def test_il_loss_decreases_when_overfitting_one_episode():
    g = maze(11, 11, corridor=1, braid=0.25, rng=np.random.default_rng(1))
    starts, goals = sample_start_goals(g, 3, rng=np.random.default_rng(1), min_sep=3)
    log = expert_positions_log(None, g, starts, goals, max_steps=40)
    labels = actions_from_log(log)
    model = _model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = None
    for step in range(20):
        loss, _, _ = il_episode_loss(model, g.occ, log, labels=labels)
        cur = loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
        if first is None:
            first = cur
    assert cur < first                # cloning a single episode should drive CE down


def test_empty_log_is_safe():
    model = _model()
    g = maze(9, 9, rng=np.random.default_rng(2))
    loss, nc, nt = il_episode_loss(model, g.occ, [[(1, 1)]], device="cpu")
    assert nt == 0 and nc == 0 and float(loss) == 0.0
