"""Mechanics for the action-map PPO training path."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.envs.grid import maze, sample_start_goals
from src.priority.model import build_model
from src.train.reward import StepRewardWeights
from src.train.rl_action import (make_critic, collect_episode_action,
                                 compute_gae_action, ppo_update_action,
                                 train_action_ppo_step)


def _setup(n=4, seed=0):
    rng = np.random.default_rng(seed)
    g = maze(11, 11, corridor=1, braid=0.25, rng=rng)
    starts, goals = sample_start_goals(g, n, rng=rng, min_sep=3)
    model = build_model("embedding_action", dim=32, enc_depth=1, dec_depth=1,
                        heads=2, history=4, hist_dim=16, hist_depth=1)
    critic = make_critic(model)
    return g, starts, goals, model, critic, rng


def test_collect_records_are_aligned():
    g, starts, goals, model, critic, rng = _setup()
    b = collect_episode_action(model, critic, g, starts, goals,
                               weights=StepRewardWeights(), max_steps=20,
                               device="cpu", rng=rng)
    T = len(b)
    assert T >= 1
    assert len(b.cells) == T and len(b.actions) == T
    assert b.old_logps.shape == (T,) and b.old_values.shape == (T,)
    assert b.rewards.shape == (T,)
    # every recorded action is a valid index for every agent
    for a in b.actions:
        assert a.shape == (len(starts),)
        assert a.min() >= 0 and a.max() < model.config["n_actions"]


def test_collision_applies_terminal_penalty():
    g, starts, goals, model, critic, rng = _setup()
    w = StepRewardWeights(collision=-5.0)
    b = collect_episode_action(model, critic, g, starts, goals, weights=w,
                               max_steps=40, device="cpu", rng=rng)
    if b.collided:
        # the terminal reward carries the collision penalty (progress/time are small)
        assert b.rewards[-1] < 0
    assert np.isfinite(b.rewards).all()


def test_compute_gae_shapes_and_bootstrap():
    g, starts, goals, model, critic, rng = _setup()
    b = collect_episode_action(model, critic, g, starts, goals,
                               weights=StepRewardWeights(), max_steps=20,
                               device="cpu", rng=rng)
    compute_gae_action(b, model, critic, gamma=0.99, lam=0.95, device="cpu")
    T = len(b)
    assert b.advantages.shape == (T,) and b.returns.shape == (T,)
    assert np.isfinite(b.advantages).all() and np.isfinite(b.returns).all()


def test_ppo_update_runs_and_steps():
    g, starts, goals, model, critic, rng = _setup()
    opt = torch.optim.Adam(list(model.parameters()) + list(critic.parameters()),
                           lr=1e-3)
    before = [p.detach().clone() for p in model.parameters()]
    batch = []
    for _ in range(3):
        b = collect_episode_action(model, critic, g, starts, goals,
                                   weights=StepRewardWeights(), max_steps=20,
                                   device="cpu", rng=rng)
        compute_gae_action(b, model, critic, gamma=0.99, lam=0.95, device="cpu")
        batch.append(b)
    out = ppo_update_action(model, critic, opt, batch, epochs=2, entropy_coef=0.01,
                            device="cpu")
    assert np.isfinite(out["policy_loss"]) and np.isfinite(out["value_loss"])
    assert out["entropy"] > 0.0                # categorical entropy is positive
    assert out["epochs_run"] >= 1
    after = [p.detach().clone() for p in model.parameters()]
    assert any(not torch.equal(a, b0) for a, b0 in zip(after, before))


def test_train_step_reports_rates():
    g, starts, goals, model, critic, rng = _setup()
    opt = torch.optim.Adam(list(model.parameters()) + list(critic.parameters()),
                           lr=1e-3)
    inst = [(g, starts, goals)]
    st = train_action_ppo_step(model, critic, inst, opt, max_steps=20, epochs=2,
                               device="cpu", rng=rng)
    for k in ("success_rate", "frac_reached", "collision_rate", "reward",
              "entropy", "approx_kl"):
        assert k in st
    assert 0.0 <= st["collision_rate"] <= 1.0
