"""Mechanics checks for per-step A2C training of the embedding model.

Fast/deterministic: verifies one update runs, losses are finite, and gradient
flows into both the actor's map encoder and the critic. Learning-quality (reward
goes up) is exercised separately by the training script, not here.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.envs.grid import random_forest, sample_start_goals
from src.priority.model import build_model
from src.train.reward import DEFAULT_STEP_WEIGHTS, StepRewardWeights
from src.train.rl_embedding import (Critic, collect_episode, compute_gae,
                                    make_critic, train_embedding_episode,
                                    train_embedding_ppo_step)


def _setup(seed=4):
    rng = np.random.default_rng(seed)
    g = random_forest(14, 14, 14, rng=rng)
    starts, goals = sample_start_goals(g, 6, rng=rng, min_sep=3)
    torch.manual_seed(0)
    model = build_model("embedding", dim=32, enc_depth=1, dec_depth=1)
    critic = make_critic(model)
    return g, starts, goals, model, critic


def test_make_critic_shapes():
    _, _, _, model, critic = _setup()
    assert isinstance(critic, Critic)
    emb = torch.zeros(2, model.config["dim"], 14, 14)
    occ = torch.zeros(2, 14, 14)
    v = critic(emb, occ)
    assert v.shape == (2,)


def test_one_update_finite_and_grads_flow():
    g, starts, goals, model, critic = _setup()
    opt = torch.optim.Adam(list(model.parameters()) + list(critic.parameters()),
                           lr=3e-4)
    st = train_embedding_episode(model, critic, g, starts, goals, opt,
                                 sigma=0.3, max_steps=60, oracle="paper",
                                 rng=np.random.default_rng(0))
    for k in ("loss", "policy_loss", "value_loss", "reward"):
        assert np.isfinite(st[k]), f"{k} not finite: {st[k]}"
    assert 0 <= st["n_reached"] <= len(starts)
    assert st["steps"] >= 1

    enc = any(p.grad is not None and torch.isfinite(p.grad).all()
              for p in model.map_encoder.parameters())
    dec = any(p.grad is not None and torch.isfinite(p.grad).all()
              for p in model.priority_decoder.parameters())
    crit = any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in critic.parameters())
    assert enc and dec and crit, "gradient must reach encoder, decoder, and critic"


def test_params_change_after_updates():
    g, starts, goals, model, critic = _setup()
    opt = torch.optim.Adam(list(model.parameters()) + list(critic.parameters()),
                           lr=1e-3)
    before = [p.detach().clone() for p in model.parameters()]
    rng = np.random.default_rng(1)
    for _ in range(3):
        train_embedding_episode(model, critic, g, starts, goals, opt,
                                sigma=0.4, max_steps=60, oracle="paper", rng=rng)
    changed = any(not torch.allclose(b, p)
                  for b, p in zip(before, model.parameters()))
    assert changed, "policy parameters should update"


def test_gae_shapes_and_return_identity():
    g, starts, goals, model, critic = _setup()
    b = collect_episode(model, critic, g, starts, goals, sigma=0.4, max_steps=60,
                        weights=DEFAULT_STEP_WEIGHTS, oracle="paper",
                        device="cpu", rng=np.random.default_rng(0))
    compute_gae(b, model, critic, gamma=0.99, lam=0.95, device="cpu")
    T = len(b)
    assert b.advantages.shape == (T,) and b.returns.shape == (T,)
    assert np.isfinite(b.advantages).all() and np.isfinite(b.returns).all()
    # returns == advantages + old baseline values (definitional)
    assert np.allclose(b.returns, b.advantages + b.old_values, atol=1e-4)


def test_ppo_step_updates_and_diagnostics():
    g, starts, goals, model, critic = _setup()
    opt = torch.optim.Adam(list(model.parameters()) + list(critic.parameters()),
                           lr=3e-4)
    insts = [(g, starts, goals)]
    rng = np.random.default_rng(0)
    g2, s2, gl2, _, _ = _setup(seed=5)
    insts.append((g2, s2, gl2))

    before = [p.detach().clone() for p in model.parameters()]
    st = train_embedding_ppo_step(model, critic, insts, opt, sigma=0.4, epochs=3,
                                  clip=0.2, max_steps=60, oracle="paper", rng=rng)
    for k in ("policy_loss", "value_loss", "approx_kl", "clip_frac", "reward"):
        assert np.isfinite(st[k]), f"{k} not finite"
    assert 0.0 <= st["clip_frac"] <= 1.0
    assert 0.0 <= st["success_rate"] <= 1.0
    changed = any(not torch.allclose(b, p)
                  for b, p in zip(before, model.parameters()))
    assert changed, "PPO update must move policy params"


def test_collect_episode_rejects_zero_sigma():
    g, starts, goals, model, critic = _setup()
    try:
        collect_episode(model, critic, g, starts, goals, sigma=0.0, max_steps=20,
                        weights=DEFAULT_STEP_WEIGHTS, oracle="paper",
                        device="cpu", rng=np.random.default_rng(0))
        assert False, "expected ValueError for sigma=0"
    except ValueError:
        pass


def test_reward_weights_load_partial(tmp_path):
    p = tmp_path / "w.yaml"
    p.write_text("progress: 2.0\ntime_penalty: 0.05\n")
    w = StepRewardWeights.load(str(p))
    assert w.progress == 2.0 and w.time_penalty == 0.05
    assert w.success == StepRewardWeights().success  # unspecified -> default


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn) and "tmp_path" not in fn.__code__.co_varnames:
            fn()
            print(f"{name}: OK")
