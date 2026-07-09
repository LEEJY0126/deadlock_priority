"""Mechanics for the action-map PPO training path."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.envs.grid import maze, sample_start_goals
from src.priority.model import build_model
from src.train.reward import StepRewardWeights
from src.priority.features import build_features
from src.priority.model_embedding import occ_history_window
from src.train.rl_action import (make_critic, collect_episode_action,
                                 collect_batch_action, compute_gae_action,
                                 ppo_update_action, train_action_ppo_step,
                                 _agent_logits, _replay_tensors)
from torch.distributions import Categorical


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


def _instances(n=4, n_maps=3, seed=0):
    rng = np.random.default_rng(seed)
    inst = []
    for _ in range(n_maps):
        g = maze(11, 11, corridor=1, braid=0.25, rng=rng)
        s, gl = sample_start_goals(g, n, rng=rng, min_sep=3)
        inst.append((g, s, gl))
    return inst


def test_collect_batch_shapes_and_validity():
    _, _, _, model, critic, _ = _setup()
    inst = _instances(n=4, n_maps=5)
    torch.manual_seed(0)
    batch = collect_batch_action(model, critic, inst, weights=StepRewardWeights(),
                                 max_steps=25, device="cpu")
    assert len(batch) == len(inst)
    for b in batch:
        T = len(b)
        assert T >= 1
        assert len(b.cells) == T and len(b.actions) == T
        assert b.old_logps.shape == (T,) and b.old_values.shape == (T,)
        assert b.rewards.shape == (T,)
        assert b.final_occ.shape == inst[0][0].occ.shape
        for a in b.actions:
            assert a.shape == (4,) and a.min() >= 0 and a.max() < model.config["n_actions"]


def test_batch_collection_matches_reference_decode():
    """The vectorized gather must reproduce the single-decode logp/value exactly."""
    _, _, _, model, critic, _ = _setup()
    model.eval(); critic.eval()
    inst = _instances(n=4, n_maps=4)
    torch.manual_seed(1)
    batch = collect_batch_action(model, critic, inst, weights=StepRewardWeights(),
                                 max_steps=20, device="cpu")
    history = model.config["history"]
    with torch.no_grad():
        for b, (g, _, _) in zip(batch, inst):
            emb = model.encode(torch.from_numpy(build_features(g))[None])
            for t in range(len(b)):
                occ = torch.from_numpy(b.occs[t])[None]
                win = occ_history_window(b.occs, t, history)
                hist = torch.from_numpy(win)[None]
                logits = model.decode(emb, occ, hist)[0]
                la = _agent_logits(logits, b.cells[t])
                action = torch.from_numpy(b.actions[t])
                logp = float(Categorical(logits=la).log_prob(action).sum())
                value = float(critic(emb, occ)[0])
                assert abs(logp - b.old_logps[t]) < 1e-4
                assert abs(value - b.old_values[t]) < 1e-4


def test_update_replay_batched_matches_perstep():
    """The batched T-step decode in ppo_update_action must equal per-step decode."""
    _, _, _, model, critic, _ = _setup()
    model.eval(); critic.eval()
    inst = _instances(n=4, n_maps=3)
    torch.manual_seed(2)
    batch = collect_batch_action(model, critic, inst, weights=StepRewardWeights(),
                                 max_steps=20, device="cpu")
    for b in batch:
        compute_gae_action(b, model, critic, 0.99, 0.95, "cpu")
    history = model.config["history"]
    with torch.no_grad():
        for b in batch:
            T = len(b)
            p = _replay_tensors(b, history, 0.0, 1.0, "cpu")
            emb = model.encode(p["feats"])
            emb_T = emb.expand(T, -1, -1, -1)
            logits = model.decode(emb_T, p["occ"], p["hist"])          # (T,5,H,W)
            ar = torch.arange(T)[:, None]
            la = logits.permute(0, 2, 3, 1)[ar, p["rows"], p["cols"]]  # (T,N,5)
            blogp = Categorical(logits=la).log_prob(p["act"]).sum(dim=1)
            bval = critic(emb_T.detach(), p["occ"])
            for t in range(T):
                lt = model.decode(emb, p["occ"][t:t + 1], p["hist"][t:t + 1])[0]
                lat = _agent_logits(lt, b.cells[t])
                logp_t = Categorical(logits=lat).log_prob(
                    torch.from_numpy(b.actions[t])).sum()
                val_t = critic(emb.detach(), p["occ"][t:t + 1])[0]
                assert abs(float(blogp[t]) - float(logp_t)) < 1e-4
                assert abs(float(bval[t]) - float(val_t)) < 1e-4


def test_vectorized_and_reference_paths_both_train():
    for vec in (True, False):
        _, _, _, model, critic, rng = _setup()
        opt = torch.optim.Adam(list(model.parameters()) + list(critic.parameters()),
                               lr=1e-3)
        before = [p.detach().clone() for p in model.parameters()]
        st = train_action_ppo_step(model, critic, _instances(n=4, n_maps=4), opt,
                                   max_steps=20, epochs=2, device="cpu", rng=rng,
                                   vectorized=vec)
        after = [p.detach().clone() for p in model.parameters()]
        assert any(not torch.equal(a, b) for a, b in zip(after, before))
        assert 0.0 <= st["collision_rate"] <= 1.0


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
