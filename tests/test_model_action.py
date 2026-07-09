"""Shapes / wiring for the dynamic action-map model."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.envs.grid import maze, sample_start_goals
from src.priority.features import build_features
from src.priority.model import build_model, load_model
from src.priority.model_action import (EmbeddingActionModel, N_ACTIONS,
                                       agent_action_logits, action_field_fn,
                                       greedy_action_fn)


def _model(history=4):
    return build_model("embedding_action", dim=32, enc_depth=1, dec_depth=1,
                       heads=2, history=history, hist_dim=16, hist_depth=1)


def _inputs(g, model, n=4, seed=0):
    rng = np.random.default_rng(seed)
    feats = torch.from_numpy(build_features(g))[None]
    occ = torch.zeros(1, g.H, g.W)
    pos = [tuple(p) for p in np.argwhere(g.occ == 0)[:n]]
    for (r, c) in pos:
        occ[0, r, c] = 1.0
    hist = occ[:, None].repeat(1, model.config["history"], 1, 1)
    return feats, occ, hist, pos


def test_build_returns_action_model():
    m = _model()
    assert isinstance(m, EmbeddingActionModel)
    assert m.config["n_actions"] == N_ACTIONS


def test_forward_shape_is_5_channels():
    g = maze(11, 11, rng=np.random.default_rng(1))
    m = _model()
    feats, occ, hist, _ = _inputs(g, m)
    logits = m(feats, occ, hist)
    assert logits.shape == (1, N_ACTIONS, g.H, g.W)


def test_encode_decode_matches_forward():
    g = maze(11, 11, rng=np.random.default_rng(2))
    m = _model().eval()
    feats, occ, hist, _ = _inputs(g, m)
    with torch.no_grad():
        a = m(feats, occ, hist)
        b = m.decode(m.encode(feats), occ, hist)
    assert torch.allclose(a, b, atol=1e-6)


def test_agent_action_logits_shape_and_softmax():
    g = maze(11, 11, rng=np.random.default_rng(3))
    m = _model().eval()
    feats, occ, hist, pos = _inputs(g, m, n=5)
    with torch.no_grad():
        field = m.decode(m.encode(feats), occ, hist)[0]  # (5, H, W)
    la = agent_action_logits(field, pos)                 # (N, 5)
    assert la.shape == (len(pos), N_ACTIONS)
    probs = torch.softmax(torch.from_numpy(la), dim=1).numpy()
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_greedy_action_fn_returns_valid_indices():
    g = maze(13, 13, rng=np.random.default_rng(4))
    m = _model().eval()
    starts, goals = sample_start_goals(g, 4, rng=np.random.default_rng(4), min_sep=3)
    afn = greedy_action_fn(m, g, goals)
    acts = afn(starts, 0)
    assert acts.shape == (4,)
    assert acts.min() >= 0 and acts.max() < N_ACTIONS


def test_checkpoint_round_trip(tmp_path):
    g = maze(11, 11, rng=np.random.default_rng(5))
    m = _model().eval()
    feats, occ, hist, _ = _inputs(g, m)
    with torch.no_grad():
        before = m(feats, occ, hist)
    path = tmp_path / "action.pt"
    torch.save({"arch": "embedding_action", "config": m.config,
                "model": m.state_dict()}, path)
    m2 = load_model(str(path)).eval()
    assert isinstance(m2, EmbeddingActionModel)
    with torch.no_grad():
        after = m2(feats, occ, hist)
    assert torch.allclose(before, after, atol=1e-6)
