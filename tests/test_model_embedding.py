"""Contract checks for the embedding-map priority architecture.

MapEncoder is map-only (comms-free invariant); PriorityDecoder turns the shared
embedding + live occupancy into a *dynamic* priority field that is read off per
agent by indexing.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.envs.grid import random_forest, sample_start_goals
from src.envs.simulator import Simulator
from src.priority.features import build_features
from src.priority.model import build_model
from src.priority.model_embedding import (
    EmbeddingPriorityModel,
    agent_priorities,
    embedding_field_fn,
    occ_history_window,
    predict_priority_field,
)


def _map_and_agents(seed=0, n_agents=8, H=24, W=24):
    rng = np.random.default_rng(seed)
    gmap = random_forest(H, W, n_obstacles=40, rng=rng)
    starts, goals = sample_start_goals(gmap, n_agents, rng=rng, min_sep=3)
    return gmap, starts, goals


def _occ(gmap, positions):
    occ = np.zeros((gmap.H, gmap.W), np.float32)
    for (r, c) in positions:
        occ[r, c] = 1.0
    return torch.from_numpy(occ)[None]


def _hist(model, occ):
    """[B, history, H, W] history tensor: the current frame repeated."""
    return occ[:, None].repeat(1, model.config["history"], 1, 1)


def test_build_model_dispatch():
    model = build_model("embedding", dim=32, enc_depth=1, dec_depth=1)
    assert isinstance(model, EmbeddingPriorityModel)


def test_encode_decode_shapes_and_positivity():
    torch.manual_seed(0)
    gmap, starts, goals = _map_and_agents()
    model = build_model("embedding", dim=32, enc_depth=1, dec_depth=1).eval()

    x = torch.from_numpy(build_features(gmap, goals))[None]
    emb = model.encode(x)
    assert emb.shape == (1, 32, gmap.H, gmap.W)

    occ = _occ(gmap, starts)
    field = model.decode(emb, occ, _hist(model, occ))
    assert field.shape == (1, gmap.H, gmap.W)
    assert bool((field >= 0).all()), "softplus output must be non-negative"


def test_encoder_is_map_only_decoder_is_dynamic():
    """The embedding must NOT depend on agent positions (comms-free), but the
    field MUST change when occupancy changes."""
    torch.manual_seed(0)
    gmap, starts, goals = _map_and_agents()
    model = build_model("embedding", dim=32, enc_depth=1, dec_depth=1).eval()

    x = torch.from_numpy(build_features(gmap, goals))[None]
    emb = model.encode(x)

    occ_a, occ_b = _occ(gmap, starts), _occ(gmap, goals)
    field_a = model.decode(emb, occ_a, _hist(model, occ_a))
    field_b = model.decode(emb, occ_b, _hist(model, occ_b))  # different occupancy
    assert not torch.allclose(field_a, field_b), "field should react to occupancy"

    # encode takes no occupancy at all -> same map always yields the same embedding
    emb_again = model.encode(x)
    assert torch.allclose(emb, emb_again)


def test_agent_priorities_indexes_field():
    torch.manual_seed(0)
    gmap, starts, goals = _map_and_agents()
    model = build_model("embedding", dim=32, enc_depth=1, dec_depth=1).eval()

    x = torch.from_numpy(build_features(gmap, goals))[None]
    occ = _occ(gmap, starts)
    with torch.no_grad():
        field = model.decode(model.encode(x), occ, _hist(model, occ))[0]

    rho = agent_priorities(field, starts)
    assert rho.shape == (len(starts),)
    for i, (r, c) in enumerate(starts):
        assert rho[i] == float(field[r, c])


def test_predict_priority_field_zeroes_obstacles_and_caches_emb():
    torch.manual_seed(0)
    gmap, starts, goals = _map_and_agents()
    model = build_model("embedding", dim=32, enc_depth=1, dec_depth=1).eval()

    field, emb = predict_priority_field(model, gmap, starts, goals)
    assert field.shape == (gmap.H, gmap.W)
    obstacle = gmap.occ == 1
    if obstacle.any():
        assert float(field[obstacle].max()) == 0.0, "obstacles must never win priority"

    # Passing the cached embedding back must reproduce the same field.
    field2, _ = predict_priority_field(model, gmap, starts, goals, emb=emb)
    assert np.allclose(field, field2)


def test_backward_reaches_both_modules():
    torch.manual_seed(0)
    gmap, starts, goals = _map_and_agents()
    model = build_model("embedding", dim=32, enc_depth=1, dec_depth=1)

    x = torch.from_numpy(build_features(gmap, goals))[None]
    occ = _occ(gmap, starts)
    field = model(x, occ, _hist(model, occ))
    field.mean().backward()

    enc_grad = any(p.grad is not None for p in model.map_encoder.parameters())
    dec_grad = any(p.grad is not None for p in model.priority_decoder.parameters())
    assert enc_grad and dec_grad, "gradient must flow into both encoder and decoder"
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), "non-finite gradient"


def test_size_agnostic_and_batched():
    """Sinusoidal PE is built from (H, W) so a different map size works, and a
    batch of maps encodes/decodes in one pass."""
    torch.manual_seed(0)
    model = build_model("embedding", dim=32, enc_depth=1, dec_depth=1).eval()

    feats, occs = [], []
    for seed, (H, W) in enumerate([(18, 22), (18, 22)]):
        gmap, starts, goals = _map_and_agents(seed=seed, H=H, W=W)
        feats.append(build_features(gmap, goals))
        o = np.zeros((H, W), np.float32)
        for (r, c) in starts:
            o[r, c] = 1.0
        occs.append(o)
    x = torch.from_numpy(np.stack(feats))
    occ = torch.from_numpy(np.stack(occs))

    field = model.decode(model.encode(x), occ, _hist(model, occ))
    assert field.shape == (2, 18, 22)


def test_occ_history_window_pads_and_shapes():
    occs = [np.full((3, 3), i, np.float32) for i in range(5)]
    # early step: fewer than `history` frames -> front-padded with occs[0]
    w = occ_history_window(occs, 1, history=4)
    assert w.shape == (4, 3, 3)
    assert w[0, 0, 0] == 0 and w[1, 0, 0] == 0  # padding = occs[0] (==0)
    assert w[2, 0, 0] == 0 and w[3, 0, 0] == 1  # then occs[0], occs[1]
    # later step: exact window occs[1..4]
    w2 = occ_history_window(occs, 4, history=4)
    assert [w2[k, 0, 0] for k in range(4)] == [1, 2, 3, 4]


def test_decoder_uses_history():
    """Same current occupancy but different *history* must change the field —
    otherwise the new input is being ignored."""
    torch.manual_seed(0)
    gmap, starts, goals = _map_and_agents()
    model = build_model("embedding", dim=32, enc_depth=1, dec_depth=1,
                        history=4).eval()
    x = torch.from_numpy(build_features(gmap, goals))[None]
    emb = model.encode(x)
    occ = _occ(gmap, starts)

    hist_same = _hist(model, occ)                 # current frame repeated
    hist_moved = occ.clone()[:, None].repeat(1, 4, 1, 1)
    hist_moved[:, 0] = _occ(gmap, goals)[0]       # a different past frame
    with torch.no_grad():
        f_same = model.decode(emb, occ, hist_same)
        f_moved = model.decode(emb, occ, hist_moved)
    assert not torch.allclose(f_same, f_moved), "field should depend on history"


def test_dynamic_run_is_collision_free_and_varies():
    """The embedding model drives Simulator.run via field_fn under paper mode,
    the field changes across steps, and the rollout stays collision-free."""
    torch.manual_seed(0)
    gmap, starts, goals = _map_and_agents(seed=3, H=20, W=20)
    model = build_model("embedding", dim=32, enc_depth=1, dec_depth=1)

    sim = Simulator(gmap, starts, goals, max_steps=120, yield_mode="paper",
                    log_positions=True)
    ffn = embedding_field_fn(model, gmap, goals)
    res = sim.run(field_fn=ffn)

    log = res.positions_log
    assert len(log) >= 2
    for t in range(len(log) - 1):
        nxt = log[t + 1]
        assert len(set(nxt)) == len(nxt), f"vertex conflict at t={t}"
        prev = {a: i for i, a in enumerate(log[t])}
        for a, b in zip(log[t], nxt):
            assert b == a or b in gmap.neighbors(a), f"illegal move {a}->{b}"
        for i, (a, b) in enumerate(zip(log[t], nxt)):
            if b in prev and b != a:
                assert nxt[prev[b]] != a, "swap conflict"

    # The whole point of the dynamic model: the field is not constant.
    fields = [ffn(p) for p in log[:4]]
    assert any(not np.allclose(fields[0], f) for f in fields[1:])


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: OK")
