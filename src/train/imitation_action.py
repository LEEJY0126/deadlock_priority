"""Behavioral-cloning helpers for the action-map policy.

The action-map policy (:class:`~src.priority.model_action.EmbeddingActionModel`)
must learn collision-free moves from scratch, which is slow under
terminate-on-first-collision RL. This module builds an imitation-learning
warm-start from an **expert that never collides**: the trained priority model (or
MST) driving PIBT. Because PIBT guarantees a legal joint move every step, *every*
transition of an expert rollout is a valid collision-free demonstration.

Pipeline: :func:`expert_positions_log` rolls out the expert and returns its
position log; :func:`actions_from_log` turns each step's per-agent displacement
into a discrete action label ``[UP, DOWN, LEFT, RIGHT, STAY]``; and
:func:`il_episode_loss` trains the decoder by cross-entropy at the agent cells
(the same field-then-index contract as at inference). A dataset "sample" is just
``(occ_grid, positions_log)`` — features are map-only so occupancy, the history
window, the agent cells, and the labels all reconstruct from the log.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..envs.action_exec import ACTION_MOVES
from ..envs.simulator import Simulator, oracle_kwargs
from ..priority.features import build_features
from ..priority.model_embedding import occ_history_window

# reverse map: displacement (dr, dc) -> action index in [UP,DOWN,LEFT,RIGHT,STAY]
_DELTA_TO_ACTION = {d: i for i, d in enumerate(ACTION_MOVES)}


def action_from_delta(delta) -> int:
    """Map a one-step displacement ``(dr, dc)`` to an action index.

    PIBT only ever moves an agent to an adjacent cell or leaves it in place, so a
    valid expert log yields deltas in ``ACTION_MOVES``. Anything else (a teleport /
    multi-cell jump) signals a malformed log and raises."""
    key = (int(delta[0]), int(delta[1]))
    if key not in _DELTA_TO_ACTION:
        raise ValueError(f"non-unit displacement {key} is not a valid action")
    return _DELTA_TO_ACTION[key]


def actions_from_log(positions_log):
    """Position log -> list of per-step ``[N]`` action-index labels.

    ``positions_log[t]`` is the agent configuration at step ``t``; the label at
    step ``t`` is the move that took the agents from ``t`` to ``t+1``. Returns
    ``len(positions_log) - 1`` label arrays."""
    labels = []
    for t in range(len(positions_log) - 1):
        cur, nxt = positions_log[t], positions_log[t + 1]
        labels.append(np.array(
            [action_from_delta((nxt[i][0] - cur[i][0], nxt[i][1] - cur[i][1]))
             for i in range(len(cur))], dtype=np.int64))
    return labels


def expert_positions_log(expert, gmap, starts, goals, *, oracle="paper",
                         max_steps=256, device="cpu", rng=None):
    """Roll out an expert (collision-free) and return its position log.

    ``expert`` selects the priority field driving PIBT:
      * an :class:`~src.priority.model_embedding.EmbeddingPriorityModel` -> the
        dynamic per-step ``embedding_field_fn`` (occupancy-conditioned);
      * any other learned model -> its static ``predict_field``;
      * ``None`` -> the MST baseline field.
    The rollout uses the real :class:`Simulator`, so the moves are exactly what the
    priority pipeline would execute (and are collision-free by construction)."""
    from ..priority.model_embedding import EmbeddingPriorityModel, embedding_field_fn
    sim = Simulator(gmap, starts, goals, max_steps=max_steps, log_positions=True,
                    **oracle_kwargs(oracle))
    if expert is None:
        from ..priority.mst_baseline import mst_priority_field
        res = sim.run(mst_priority_field(gmap), rng=rng)
    elif isinstance(expert, EmbeddingPriorityModel):
        field_fn = embedding_field_fn(expert, gmap, goals, device=device)
        res = sim.run(field_fn=field_fn, rng=rng)
    else:
        from ..priority.model import predict_field
        res = sim.run(predict_field(expert, gmap, device=device), rng=rng)
    return res.positions_log


def _agent_logits(logits_field, positions):
    """(5,H,W) tensor + [(r,c)] -> (N,5) tensor of per-agent action logits."""
    rows = [r for (r, c) in positions]
    cols = [c for (r, c) in positions]
    return logits_field[:, rows, cols].transpose(0, 1)   # (N, 5), grad-preserving


def il_episode_loss(model, occ_grid, positions_log, device="cpu", labels=None):
    """Cross-entropy behavioral-cloning loss for one expert episode.

    Encodes the map once (features are map-only), then replays each recorded step:
    build occupancy + the history window, decode the action-logit field, index it
    at the agent cells, and score the softmax against the expert's action labels.
    Returns ``(loss, n_correct, n_total)`` where ``loss`` carries grad and the
    counts (argmax == label) feed an accuracy metric. ``labels`` may be precomputed
    via :func:`actions_from_log`; otherwise it is derived from ``positions_log``."""
    from ..envs.grid import GridMap
    gmap = occ_grid if isinstance(occ_grid, GridMap) else GridMap(np.asarray(occ_grid))
    H, W = gmap.H, gmap.W
    if labels is None:
        labels = actions_from_log(positions_log)
    T = len(labels)
    if T == 0:
        z = torch.zeros((), device=device)
        return z, 0, 0

    feats = torch.from_numpy(build_features(gmap))[None].to(device)
    emb = model.encode(feats)
    history = model.config["history"]

    occs = []
    total_loss = torch.zeros((), device=device)
    n_correct = n_total = 0
    for t in range(T):
        positions = positions_log[t]
        occ_np = np.zeros((H, W), np.float32)
        for (r, c) in positions:
            occ_np[r, c] = 1.0
        occs.append(occ_np)
        occ = torch.from_numpy(occ_np)[None].to(device)
        win = occ_history_window(occs, len(occs) - 1, history)
        hist = torch.from_numpy(win)[None].to(device)
        logits_field = model.decode(emb, occ, hist)[0]       # (5, H, W)
        la = _agent_logits(logits_field, positions)          # (N, 5)
        target = torch.from_numpy(labels[t]).to(device)
        total_loss = total_loss + F.cross_entropy(la, target, reduction="sum")
        n_correct += int((la.argmax(dim=1) == target).sum())
        n_total += len(positions)

    return total_loss / max(n_total, 1), n_correct, n_total


__all__ = ["action_from_delta", "actions_from_log", "expert_positions_log",
           "il_episode_loss"]
