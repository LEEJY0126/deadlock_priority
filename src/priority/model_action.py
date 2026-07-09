"""Dynamic action-map architecture (feature/embedding branch).

A sibling of :mod:`model_embedding` that emits a per-cell **action distribution**
instead of a scalar priority. The trunk is shared and unchanged:

  MapEncoder    : map features (B, C, H, W) -> shared embedding map (B, D, H, W).
                  Map-only, run ONCE per map (communication-free invariant).
  HistoryEncoder: occupancy history (B, T, H, W) -> temporal embedding.

  ActionDecoder : embedding + live occupancy (+ history) -> action logits
                  (B, 5, H, W), a categorical distribution over
                  ``[UP, DOWN, LEFT, RIGHT, STAY]`` for *every* cell.

Each agent reads the 5-vector at its own cell (see :func:`agent_action_logits`)
and moves accordingly. This keeps the "field-then-index" contract -- one shared
field, read off per agent -- so the policy is agnostic to agent count/ordering,
exactly like the priority model. The difference: there is no PIBT downstream, so
the model is directly responsible for avoiding collisions (see
:mod:`src.envs.action_exec`).

The channel order is fixed to the request ``[UP, DOWN, LEFT, RIGHT, STAY]``; the
matching deltas live in :data:`src.envs.action_exec.ACTION_MOVES`.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn

from .features import N_CHANNELS, build_features
from .model_embedding import (HistoryEncoder, MapEncoder, _encoder_layer, _sync,
                              occ_history_window)
from .model_transformer import sinusoidal_pe_2d
from .model import ConvBlock

N_ACTIONS = 5  # [UP, DOWN, LEFT, RIGHT, STAY]


class ActionDecoder(nn.Module):
    """Embedding map + live occupancy (+ history) -> action logits (B, 5, H, W).

    Body is identical to :class:`~src.priority.model_embedding.PriorityDecoder`
    (occupancy lifted by a conv so a cell sees nearby agents; history encoded and
    projected; both fused with the shared embedding; a shallow Transformer mixes
    global context) -- only the head differs: it emits ``n_actions`` logits per
    cell instead of one positive scalar. Kept shallow because it runs every step.
    """

    def __init__(self, dim=128, depth=2, heads=4, mlp_ratio=4, dropout=0.0,
                 occ_dim=16, history=8, hist_dim=64, hist_depth=2,
                 n_actions=N_ACTIONS):
        super().__init__()
        self.dim = dim
        self.history = history
        self.n_actions = n_actions
        self.occ = ConvBlock(1, occ_dim)               # local context for occupancy
        self.hist_encoder = HistoryEncoder(history, hist_dim, hist_depth, heads,
                                           mlp_ratio, dropout)
        self.hist_proj = nn.Conv2d(hist_dim, hist_dim, 1)  # history projection
        # fuse embedding + occupancy + history
        self.proj = nn.Conv2d(dim + occ_dim + hist_dim, dim, 1)
        self.pe_scale = nn.Parameter(torch.tensor(1.0))
        self.decoder = nn.TransformerEncoder(
            _encoder_layer(dim, heads, mlp_ratio, dropout), num_layers=depth,
            enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(),
            nn.Linear(dim, n_actions),
        )

    def forward(self, emb: torch.Tensor, occ: torch.Tensor,
                occ_history: torch.Tensor) -> torch.Tensor:
        """emb (B,D,H,W), occ (B,H,W) with 1=agent, occ_history (B,history,H,W)
        -> action logits (B, n_actions, H, W)."""
        if occ.dim() != 3:
            raise ValueError(f"occ must be (B, H, W), got {tuple(occ.shape)}")
        B, D, H, W = emb.shape
        o = self.occ(occ[:, None])                     # (B, occ_dim, H, W)
        h = self.hist_proj(self.hist_encoder(occ_history))  # (B, hist_dim, H, W)
        fused = self.proj(torch.cat([emb, o, h], dim=1))    # (B, D, H, W)
        tokens = fused.flatten(2).transpose(1, 2)      # (B, H*W, D)
        pe = sinusoidal_pe_2d(H, W, D, device=emb.device, dtype=tokens.dtype)
        dec = self.norm(self.decoder(tokens + self.pe_scale * pe[None]))
        logits = self.head(dec)                        # (B, H*W, n_actions)
        return logits.transpose(1, 2).view(B, self.n_actions, H, W)


class EmbeddingActionModel(nn.Module):
    """MapEncoder + ActionDecoder wired together (action-policy sibling).

    ``forward`` runs both (training convenience). At rollout time call
    :meth:`encode` once per map and :meth:`decode` each step to avoid re-running
    the encoder trunk. ``decode`` returns action logits ``(B, 5, H, W)``.
    """

    def __init__(self, cin=N_CHANNELS, dim=128, enc_depth=4, dec_depth=2,
                 heads=4, mlp_ratio=4, dropout=0.0, occ_dim=16, history=8,
                 hist_dim=64, hist_depth=2, n_actions=N_ACTIONS):
        super().__init__()
        self.history = history
        self.n_actions = n_actions
        self.map_encoder = MapEncoder(cin, dim, enc_depth, heads, mlp_ratio, dropout)
        self.action_decoder = ActionDecoder(dim, dec_depth, heads, mlp_ratio,
                                            dropout, occ_dim, history, hist_dim,
                                            hist_depth, n_actions)
        self.config = dict(cin=cin, dim=dim, enc_depth=enc_depth,
                           dec_depth=dec_depth, heads=heads, mlp_ratio=mlp_ratio,
                           dropout=dropout, occ_dim=occ_dim, history=history,
                           hist_dim=hist_dim, hist_depth=hist_depth,
                           n_actions=n_actions)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map features (B, C, H, W) -> shared embedding map (B, D, H, W)."""
        return self.map_encoder(x)

    def decode(self, emb: torch.Tensor, occ: torch.Tensor,
               occ_history: torch.Tensor) -> torch.Tensor:
        return self.action_decoder(emb, occ, occ_history)

    def forward(self, x: torch.Tensor, occ: torch.Tensor,
                occ_history: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x), occ, occ_history)

    @property
    def no_pool(self):  # symmetry with the other arch ckpt flags; trunk is full-res
        return True


def agent_action_logits(field, positions) -> np.ndarray:
    """Index an action field at each agent's cell -> per-agent logits [N, 5].

    ``field`` is (5, H, W) (numpy or tensor); ``positions`` is a list of (r, c).
    The field-then-index step for actions: one shared action field, read off per
    agent, so the result is agnostic to agent count and ordering.
    """
    if isinstance(field, torch.Tensor):
        field = field.detach().cpu().numpy()
    return np.stack([field[:, r, c] for (r, c) in positions], axis=0).astype(np.float32)


def _occ_from_positions(positions, H, W):
    occ = np.zeros((H, W), dtype=np.float32)
    for (r, c) in positions:
        occ[r, c] = 1.0
    return occ


def action_field_fn(model, gmap, goals=None, device="cpu", timer=None):
    """Build a per-step ``field_fn(positions) -> (5, H, W)`` action logits map.

    Encodes the (static) map **once**, then each call runs only the cheap decoder
    on the current occupancy + history window. Mirrors
    :func:`src.priority.model_embedding.embedding_field_fn`; used by the greedy
    eval runner. Pass an :class:`~src.priority.model_embedding.InferenceTimer` to
    record encoder (once) and decoder (per step) forward latency.
    """
    model.eval()
    feats = build_features(gmap, goals)
    x = torch.from_numpy(feats)[None].to(device)
    with torch.no_grad():
        _sync(device)
        t0 = time.perf_counter()
        emb = model.encode(x)
        _sync(device)
        if timer is not None:
            timer.encode_s += time.perf_counter() - t0
            timer.encode_n += 1
    history = model.config["history"]
    occs = []  # per-episode occupancy buffer for the history window

    def field_fn(positions):
        occ = _occ_from_positions(positions, gmap.H, gmap.W)
        occs.append(occ)
        win = occ_history_window(occs, len(occs) - 1, history)  # (history,H,W)
        occ_t = torch.from_numpy(occ)[None].to(device)
        hist_t = torch.from_numpy(win)[None].to(device)
        with torch.no_grad():
            _sync(device)
            t0 = time.perf_counter()
            logits = model.decode(emb, occ_t, hist_t)[0]        # (5, H, W)
            _sync(device)
            if timer is not None:
                timer.decode_s += time.perf_counter() - t0
                timer.decode_n += 1
        return logits.cpu().numpy()

    return field_fn


def greedy_action_fn(model, gmap, goals=None, device="cpu", timer=None):
    """``action_fn(positions, t) -> [N] argmax action indices`` for eval.

    Wraps :func:`action_field_fn` and turns the per-step action-logits map into a
    greedy per-agent action by indexing at each agent cell and taking argmax.
    Suitable for :func:`src.envs.action_exec.run_action_episode`.
    """
    ffn = action_field_fn(model, gmap, goals, device=device, timer=timer)

    def action_fn(positions, t):
        field = ffn(positions)                    # (5, H, W)
        logits = agent_action_logits(field, positions)  # (N, 5)
        return logits.argmax(axis=1)

    return action_fn


__all__ = ["N_ACTIONS", "ActionDecoder", "EmbeddingActionModel",
           "agent_action_logits", "action_field_fn", "greedy_action_fn"]
