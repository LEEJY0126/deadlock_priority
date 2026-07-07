"""Embedding-map priority architecture (feature/embedding branch).

Splits the single-shot :class:`PriorityTransformer` into two models so that
priority can become *dynamic* (recomputed per step) instead of a static field:

  MapEncoder      : map features (B, C, H, W)  -> shared embedding map (B, D, H, W)
                    Map-only input, so the embedding is identical for every agent
                    -- preserving the communication-free invariant. Expensive
                    (a Transformer trunk), so it is meant to run ONCE per map.

  PriorityDecoder : embedding map (B, D, H, W) + live occupancy (B, H, W)
                    -> priority FIELD (B, H, W). Cheap; meant to run every step.

Each agent reads its scalar priority by indexing the decoder's field at its own
cell (see :func:`agent_priorities`). Emitting a *field* rather than a raw ``[N]``
vector keeps this a drop-in for the PIBT consumer (which already indexes a field
per agent) and stays agnostic to the number and ordering of agents. Feeding live
occupancy is what makes the field respond to the current configuration -- the
piece a static map-only field cannot express (cf. the hand-designed ``beta``
stuck-time boost in the simulator).

Reuses the transformer trunk pieces from :mod:`model_transformer`:
  * ``ConvBlock``       -- local conv stem.
  * ``sinusoidal_pe_2d`` -- size-agnostic 2D positional encoding.

Usage (per map, then per step)::

    model = EmbeddingPriorityModel()
    emb = model.encode(feats)                 # once per map
    for t in range(T):
        field = model.decode(emb, occ_t)      # cheap, per step
        rho = agent_priorities(field[0], positions_t)
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import N_CHANNELS, build_features
from .model import ConvBlock
from .model_transformer import sinusoidal_pe_2d


def _encoder_layer(dim, heads, mlp_ratio, dropout):
    return nn.TransformerEncoderLayer(
        d_model=dim, nhead=heads, dim_feedforward=dim * mlp_ratio,
        dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
    )


class MapEncoder(nn.Module):
    """Map features -> shared per-cell embedding map (B, D, H, W).

    Same trunk as :class:`PriorityTransformer` up to (but not including) the
    scalar head: conv stem -> per-cell tokens + sinusoidal PE -> global
    self-attention. The global-attended tokens are fused with the local stem
    skip (as in the U-Net skips) so fine detail survives, then reshaped back to
    a spatial embedding map for the decoder to index.
    """

    def __init__(self, cin=N_CHANNELS, dim=128, depth=4, heads=4, mlp_ratio=4,
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.stem = nn.Sequential(ConvBlock(cin, dim), ConvBlock(dim, dim))
        self.pe_scale = nn.Parameter(torch.tensor(1.0))
        self.encoder = nn.TransformerEncoder(
            _encoder_layer(dim, heads, mlp_ratio, dropout), num_layers=depth,
            enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.fuse = nn.Linear(dim * 2, dim)  # global-attended + local stem skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        feat = self.stem(x)                            # (B, D, H, W)
        tokens = feat.flatten(2).transpose(1, 2)       # (B, H*W, D)
        pe = sinusoidal_pe_2d(H, W, self.dim, device=x.device, dtype=tokens.dtype)
        enc = self.norm(self.encoder(tokens + self.pe_scale * pe[None]))
        fused = self.fuse(torch.cat([enc, tokens], dim=-1))   # (B, H*W, D)
        return fused.transpose(1, 2).view(B, self.dim, H, W)  # (B, D, H, W)


def occ_history_window(occs, t, history):
    """Stack the ``history`` occupancy frames ending at step ``t`` (inclusive)
    into a ``[history, H, W]`` array.

    ``occs`` is a list of ``[H, W]`` occupancy grids (1 = agent). When fewer than
    ``history`` steps have elapsed the window is front-padded with the earliest
    frame (``occs[0]`` -- the start configuration, i.e. "stationary before the
    episode"). Used identically for the live rollout and for PPO replay so the
    two see the exact same input.
    """
    frames = [occs[k] if k >= 0 else occs[0]
              for k in range(t - history + 1, t + 1)]
    return np.stack(frames, axis=0).astype(np.float32)


class HistoryEncoder(nn.Module):
    """Occupancy history ``[B, history, H, W]`` -> embedding ``[B, hist_dim, H, W]``.

    The ``history`` frames are treated as input channels, lifted by a conv stem,
    then a shallow spatial Transformer (same trunk pattern as ``MapEncoder``)
    mixes global context. The result is a per-cell embedding of *where agents
    have been*, giving the decoder temporal signal (motion / stuck-time) that a
    single occupancy snapshot cannot express.
    """

    def __init__(self, history, hist_dim=64, depth=2, heads=4, mlp_ratio=4,
                 dropout=0.0):
        super().__init__()
        self.hist_dim = hist_dim
        self.stem = nn.Sequential(ConvBlock(history, hist_dim),
                                  ConvBlock(hist_dim, hist_dim))
        self.pe_scale = nn.Parameter(torch.tensor(1.0))
        self.encoder = nn.TransformerEncoder(
            _encoder_layer(hist_dim, heads, mlp_ratio, dropout), num_layers=depth,
            enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hist_dim)

    def forward(self, hist: torch.Tensor) -> torch.Tensor:
        """hist (B, history, H, W) -> (B, hist_dim, H, W)."""
        if hist.dim() != 4:
            raise ValueError(f"history must be (B, T, H, W), got {tuple(hist.shape)}")
        B, _, H, W = hist.shape
        feat = self.stem(hist)
        tokens = feat.flatten(2).transpose(1, 2)
        pe = sinusoidal_pe_2d(H, W, self.hist_dim, device=hist.device, dtype=tokens.dtype)
        enc = self.norm(self.encoder(tokens + self.pe_scale * pe[None]))
        return enc.transpose(1, 2).view(B, self.hist_dim, H, W)


class PriorityDecoder(nn.Module):
    """Embedding map + live occupancy (+ occupancy history) -> priority field.

    Occupancy is lifted with a small conv (so a cell sees *nearby* agents, not
    just its own); the occupancy **history** is encoded by a `HistoryEncoder`
    and projected; both are concatenated with the shared embedding, and a shallow
    Transformer lets every cell attend to where agents are (and have been) before
    a per-cell head emits a positive priority. Kept shallow because it runs every
    step; the heavy map understanding lives in the (once-per-map) encoder.
    """

    def __init__(self, dim=128, depth=2, heads=4, mlp_ratio=4, dropout=0.0,
                 occ_dim=16, history=8, hist_dim=64, hist_depth=2):
        super().__init__()
        self.dim = dim
        self.history = history
        self.occ = ConvBlock(1, occ_dim)               # local context for occupancy
        self.hist_encoder = HistoryEncoder(history, hist_dim, hist_depth, heads,
                                           mlp_ratio, dropout)
        self.hist_proj = nn.Conv2d(hist_dim, occ_dim, 1)  # history -> occ_dim chans
        # fuse embedding + occupancy + history
        self.proj = nn.Conv2d(dim + occ_dim + occ_dim, dim, 1)
        self.pe_scale = nn.Parameter(torch.tensor(1.0))
        self.decoder = nn.TransformerEncoder(
            _encoder_layer(dim, heads, mlp_ratio, dropout), num_layers=depth,
            enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1),
        )

    def forward(self, emb: torch.Tensor, occ: torch.Tensor,
                occ_history: torch.Tensor, return_logits=False) -> torch.Tensor:
        """emb (B,D,H,W), occ (B,H,W) with 1=agent, occ_history (B,history,H,W)."""
        if occ.dim() != 3:
            raise ValueError(f"occ must be (B, H, W), got {tuple(occ.shape)}")
        B, D, H, W = emb.shape
        o = self.occ(occ[:, None])                     # (B, occ_dim, H, W)
        h = self.hist_proj(self.hist_encoder(occ_history))  # (B, occ_dim, H, W)
        fused = self.proj(torch.cat([emb, o, h], dim=1))    # (B, D, H, W)
        tokens = fused.flatten(2).transpose(1, 2)      # (B, H*W, D)
        pe = sinusoidal_pe_2d(H, W, D, device=emb.device, dtype=tokens.dtype)
        dec = self.norm(self.decoder(tokens + self.pe_scale * pe[None]))
        logits = self.head(dec).squeeze(-1).view(B, H, W)
        if return_logits:
            return logits
        return F.softplus(logits)                      # (B, H, W), positive


class EmbeddingPriorityModel(nn.Module):
    """MapEncoder + PriorityDecoder wired together.

    ``forward`` runs both (convenient for training on single (map, occupancy)
    pairs). At rollout time call :meth:`encode` once per map and :meth:`decode`
    each step to avoid re-running the encoder trunk.
    """

    def __init__(self, cin=N_CHANNELS, dim=128, enc_depth=4, dec_depth=2,
                 heads=4, mlp_ratio=4, dropout=0.0, occ_dim=16, history=8,
                 hist_dim=64, hist_depth=2):
        super().__init__()
        self.history = history
        self.map_encoder = MapEncoder(cin, dim, enc_depth, heads, mlp_ratio, dropout)
        self.priority_decoder = PriorityDecoder(dim, dec_depth, heads, mlp_ratio,
                                                dropout, occ_dim, history, hist_dim,
                                                hist_depth)
        self.config = dict(cin=cin, dim=dim, enc_depth=enc_depth,
                           dec_depth=dec_depth, heads=heads, mlp_ratio=mlp_ratio,
                           dropout=dropout, occ_dim=occ_dim, history=history,
                           hist_dim=hist_dim, hist_depth=hist_depth)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map features (B, C, H, W) -> shared embedding map (B, D, H, W)."""
        return self.map_encoder(x)

    def decode(self, emb: torch.Tensor, occ: torch.Tensor,
               occ_history: torch.Tensor, return_logits=False) -> torch.Tensor:
        return self.priority_decoder(emb, occ, occ_history, return_logits=return_logits)

    def forward(self, x: torch.Tensor, occ: torch.Tensor,
                occ_history: torch.Tensor, return_logits=False) -> torch.Tensor:
        return self.decode(self.encode(x), occ, occ_history, return_logits=return_logits)

    @property
    def no_pool(self):  # symmetry with the other arch ckpt flags; trunk is full-res
        return True


def agent_priorities(field, positions) -> np.ndarray:
    """Index a priority field at each agent's cell -> per-agent priorities [N].

    `field` is (H, W) (numpy array or tensor); `positions` is a list of (r, c).
    This is the field-then-index step: one shared field, read off per agent, so
    the result is agnostic to agent count and ordering.
    """
    if isinstance(field, torch.Tensor):
        field = field.detach().cpu().numpy()
    return np.array([float(field[r, c]) for (r, c) in positions], dtype=np.float32)


@torch.no_grad()
def predict_priority_field(model, gmap, positions, goals=None, device="cpu",
                           emb=None):
    """Dynamic analogue of ``model.predict_field`` for the embedding model.

    Returns ``(field, emb)`` where `field` is a dense (H, W) priority field for
    the current `positions`, and `emb` is the cached embedding map so a caller
    stepping a rollout can pass it back in and skip the encoder trunk.
    Obstacle cells are zeroed so they never win a priority comparison.
    """
    model.eval()
    if emb is None:
        feats = build_features(gmap, goals)
        emb = model.encode(torch.from_numpy(feats)[None].to(device))
    occ = np.zeros((gmap.H, gmap.W), dtype=np.float32)
    for (r, c) in positions:
        occ[r, c] = 1.0
    # single snapshot -> history is the current frame repeated (no motion info)
    hist = np.repeat(occ[None], model.config["history"], axis=0)
    occ_t = torch.from_numpy(occ)[None].to(device)
    hist_t = torch.from_numpy(hist)[None].to(device)
    field = model.decode(emb, occ_t, hist_t)[0]
    field = field.cpu().numpy() * (gmap.occ == 0)
    return field.astype(np.float32), emb


class InferenceTimer:
    """Accumulates encoder / decoder forward times to report mean latency.

    ``embedding_field_fn`` records one encode (once per map) and one decode (per
    step) per call when a timer is passed; ``encode_ms`` / ``decode_ms`` are the
    running means. Timing is CUDA-synchronized so GPU numbers are accurate.
    """

    def __init__(self):
        self.encode_s = self.decode_s = 0.0
        self.encode_n = self.decode_n = 0

    @staticmethod
    def _mean_ms(total_s, n):
        return 1e3 * total_s / n if n else float("nan")

    @property
    def encode_ms(self):
        return self._mean_ms(self.encode_s, self.encode_n)

    @property
    def decode_ms(self):
        return self._mean_ms(self.decode_s, self.decode_n)


def _sync(device):
    if isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def embedding_field_fn(model, gmap, goals=None, device="cpu", timer=None):
    """Build a per-step ``field_fn(positions) -> (H, W)`` for ``Simulator.run``.

    Encodes the (static) map **once**, then each call runs only the cheap decoder
    on the current occupancy. Obstacle cells are zeroed. This is the bridge that
    lets the dynamic embedding model drop into the existing episode runner
    (``sim.run(field_fn=embedding_field_fn(model, gmap, goals))``) without the
    simulator needing to import torch. Pass an :class:`InferenceTimer` to record
    encoder (once) and decoder (per step) forward latency.
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
    free = (gmap.occ == 0)
    history = model.config["history"]
    occs = []  # per-episode occupancy buffer for the history window

    def field_fn(positions):
        occ = np.zeros((gmap.H, gmap.W), dtype=np.float32)
        for (r, c) in positions:
            occ[r, c] = 1.0
        occs.append(occ)
        win = occ_history_window(occs, len(occs) - 1, history)  # (history,H,W)
        occ_t = torch.from_numpy(occ)[None].to(device)
        hist_t = torch.from_numpy(win)[None].to(device)
        with torch.no_grad():
            _sync(device)
            t0 = time.perf_counter()
            field = model.decode(emb, occ_t, hist_t)[0]
            _sync(device)
            if timer is not None:
                timer.decode_s += time.perf_counter() - t0
                timer.decode_n += 1
        return field.cpu().numpy() * free

    return field_fn


__all__ = ["MapEncoder", "HistoryEncoder", "PriorityDecoder",
           "EmbeddingPriorityModel", "agent_priorities", "predict_priority_field",
           "embedding_field_fn", "occ_history_window", "InferenceTimer"]
