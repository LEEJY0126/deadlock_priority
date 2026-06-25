"""Transformer alternative that maps map features -> a per-cell priority field.

Same contract as :mod:`model` (input ``(B, N_CHANNELS, H, W)`` -> output
``(B, H, W)`` positive field), so it is a drop-in swap for ``PriorityUNet``.

Pipeline:
  1. A small conv stem extracts local features and lifts to the token dim:
     ``(B, C, H, W) -> (B, D, H, W)``.
  2. Each cell becomes a token: ``(B, D, H, W) -> (B, H*W, D)``.
  3. A plain Transformer *encoder* lets every cell attend to every other cell,
     adding global map context (corridors/junctions across the whole map).
  4. A per-token MLP head projects each token to a scalar priority.

Design notes (vs the questions in the design doc):
  * Positional encoding is added before the encoder. It is *sinusoidal* and
    built on the fly from (H, W) so the model generalizes to any map size --
    a learned position table would lock us to one resolution, but
    ``predict_field`` runs on arbitrary maps.
  * A Transformer encoder (not a ViT) is the right tool: we want a dense,
    per-cell output, so one token per cell and no ``[CLS]`` token.
  * The encoder output goes straight into a small per-token MLP head -- no
    Transformer decoder. A conv-stem skip is added back before the head so
    global attention does not wash out local detail (cf. the U-Net skips).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import N_CHANNELS
from .model import ConvBlock, load_model, predict_field  # re-exported (arch-generic)


def sinusoidal_pe_2d(h: int, w: int, dim: int, device=None, dtype=None) -> torch.Tensor:
    """2D sinusoidal positional encoding, returned flattened as (H*W, dim).

    Half the channels encode the row index, half the column index, using the
    standard Transformer sin/cos frequency schedule. Works for any (h, w).
    """
    if dim % 4 != 0:
        raise ValueError(f"dim must be divisible by 4 for 2D PE, got {dim}")
    d = dim // 2  # channels per spatial axis
    div = torch.exp(torch.arange(0, d, 2, device=device, dtype=torch.float32)
                    * (-np.log(10000.0) / d))
    rows = torch.arange(h, device=device, dtype=torch.float32)[:, None]  # (H,1)
    cols = torch.arange(w, device=device, dtype=torch.float32)[:, None]  # (W,1)

    pe_r = torch.zeros(h, d, device=device)
    pe_r[:, 0::2], pe_r[:, 1::2] = torch.sin(rows * div), torch.cos(rows * div)
    pe_c = torch.zeros(w, d, device=device)
    pe_c[:, 0::2], pe_c[:, 1::2] = torch.sin(cols * div), torch.cos(cols * div)

    pe = torch.cat([pe_r[:, None, :].expand(h, w, d),   # row enc broadcast over cols
                    pe_c[None, :, :].expand(h, w, d)], dim=-1)  # col enc over rows
    pe = pe.reshape(h * w, dim)
    return pe.to(dtype) if dtype is not None else pe


class PriorityTransformer(nn.Module):
    def __init__(self, cin=N_CHANNELS, dim=128, depth=4, heads=4, mlp_ratio=4,
                 dropout=0.0):
        super().__init__()
        # Conv stem: local features + lift to token dim, full resolution.
        self.stem = nn.Sequential(ConvBlock(cin, dim), ConvBlock(dim, dim))
        self.pe_scale = nn.Parameter(torch.tensor(1.0))
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * mlp_ratio,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        # Per-token head; the stem skip is concatenated so local detail survives.
        self.head = nn.Sequential(
            nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim), nn.GELU(),
            nn.Linear(dim, 1),
        )
        self.config = dict(cin=cin, dim=dim, depth=depth, heads=heads,
                           mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x, return_logits=False):
        B, _, H, W = x.shape
        feat = self.stem(x)                          # (B, D, H, W)
        D = feat.shape[1]
        tokens = feat.flatten(2).transpose(1, 2)     # (B, H*W, D)
        pe = sinusoidal_pe_2d(H, W, D, device=x.device, dtype=tokens.dtype)
        enc = self.encoder(tokens + self.pe_scale * pe[None])  # (B, H*W, D)
        enc = self.norm(enc)
        fused = torch.cat([enc, tokens], dim=-1)      # stem skip -> (B, H*W, 2D)
        logits = self.head(fused).squeeze(-1)         # (B, H*W)
        logits = logits.view(B, H, W)
        if return_logits:
            return logits
        return F.softplus(logits)                     # (B,H,W), positive

    @property
    def no_pool(self):  # symmetry with PriorityUNet ckpt flag; stem is full-res
        return True


# load_model / predict_field are arch-generic and live in model.py; they are
# imported above and re-exported here so ``from .model_transformer import
# predict_field`` keeps working.
__all__ = ["PriorityTransformer", "sinusoidal_pe_2d", "load_model", "predict_field"]
