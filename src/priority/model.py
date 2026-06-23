"""CNN that maps map features -> a per-cell priority field.

A small U-Net-style encoder/decoder gives each cell a receptive field large
enough to reason about corridors and junctions. The output is a single positive
scalar per cell (softplus), used directly as the position-priority field. Because
the input is map-only, one forward pass produces the shared field for all agents.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import N_CHANNELS, build_features


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class PriorityUNet(nn.Module):
    def __init__(self, cin=N_CHANNELS, base=32):
        super().__init__()
        self.enc1 = ConvBlock(cin, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.Conv2d(base * 4, base * 2, 1)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.Conv2d(base * 2, base, 1)
        self.dec1 = ConvBlock(base * 2, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x, return_logits=False):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.dec2(torch.cat([self._up(self.up2(e3), e2), e2], 1))
        d1 = self.dec1(torch.cat([self._up(self.up1(d2), e1), e1], 1))
        logits = self.head(d1).squeeze(1)  # (B,H,W) pre-activation
        if return_logits:
            return logits
        return F.softplus(logits)          # (B,H,W), positive

    @staticmethod
    def _up(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode="nearest")


@torch.no_grad()
def predict_field(model, gmap, goals=None, device="cpu") -> np.ndarray:
    """Run the model on one map and return a dense (H, W) priority field.

    Obstacle cells are zeroed so they never win a priority comparison.
    """
    model.eval()
    feats = build_features(gmap, goals)
    x = torch.from_numpy(feats)[None].to(device)
    field = model(x)[0].cpu().numpy()
    field = field * (gmap.occ == 0)
    return field.astype(np.float32)
