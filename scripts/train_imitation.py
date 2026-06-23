"""Imitation pretraining: regress the CNN onto oracle-selected priority fields.

We standardize both prediction and target per-map over free cells, so the loss
cares only about the *spatial ordering* of priorities (scale/offset are absorbed
by the tie-break and learned later by RL).
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from src.envs.grid import GridMap
from src.priority.features import build_features
from src.priority.model import PriorityUNet


def load(path):
    d = np.load(path, allow_pickle=True)
    occ, label = d["occ"], d["label"]
    feats = np.stack([build_features(GridMap(o)) for o in occ]).astype(np.float32)
    free = (occ == 0).astype(np.float32)
    return (torch.from_numpy(feats), torch.from_numpy(label.astype(np.float32)),
            torch.from_numpy(free))


def standardize(x, mask):
    """Zero-mean unit-var over masked cells, per sample."""
    m = mask.sum(dim=(-2, -1)).clamp(min=1)
    mean = (x * mask).sum(dim=(-2, -1)) / m
    xc = (x - mean[:, None, None]) * mask
    var = (xc ** 2).sum(dim=(-2, -1)) / m
    return xc / (var[:, None, None].sqrt() + 1e-5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/imitation.npz")
    ap.add_argument("--out", default="runs/imitation.pt")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    feats, label, free = load(args.data)
    n = feats.shape[0]
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(n, generator=g)
    n_val = max(1, n // 6)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    dev = args.device
    feats, label, free = feats.to(dev), label.to(dev), free.to(dev)
    model = PriorityUNet().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    def loss_on(idx):
        pred = model(feats[idx])
        ps = standardize(pred, free[idx])
        ls = standardize(label[idx], free[idx])
        return F.mse_loss(ps * free[idx], ls * free[idx])

    best = float("inf")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for ep in range(args.epochs):
        model.train()
        tr = tr_idx[torch.randperm(len(tr_idx))]
        tot = 0.0
        for i in range(0, len(tr), args.bs):
            b = tr[i:i + args.bs]
            opt.zero_grad()
            loss = loss_on(b)
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            vl = loss_on(val_idx).item()
        if vl < best:
            best = vl
            torch.save({"model": model.state_dict()}, args.out)
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"ep {ep:3d} train {tot/len(tr):.4f} val {vl:.4f} best {best:.4f}")
    print(f"saved best (val {best:.4f}) -> {args.out}")


if __name__ == "__main__":
    main()
