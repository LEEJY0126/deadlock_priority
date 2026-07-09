"""Imitation pretraining (behavioral cloning) for the action-map policy.

Trains an ``EmbeddingActionModel`` to reproduce the expert's per-agent moves from a
dataset built by ``gen_dataset_action.py``: cross-entropy over
``[UP, DOWN, LEFT, RIGHT, STAY]`` at each agent cell, every step of every episode.
Saves an ``arch="embedding_action"`` checkpoint that ``evaluate.py`` auto-routes and
that warm-starts RL via ``train_embedding_action_rl.py --init``.
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from tqdm import tqdm

from src.priority.model import build_model
from src.train.imitation_action import actions_from_log, il_episode_loss


def load_dataset(path):
    d = np.load(path, allow_pickle=True)
    occ = d["occ"]                                   # (E, H, W) uint8
    positions = d["positions"]                       # object[E]: list[list[(r,c)]]
    episodes = []
    for e in range(len(occ)):
        log = [[tuple(p) for p in step] for step in positions[e]]
        episodes.append((occ[e], log, actions_from_log(log)))
    meta = str(d["meta"]) if "meta" in d else "{}"
    return episodes, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/imitation_action.npz")
    ap.add_argument("--out", default="runs/imitation_action.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=8, help="episodes per optimizer step")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--enc_depth", type=int, default=4)
    ap.add_argument("--dec_depth", type=int, default=2)
    ap.add_argument("--history", type=int, default=8,
                    help="occupancy-history frames (model hyperparam; the window is "
                         "rebuilt from the stored log, so any value is valid)")
    ap.add_argument("--hist_dim", type=int, default=64)
    ap.add_argument("--hist_depth", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    _argv = getattr(sys, "orig_argv", None) or ([sys.executable] + sys.argv)
    print(f"command: {' '.join(_argv)}", flush=True)

    torch.manual_seed(args.seed)
    episodes, meta = load_dataset(args.data)
    print(f"loaded {len(episodes)} episodes from {args.data}; gen config: {meta}",
          flush=True)

    dev = args.device
    model = build_model("embedding_action", dim=args.dim, enc_depth=args.enc_depth,
                        dec_depth=args.dec_depth, history=args.history,
                        hist_dim=args.hist_dim, hist_depth=args.hist_depth).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(episodes), generator=g).tolist()
    n_val = max(1, len(episodes) // 6)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    shuffle_rng = np.random.default_rng(args.seed)

    def run_epoch(idx, train, desc=None):
        model.train() if train else model.eval()
        order = list(idx)
        if train:
            shuffle_rng.shuffle(order)
        tot_loss, correct, total = 0.0, 0, 0
        step = args.bs if train else len(order) or 1
        starts = range(0, len(order), step)
        # inner bar over training batches (disappears when the epoch ends)
        bar = tqdm(starts, desc=desc, leave=False, disable=desc is None)
        for i in bar:
            chunk = order[i:i + step]
            batch_loss = torch.zeros((), device=dev)
            batch_n = 0
            for e in chunk:
                occ, log, labels = episodes[e]
                loss, nc, nt = il_episode_loss(model, occ, log, device=dev,
                                               labels=labels)
                if nt == 0:
                    continue
                batch_loss = batch_loss + loss * nt   # undo per-episode mean
                batch_n += nt
                correct += nc
                total += nt
                tot_loss += loss.item() * nt
            if train and batch_n > 0:
                opt.zero_grad()
                (batch_loss / batch_n).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            if desc is not None and total:
                bar.set_postfix(loss=f"{tot_loss / total:.3f}",
                                acc=f"{correct / total * 100:.1f}%")
        return tot_loss / max(total, 1), correct / max(total, 1)

    best = -1.0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    epochs = tqdm(range(args.epochs), desc="IL", unit="ep")
    for ep in epochs:
        tr_loss, tr_acc = run_epoch(tr_idx, train=True, desc=f"ep {ep} train")
        with torch.no_grad():
            val_loss, val_acc = run_epoch(val_idx, train=False)
        improved = val_acc > best
        if improved:
            best = val_acc
            torch.save({"arch": "embedding_action", "config": model.config,
                        "model": model.state_dict()}, args.out)
        # live metrics on the epoch bar; a persistent line every 5 epochs so a
        # redirected log still keeps a history (tqdm.write stays above the bar).
        epochs.set_postfix(tr_loss=f"{tr_loss:.3f}", tr_acc=f"{tr_acc*100:.1f}%",
                           val_loss=f"{val_loss:.3f}", val_acc=f"{val_acc*100:.1f}%",
                           best=f"{best*100:.1f}%")
        if ep % 5 == 0 or ep == args.epochs - 1 or improved:
            tqdm.write(f"ep {ep:3d} train_loss {tr_loss:.4f} acc {tr_acc*100:5.1f}%  "
                       f"val_loss {val_loss:.4f} acc {val_acc*100:5.1f}%  "
                       f"best_val_acc {best*100:5.1f}%"
                       + ("  *" if improved else ""))
    epochs.close()

    print(f"saved best (val acc {best*100:.1f}%) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
