"""Lightweight experiment tracking for the training scripts.

Each run creates logs/{script_name}_{timestamp}/ holding:
  - config.yaml      hyperparameters for the run
  - train.log        human-readable progress log (also echoed to stdout)
  - events.out.*     TensorBoard scalars
  - *.pt             model checkpoints
  - (train_rl only)  reward_weight.yaml snapshot

Usage:
    exp = Experiment("train_rl", config={...})
    exp.log("starting")
    exp.scalar("reward/ema", ema, it)
    torch.save(sd, exp.path("best.pt"))
    exp.close()
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import yaml

try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except Exception:  # tensorboard not installed -> degrade gracefully
    _HAS_TB = False


class Experiment:
    def __init__(self, script_name, config=None, base="logs", timestamp=None):
        ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.name = f"{script_name}_{ts}"
        self.dir = Path(base) / self.name
        self.dir.mkdir(parents=True, exist_ok=True)
        self._log = open(self.dir / "train.log", "a")
        self.writer = SummaryWriter(str(self.dir)) if _HAS_TB else None
        if config is not None:
            self.save_yaml("config.yaml", config)
        self.log(f"run dir: {self.dir}")
        if not _HAS_TB:
            self.log("tensorboard not available -- scalar logging disabled")

    def path(self, name) -> str:
        return str(self.dir / name)

    def save_yaml(self, name, data):
        with open(self.dir / name, "w") as f:
            yaml.safe_dump(dict(data), f, default_flow_style=False, sort_keys=False)

    def log(self, msg):
        print(msg, flush=True)
        self._log.write(msg + "\n")
        self._log.flush()

    def scalar(self, tag, value, step):
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def close(self):
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
        self._log.close()
