"""Configurable reward weights for RL fine-tuning.

Loaded from reward_weight.yaml so the reward shaping can be tuned without code
changes and snapshotted per experiment run.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import yaml


@dataclass
class RewardWeights:
    success: float = 2.0
    makespan: float = 0.5
    flowtime: float = 0.5

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        known = {k: float(d[k]) for k in ("success", "makespan", "flowtime") if k in d}
        return cls(**known)

    def to_dict(self):
        return asdict(self)

    def episode(self, success, makespan, flowtime, n_agents, max_steps):
        """Reward for one episode (scalars or numpy arrays)."""
        return (self.success * success
                - self.makespan * makespan / max_steps
                - self.flowtime * flowtime / (n_agents * max_steps))


DEFAULT_WEIGHTS = RewardWeights()
