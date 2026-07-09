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


@dataclass
class StepRewardWeights:
    """Per-step reward shaping for the dynamic embedding model's A2C training.

    Unlike :class:`RewardWeights` (one scalar per episode for the field bandit),
    these terms are emitted *every timestep* so credit is assigned per step:

      - ``progress``: potential-based shaping on team distance-to-goal. The
        per-step term is ``progress * (Phi_t - Phi_{t-1})`` with potential
        ``Phi = -mean_agent_distance`` (normalized), so it is positive exactly
        when the *team* got closer. Potential-based => does not change the
        optimal policy (Ng et al. 1999), and it credits partial progress on the
        failure-heavy narrow maps where success is sparse.
      - ``time_penalty``: a small negative every step, i.e. the makespan term as
        a per-step primitive (do not also add an episode makespan penalty).
      - ``success`` / ``reached``: terminal only -- a bonus for solving the whole
        instance plus a fraction for how many agents reached goal (partial credit).
      - ``collision``: terminal only, for the action-map policy (:mod:`src.envs.
        action_exec`). Applied (negative) when an episode ends on a collision;
        mirrors ``success`` in magnitude so a collision roughly cancels a solve.
        Unused by the PIBT priority path (which is collision-free by construction).
    """
    progress: float = 1.0
    time_penalty: float = 0.01
    success: float = 5.0
    reached: float = 1.0
    collision: float = -5.0

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        known = {k: float(d[k]) for k in
                 ("progress", "time_penalty", "success", "reached", "collision")
                 if k in d}
        return cls(**known)

    def to_dict(self):
        return asdict(self)


DEFAULT_STEP_WEIGHTS = StepRewardWeights()
