"""Per-step PPO for the dynamic action-map policy (:class:`EmbeddingActionModel`).

The action-map sibling of :mod:`rl_embedding`. Instead of a Gaussian over
per-agent priorities feeding PIBT, the policy is a **categorical** over each
agent's five moves ``[UP, DOWN, LEFT, RIGHT, STAY]`` (read from the decoder's
per-cell action field at the agent cell), executed directly by
:func:`src.envs.action_exec.run_action_episode`. There is no PIBT, so collisions
end the episode; the reward adds a terminal collision penalty
(:class:`~src.train.reward.StepRewardWeights`).

Structure mirrors the PPO+GAE path in :mod:`rl_embedding` (grad-free collection
that records everything to replay; a clipped multi-epoch update with KL
early-stop), and **reuses** its :class:`~src.train.rl_embedding.Critic`,
``make_critic``, and ``occ_history_window``. Exploration comes from categorical
sampling plus an entropy bonus (there is no ``sigma``).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field as dfield

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from ..envs.action_exec import run_action_episode
from ..priority.features import build_features
from ..priority.model_embedding import occ_history_window
from .reward import DEFAULT_STEP_WEIGHTS
from .rl_embedding import Critic, make_critic  # reuse the state-value critic


def _occ(positions, H, W):
    occ = np.zeros((H, W), np.float32)
    for (r, c) in positions:
        occ[r, c] = 1.0
    return occ


def _agent_logits(logits_field, positions):
    """(5,H,W) tensor + [(r,c)] -> (N,5) tensor of per-agent action logits."""
    rows = [r for (r, c) in positions]
    cols = [c for (r, c) in positions]
    return logits_field[:, rows, cols].transpose(0, 1)   # (N, 5), grad-preserving


@dataclass
class ActionEpisodeBuffer:
    """One collected episode: transitions + behavior-policy statistics."""
    gmap: object
    goals: list
    occs: list            # list[np.ndarray (H,W)] occupancy at each step
    cells: list           # list[list[(r,c)]] agent cells at each step
    actions: list         # list[np.ndarray (N,) int] sampled action indices
    old_logps: np.ndarray  # (T,) behavior log-prob of the joint action
    old_values: np.ndarray  # (T,) behavior state-value
    rewards: np.ndarray    # (T,) per-step reward
    final_occ: np.ndarray  # (H,W) occupancy at the terminal state (for bootstrap)
    success: bool
    collided: bool
    n_reached: int
    advantages: np.ndarray = dfield(default=None)  # filled by compute_gae_action
    returns: np.ndarray = dfield(default=None)

    def __len__(self):
        return len(self.occs)


def _assemble_rewards(log, potential, weights, res, n):
    """Per-step progress+time reward plus terminal success/reached/collision."""
    T = len(log) - 1
    rewards = [weights.progress * (potential(log[t + 1]) - potential(log[t]))
               - weights.time_penalty for t in range(T)]
    if T > 0:
        rewards[-1] += (weights.success * float(res.success)
                        + weights.reached * res.n_reached / n)
        if res.collided:
            rewards[-1] += weights.collision
    return np.asarray(rewards, np.float32)


@torch.no_grad()
def collect_episode_action(model, critic, gmap, starts, goals, *, weights,
                           max_steps, device, rng):
    """Run one episode with the current (sampling) policy, grad-free, recording
    everything needed to replay it in :func:`ppo_update_action`."""
    model.eval()
    critic.eval()
    n = len(starts)
    H, W = gmap.H, gmap.W
    goal_dist = [gmap.bfs_dist(g) for g in goals]
    norm = float(H + W)

    def potential(positions):
        d = sum(float(goal_dist[i][positions[i][0], positions[i][1]]) for i in range(n))
        return -d / (n * norm)

    feats = torch.from_numpy(build_features(gmap, goals))[None].to(device)
    emb = model.encode(feats)
    history = model.config["history"]

    occs, cells, actions, old_logps, old_values = [], [], [], [], []

    def action_fn(positions, t):
        occ_np = _occ(positions, H, W)
        occs.append(occ_np)  # append first so the history window includes it
        occ = torch.from_numpy(occ_np)[None].to(device)
        win = occ_history_window(occs, len(occs) - 1, history)
        hist = torch.from_numpy(win)[None].to(device)
        logits_field = model.decode(emb, occ, hist)[0]         # (5, H, W)
        la = _agent_logits(logits_field, positions)            # (N, 5)
        dist = Categorical(logits=la)
        a = dist.sample()                                      # (N,)
        cells.append(list(positions))
        actions.append(a.cpu().numpy().astype(np.int64))
        old_logps.append(float(dist.log_prob(a).sum()))
        old_values.append(float(critic(emb, occ)[0]))
        return a.cpu().numpy()

    res = run_action_episode(gmap, starts, goals, action_fn, max_steps=max_steps)
    log = res.positions_log
    T = len(occs)
    assert T == len(log) - 1, f"call/step misalignment: {T} vs {len(log) - 1}"
    rewards = _assemble_rewards(log, potential, weights, res, n)
    final_occ = _occ(log[-1], H, W)
    return ActionEpisodeBuffer(gmap, goals, occs, cells, actions,
                               np.asarray(old_logps, np.float32),
                               np.asarray(old_values, np.float32),
                               rewards, final_occ, bool(res.success),
                               bool(res.collided), res.n_reached)


@torch.no_grad()
def compute_gae_action(buf: ActionEpisodeBuffer, model, critic, gamma, lam, device):
    """Fill ``buf.advantages`` (GAE-lambda) and ``buf.returns``.

    Bootstrap value at the final state is 0 when the episode **solved or
    collided** (both are true terminals with no future reward), else
    ``V(final occupancy)`` so truncation at ``max_steps`` stays unbiased.
    """
    T = len(buf)
    if T == 0:
        buf.advantages = np.zeros(0, np.float32)
        buf.returns = np.zeros(0, np.float32)
        return buf
    if buf.success or buf.collided:
        boot = 0.0
    else:
        feats = torch.from_numpy(build_features(buf.gmap, buf.goals))[None].to(device)
        emb = model.encode(feats)
        occ = torch.from_numpy(buf.final_occ)[None].to(device)
        boot = float(critic(emb, occ)[0])
    values = np.append(buf.old_values, np.float32(boot))
    adv = np.zeros(T, np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        delta = buf.rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
    buf.advantages = adv
    buf.returns = adv + buf.old_values
    return buf


def ppo_update_action(model, critic, opt, batch, *, clip=0.2, value_coef=0.5,
                      entropy_coef=0.01, epochs=4, target_kl=None, device="cpu"):
    """Clipped PPO update over a batch of collected action episodes.

    Each episode's map is re-encoded (grad) and its stored states replayed to
    recompute categorical log-probs / entropy / values, then a clipped surrogate
    minus an entropy bonus plus a value loss steps the optimizer, for ``epochs``
    (stopping early once an epoch's mean |approx_kl| exceeds ``target_kl``).
    """
    model.train()
    critic.train()
    live = [b for b in batch if len(b) > 0]
    if not live:
        return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                "clip_frac": 0.0, "approx_kl": 0.0, "epochs_run": 0}
    all_adv = np.concatenate([b.advantages for b in live])
    adv_mean, adv_std = float(all_adv.mean()), float(all_adv.std()) + 1e-6

    stats = defaultdict(float)
    n_updates = 0
    epochs_run = 0
    history = model.config["history"]
    for _ in range(epochs):
        epoch_kl = []
        epochs_run += 1
        for b in live:
            T = len(b)
            feats = torch.from_numpy(build_features(b.gmap, b.goals))[None].to(device)
            emb = model.encode(feats)
            new_logps, new_values, entropies = [], [], []
            for t in range(T):
                occ = torch.from_numpy(b.occs[t])[None].to(device)
                win = occ_history_window(b.occs, t, history)  # same window as rollout
                hist = torch.from_numpy(win)[None].to(device)
                logits_field = model.decode(emb, occ, hist)[0]
                la = _agent_logits(logits_field, b.cells[t])   # (N, 5)
                dist = Categorical(logits=la)
                action = torch.from_numpy(b.actions[t]).to(device)
                new_logps.append(dist.log_prob(action).sum())
                entropies.append(dist.entropy().mean())
                new_values.append(critic(emb.detach(), occ)[0])
            new_logp = torch.stack(new_logps)
            new_value = torch.stack(new_values)
            entropy = torch.stack(entropies).mean()
            old_logp = torch.from_numpy(b.old_logps).to(device)
            adv = torch.from_numpy((b.advantages - adv_mean) / adv_std).to(device)
            ret = torch.from_numpy(b.returns).to(device)

            ratio = torch.exp(new_logp - old_logp)
            surr = torch.min(ratio * adv,
                             torch.clamp(ratio, 1 - clip, 1 + clip) * adv)
            policy_loss = -surr.mean()
            value_loss = F.mse_loss(new_value, ret)
            loss = policy_loss - entropy_coef * entropy + value_coef * value_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(critic.parameters()), 1.0)
            opt.step()

            with torch.no_grad():
                kl = float((old_logp - new_logp).mean())
                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += float(entropy)
                stats["clip_frac"] += float(((ratio - 1.0).abs() > clip).float().mean())
                stats["approx_kl"] += kl
                epoch_kl.append(kl)
            n_updates += 1

        if target_kl is not None and np.mean(np.abs(epoch_kl)) > target_kl:
            break  # trust-region guard

    out = {k: v / max(n_updates, 1) for k, v in stats.items()}
    out["epochs_run"] = epochs_run
    return out


def train_action_ppo_step(model, critic, instances, opt, *, gamma=0.99, lam=0.95,
                          weights=DEFAULT_STEP_WEIGHTS, clip=0.2, value_coef=0.5,
                          entropy_coef=0.01, epochs=4, target_kl=None,
                          max_steps=256, device="cpu", rng=None):
    """Collect one batch of episodes (one per instance) and do a PPO update.

    ``instances`` is a list of ``(gmap, starts, goals)``. Returns a stats dict
    with update diagnostics plus batch reward / success_rate / frac_reached /
    collision_rate.
    """
    rng = rng or np.random.default_rng()
    batch, ep_rewards = [], []
    succ, reached, n_ag, collisions = 0, 0, 0, 0
    for (gmap, starts, goals) in instances:
        b = collect_episode_action(model, critic, gmap, starts, goals,
                                   weights=weights, max_steps=max_steps,
                                   device=device, rng=rng)
        compute_gae_action(b, model, critic, gamma, lam, device)
        batch.append(b)
        ep_rewards.append(float(b.rewards.sum()) if len(b) else 0.0)
        succ += b.success
        reached += b.n_reached
        n_ag += len(starts)
        collisions += b.collided
    out = ppo_update_action(model, critic, opt, batch, clip=clip,
                            value_coef=value_coef, entropy_coef=entropy_coef,
                            epochs=epochs, target_kl=target_kl, device=device)
    out.update({"reward": float(np.mean(ep_rewards)),
                "success_rate": succ / len(instances),
                "frac_reached": reached / max(n_ag, 1),
                "collision_rate": collisions / len(instances)})
    return out


__all__ = ["Critic", "make_critic", "ActionEpisodeBuffer",
           "collect_episode_action", "compute_gae_action", "ppo_update_action",
           "train_action_ppo_step"]
