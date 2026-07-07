"""Per-step A2C training for the dynamic embedding priority model (option B).

Unlike the field-bandit in :mod:`rl` (one Gaussian-perturbed *static* field per
episode, scored by a scalar episode reward), this trains the occupancy-conditioned
:class:`~src.priority.model_embedding.EmbeddingPriorityModel` with genuine
per-timestep credit assignment:

  * **Action** (per step): a per-agent priority. The decoder's field, read at the
    agent cells, is the Gaussian mean; the sampled priorities are what PIBT orders
    on, so the policy influences *who yields to whom* this step. The paper-mode
    resolution (right-hand rule + yield) is treated as fixed environment dynamics
    (non-differentiable), exactly as in the design notes.
  * **Reward** (per step): potential-based team-progress shaping + a time penalty,
    with a terminal success / fraction-reached bonus. See
    :class:`~src.train.reward.StepRewardWeights`.
  * **Critic**: a state-value baseline sharing the actor's (once-per-map) map
    embedding. Advantage = Monte-Carlo return - value.

The rollout drives the real :class:`Simulator` via ``run(field_fn=...)``, so the
priority the sim consumes and the metrics reported are identical to eval. The
policy closure records per-step ``(logp, value)`` while the sim steps; rewards
are assembled afterward from the position log and the terminal result.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field as dfield

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..envs.simulator import Simulator, oracle_kwargs
from ..priority.features import build_features
from .reward import DEFAULT_STEP_WEIGHTS


class Critic(nn.Module):
    """State-value head over the shared map embedding + live occupancy.

    Mirrors the decoder's inputs (embedding map + occupancy) but pools to a
    single scalar V(state). Kept separate from the policy so the eval/model
    contract is untouched; it shares the actor's embedding at call time, so the
    encoder still runs once per map and gets gradient from both losses.
    """

    def __init__(self, dim=128, occ_dim=16, hidden=128):
        super().__init__()
        self.occ = nn.Conv2d(1, occ_dim, 3, padding=1)
        self.body = nn.Sequential(
            nn.Conv2d(dim + occ_dim, hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 1),
        )

    def forward(self, emb: torch.Tensor, occ: torch.Tensor) -> torch.Tensor:
        """emb (B, D, H, W), occ (B, H, W) -> value (B,)."""
        o = F.relu(self.occ(occ[:, None]))
        h = self.body(torch.cat([emb, o], dim=1))
        return self.head(h).squeeze(-1)


def make_critic(model, hidden=128) -> Critic:
    """Build a Critic matched to ``model``'s embedding dim / occupancy channels."""
    return Critic(dim=model.config["dim"], occ_dim=model.config["occ_dim"],
                  hidden=hidden)


def _rollout(model, critic, gmap, starts, goals, *, sigma, weights, max_steps,
             oracle, device, rng):
    """Run one stochastic episode; return (logps, values, rewards, EpisodeResult).

    ``logps``/``values`` are per-step tensors (grad retained through the shared
    embedding); ``rewards`` is a per-step float list aligned to them.
    """
    n = len(starts)
    H, W = gmap.H, gmap.W
    free = (gmap.occ == 0).astype(np.float32)
    goal_dist = [gmap.bfs_dist(g) for g in goals]
    norm = float(H + W)

    def potential(positions):
        # Phi = -mean normalized distance to goal (higher = closer as a team).
        d = sum(float(goal_dist[i][positions[i][0], positions[i][1]]) for i in range(n))
        return -d / (n * norm)

    feats = torch.from_numpy(build_features(gmap, goals))[None].to(device)
    emb = model.encode(feats)  # (1, D, H, W); reused every step -> trains encoder

    logps, values = [], []

    def field_fn(positions):
        occ_np = np.zeros((H, W), np.float32)
        for (r, c) in positions:
            occ_np[r, c] = 1.0
        occ = torch.from_numpy(occ_np)[None].to(device)

        field = model.decode(emb, occ)[0]                      # (H, W), grad
        rows = [r for (r, c) in positions]
        cols = [c for (r, c) in positions]
        mean = field[rows, cols]                               # (N,), grad
        eps = torch.from_numpy(rng.standard_normal(n).astype(np.float32)).to(device)
        action = mean.detach() + sigma * eps                   # fixed sample
        # Gaussian log-prob of the (fixed) action under N(mean, sigma^2 I).
        logp = (-0.5 / sigma ** 2 * (action - mean) ** 2).sum()
        # Detach emb for the critic so value regression (whose loss is much larger
        # than the policy loss) does not dominate the shared encoder -- the encoder
        # is then shaped purely by the policy gradient.
        value = critic(emb.detach(), occ)[0]                   # scalar, grad


        logps.append(logp)
        values.append(value)

        # Field the sim consumes: deterministic decode, agent cells overwritten
        # by the sampled priorities (PIBT orders on these).
        f = field.detach().cpu().numpy() * free
        a = action.detach().cpu().numpy()
        for k, (r, c) in enumerate(positions):
            f[r, c] = a[k]
        return f

    sim = Simulator(gmap, starts, goals, max_steps=max_steps, log_positions=True,
                    **oracle_kwargs(oracle))
    res = sim.run(field_fn=field_fn, rng=rng)
    log = res.positions_log
    T = len(logps)
    assert T == len(log) - 1, f"call/step misalignment: {T} vs {len(log) - 1}"

    rewards = [weights.progress * (potential(log[t + 1]) - potential(log[t]))
               - weights.time_penalty for t in range(T)]
    if T > 0:  # terminal bonus on the last step
        rewards[-1] += (weights.success * float(res.success)
                        + weights.reached * res.n_reached / n)
    return logps, values, rewards, res


def train_embedding_episode(model, critic, gmap, starts, goals, opt, *,
                            sigma=0.3, gamma=0.99, weights=DEFAULT_STEP_WEIGHTS,
                            value_coef=0.5, entropy_hint=True, max_steps=256,
                            oracle="paper", device="cpu", rng=None):
    """One A2C update from a single stochastic episode. Returns a stats dict."""
    model.train()
    critic.train()
    rng = rng or np.random.default_rng()

    logps, values, rewards, res = _rollout(
        model, critic, gmap, starts, goals, sigma=sigma, weights=weights,
        max_steps=max_steps, oracle=oracle, device=device, rng=rng)

    T = len(logps)
    if T == 0:  # degenerate (all agents already on goal) -- nothing to learn
        return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0,
                "reward": 0.0, "success": bool(res.success),
                "n_reached": res.n_reached, "steps": 0}

    # Monte-Carlo discounted returns (episodes terminate or truncate at max_steps).
    G = np.zeros(T, dtype=np.float32)
    acc = 0.0
    for t in reversed(range(T)):
        acc = rewards[t] + gamma * acc
        G[t] = acc
    Gt = torch.from_numpy(G).to(device)
    V = torch.stack(values)
    logp = torch.stack(logps)

    adv = (Gt - V).detach()
    adv = (adv - adv.mean()) / (adv.std() + 1e-6)   # normalize for stability
    policy_loss = -(adv * logp).mean()
    value_loss = F.mse_loss(V, Gt)
    loss = policy_loss + value_coef * value_loss

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(model.parameters()) + list(critic.parameters()), 1.0)
    opt.step()

    return {"loss": loss.item(), "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(), "reward": float(sum(rewards)),
            "success": bool(res.success), "n_reached": res.n_reached, "steps": T}


# ---------------------------------------------------------------------------
# PPO + GAE (the lower-variance, trust-region upgrade over the A2C step above).
#
# Collection is grad-free: it records, per step, everything needed to *recompute*
# log-probs and values later (occupancy, the agent cells, the sampled action, and
# the behavior-policy log-prob/value). The update then re-encodes each map and
# replays those states over several epochs with a clipped surrogate objective, so
# a single batch is reused safely without the policy running away.
# ---------------------------------------------------------------------------


@dataclass
class EpisodeBuffer:
    """One collected episode: stored transitions + behavior-policy statistics."""
    gmap: object
    goals: list
    occs: list           # list[np.ndarray (H,W)] occupancy at each step
    cells: list          # list[list[(r,c)]] agent cells at each step
    actions: list        # list[np.ndarray (N,)] sampled per-agent priorities
    old_logps: np.ndarray  # (T,) behavior log-prob of the joint action
    old_values: np.ndarray  # (T,) behavior state-value
    rewards: np.ndarray    # (T,) per-step reward
    final_occ: np.ndarray  # (H,W) occupancy at the terminal state (for bootstrap)
    success: bool
    n_reached: int
    advantages: np.ndarray = dfield(default=None)  # filled by compute_gae
    returns: np.ndarray = dfield(default=None)

    def __len__(self):
        return len(self.occs)


def _gauss_logp(action, mean, sigma):
    """log N(action; mean, sigma^2 I) summed over agents (const terms kept; they
    cancel in the PPO ratio but keep the scalar interpretable)."""
    n = mean.shape[0]
    return (-0.5 / sigma ** 2 * (action - mean) ** 2).sum() \
        - n * (np.log(sigma) + 0.5 * np.log(2 * np.pi))


@torch.no_grad()
def collect_episode(model, critic, gmap, starts, goals, *, sigma, weights,
                    max_steps, oracle, device, rng):
    """Run one episode with the current (behavior) policy, grad-free, and record
    everything needed to replay it in :func:`ppo_update`."""
    if sigma <= 0:
        raise ValueError("collect_episode needs sigma > 0 (stochastic behavior "
                         "policy); use embedding_field_fn for greedy rollouts.")
    model.eval()
    critic.eval()
    n = len(starts)
    H, W = gmap.H, gmap.W
    free = (gmap.occ == 0).astype(np.float32)
    goal_dist = [gmap.bfs_dist(g) for g in goals]
    norm = float(H + W)

    def potential(positions):
        d = sum(float(goal_dist[i][positions[i][0], positions[i][1]]) for i in range(n))
        return -d / (n * norm)

    feats = torch.from_numpy(build_features(gmap, goals))[None].to(device)
    emb = model.encode(feats)

    occs, cells, actions, old_logps, old_values = [], [], [], [], []

    def field_fn(positions):
        occ_np = np.zeros((H, W), np.float32)
        for (r, c) in positions:
            occ_np[r, c] = 1.0
        occ = torch.from_numpy(occ_np)[None].to(device)
        field = model.decode(emb, occ)[0]
        rows = [r for (r, c) in positions]
        cols = [c for (r, c) in positions]
        mean = field[rows, cols]
        eps = torch.from_numpy(rng.standard_normal(n).astype(np.float32)).to(device)
        action = mean + sigma * eps
        occs.append(occ_np)
        cells.append(list(positions))
        a_np = action.cpu().numpy()
        actions.append(a_np)
        old_logps.append(float(_gauss_logp(action, mean, sigma)))
        old_values.append(float(critic(emb, occ)[0]))
        f = field.cpu().numpy() * free
        for k, (r, c) in enumerate(positions):
            f[r, c] = a_np[k]
        return f

    sim = Simulator(gmap, starts, goals, max_steps=max_steps, log_positions=True,
                    **oracle_kwargs(oracle))
    res = sim.run(field_fn=field_fn, rng=rng)
    log = res.positions_log
    T = len(occs)
    rewards = [weights.progress * (potential(log[t + 1]) - potential(log[t]))
               - weights.time_penalty for t in range(T)]
    if T > 0:
        rewards[-1] += (weights.success * float(res.success)
                        + weights.reached * res.n_reached / n)
    final_occ = np.zeros((H, W), np.float32)
    for (r, c) in log[-1]:
        final_occ[r, c] = 1.0
    return EpisodeBuffer(gmap, goals, occs, cells, actions,
                         np.asarray(old_logps, np.float32),
                         np.asarray(old_values, np.float32),
                         np.asarray(rewards, np.float32), final_occ,
                         bool(res.success), res.n_reached)


@torch.no_grad()
def compute_gae(buf: EpisodeBuffer, model, critic, gamma, lam, device):
    """Fill ``buf.advantages`` (GAE-lambda) and ``buf.returns`` (advantage+value).

    Bootstrap value at the final state: 0 if the episode solved (no future
    reward), else V(final occupancy) so truncation at max_steps is unbiased.
    """
    T = len(buf)
    if T == 0:
        buf.advantages = np.zeros(0, np.float32)
        buf.returns = np.zeros(0, np.float32)
        return buf
    if buf.success:
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


def ppo_update(model, critic, opt, batch, *, sigma, clip=0.2, value_coef=0.5,
               epochs=4, target_kl=None, device="cpu"):
    """Clipped PPO update over a batch of collected episodes.

    Advantages are normalized across the whole batch. Each episode is one
    minibatch: its map is re-encoded (grad) and its stored states replayed to
    recompute new log-probs/values, then a clipped surrogate + value loss steps
    the optimizer. Repeated for ``epochs``.

    ``target_kl`` (if set) stops the epoch loop early once an epoch's mean
    |approx_kl| exceeds it — a standard PPO trust-region guard that prevents the
    late-training over-stepping (clip_frac creeping toward 0.5) seen without it.
    """
    model.train()
    critic.train()
    live = [b for b in batch if len(b) > 0]
    if not live:
        return {"policy_loss": 0.0, "value_loss": 0.0, "clip_frac": 0.0,
                "approx_kl": 0.0, "epochs_run": 0}
    all_adv = np.concatenate([b.advantages for b in live])
    adv_mean, adv_std = float(all_adv.mean()), float(all_adv.std()) + 1e-6

    stats = defaultdict(float)
    n_updates = 0
    epochs_run = 0
    for _ in range(epochs):
        epoch_kl = []
        epochs_run += 1
        for b in live:
            T = len(b)
            feats = torch.from_numpy(build_features(b.gmap, b.goals))[None].to(device)
            emb = model.encode(feats)
            new_logps, new_values = [], []
            for t in range(T):
                occ = torch.from_numpy(b.occs[t])[None].to(device)
                field = model.decode(emb, occ)[0]
                rows = [r for (r, c) in b.cells[t]]
                cols = [c for (r, c) in b.cells[t]]
                mean = field[rows, cols]
                action = torch.from_numpy(b.actions[t]).to(device)
                new_logps.append(_gauss_logp(action, mean, sigma))
                new_values.append(critic(emb.detach(), occ)[0])
            new_logp = torch.stack(new_logps)
            new_value = torch.stack(new_values)
            old_logp = torch.from_numpy(b.old_logps).to(device)
            adv = torch.from_numpy((b.advantages - adv_mean) / adv_std).to(device)
            ret = torch.from_numpy(b.returns).to(device)

            ratio = torch.exp(new_logp - old_logp)
            surr = torch.min(ratio * adv,
                             torch.clamp(ratio, 1 - clip, 1 + clip) * adv)
            policy_loss = -surr.mean()
            value_loss = F.mse_loss(new_value, ret)
            loss = policy_loss + value_coef * value_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(critic.parameters()), 1.0)
            opt.step()

            with torch.no_grad():
                kl = float((old_logp - new_logp).mean())
                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["clip_frac"] += float(((ratio - 1.0).abs() > clip).float().mean())
                stats["approx_kl"] += kl
                epoch_kl.append(kl)
            n_updates += 1

        if target_kl is not None and np.mean(np.abs(epoch_kl)) > target_kl:
            break  # trust-region guard: stop before over-stepping

    out = {k: v / max(n_updates, 1) for k, v in stats.items()}
    out["epochs_run"] = epochs_run
    return out


def train_embedding_ppo_step(model, critic, instances, opt, *, sigma,
                             gamma=0.99, lam=0.95, weights=DEFAULT_STEP_WEIGHTS,
                             clip=0.2, value_coef=0.5, epochs=4, target_kl=None,
                             max_steps=256, oracle="paper", device="cpu", rng=None):
    """Collect one batch of episodes (one per instance) and do a PPO update.

    ``instances`` is a list of ``(gmap, starts, goals)``. Returns a stats dict
    with update diagnostics plus batch reward / success_rate / frac_reached.
    """
    rng = rng or np.random.default_rng()
    batch, ep_rewards = [], []
    succ, reached, n_ag = 0, 0, 0
    for (gmap, starts, goals) in instances:
        b = collect_episode(model, critic, gmap, starts, goals, sigma=sigma,
                            weights=weights, max_steps=max_steps, oracle=oracle,
                            device=device, rng=rng)
        compute_gae(b, model, critic, gamma, lam, device)
        batch.append(b)
        ep_rewards.append(float(b.rewards.sum()) if len(b) else 0.0)
        succ += b.success
        reached += b.n_reached
        n_ag += len(starts)
    out = ppo_update(model, critic, opt, batch, sigma=sigma, clip=clip,
                     value_coef=value_coef, epochs=epochs, target_kl=target_kl,
                     device=device)
    out.update({"reward": float(np.mean(ep_rewards)),
                "success_rate": succ / len(instances),
                "frac_reached": reached / max(n_ag, 1)})
    return out


__all__ = ["Critic", "make_critic", "train_embedding_episode",
           "EpisodeBuffer", "collect_episode", "compute_gae", "ppo_update",
           "train_embedding_ppo_step"]
