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

from ..envs.action_exec import (find_collision, intended_cells,
                                run_action_episode)
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


def _replay_tensors(b, history, adv_mean, adv_std, device):
    """Precompute an episode's fixed (weight-independent) replay tensors.

    Built once and reused across PPO epochs: the stacked occupancy/history inputs,
    the per-step agent cells, the taken actions, and the behavior log-prob /
    normalized advantage / return. Only ``model.encode``/``decode`` (which depend
    on the changing weights) are recomputed per epoch."""
    T = len(b)
    return {
        "T": T,
        "feats": torch.from_numpy(build_features(b.gmap, b.goals))[None].to(device),
        "occ": torch.from_numpy(np.stack(b.occs)).to(device),            # (T,H,W)
        "hist": torch.from_numpy(np.stack(
            [occ_history_window(b.occs, t, history) for t in range(T)])).to(device),
        "rows": torch.tensor([[r for (r, c) in b.cells[t]] for t in range(T)],
                             device=device),                            # (T,N)
        "cols": torch.tensor([[c for (r, c) in b.cells[t]] for t in range(T)],
                             device=device),                            # (T,N)
        "act": torch.from_numpy(np.stack(b.actions)).to(device),        # (T,N)
        "old_logp": torch.from_numpy(b.old_logps).to(device),           # (T,)
        "adv": torch.from_numpy((b.advantages - adv_mean) / adv_std).to(device),
        "ret": torch.from_numpy(b.returns).to(device),
    }


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


def _assemble_rewards(log, potential, weights, *, success, n_reached, collided, n):
    """Per-step progress+time reward plus terminal success/reached/collision."""
    T = len(log) - 1
    rewards = [weights.progress * (potential(log[t + 1]) - potential(log[t]))
               - weights.time_penalty for t in range(T)]
    if T > 0:
        rewards[-1] += (weights.success * float(success)
                        + weights.reached * n_reached / n)
        if collided:
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
    rewards = _assemble_rewards(log, potential, weights, success=res.success,
                                n_reached=res.n_reached, collided=res.collided, n=n)
    final_occ = _occ(log[-1], H, W)
    return ActionEpisodeBuffer(gmap, goals, occs, cells, actions,
                               np.asarray(old_logps, np.float32),
                               np.asarray(old_values, np.float32),
                               rewards, final_occ, bool(res.success),
                               bool(res.collided), res.n_reached)


@torch.no_grad()
def collect_batch_action(model, critic, instances, *, weights, max_steps, device):
    """Vectorized (lockstep) analogue of :func:`collect_episode_action`.

    Runs all ``instances`` as parallel environments: the maps are encoded **once**
    in a single batch, and every timestep does **one** batched decode over the
    still-active episodes (``[A,5,H,W]``) instead of ``A`` separate batch-1 decodes.
    The per-agent categorical sample, the joint log-prob, and the critic value are
    likewise batched; only the (cheap, numpy) executor step stays per-episode. The
    active set shrinks as episodes finish (collision / solved / truncation).

    Produces exactly the same ``list[ActionEpisodeBuffer]`` as calling
    :func:`collect_episode_action` per instance, so
    :func:`compute_gae_action` / :func:`ppo_update_action` are unchanged. All maps
    must share ``(H, W)`` and agent count (true for a training batch).
    """
    model.eval()
    critic.eval()
    B = len(instances)
    gmaps = [g for (g, s, gl) in instances]
    H, W = gmaps[0].H, gmaps[0].W
    n = len(instances[0][1])
    if any(g.H != H or g.W != W for g in gmaps) or any(len(s) != n for (_, s, _) in instances):
        raise ValueError("collect_batch_action needs equal map size and agent count")
    history = model.config["history"]
    norm = float(H + W)

    feats = np.stack([build_features(g) for g in gmaps]).astype(np.float32)
    embs = model.encode(torch.from_numpy(feats).to(device))       # (B, D, H, W)

    goals = [gl for (_, _, gl) in instances]
    goal_dist = [[gmaps[b].bfs_dist(g) for g in goals[b]] for b in range(B)]
    pos = [list(s) for (_, s, _) in instances]

    occs = [[] for _ in range(B)]
    cells = [[] for _ in range(B)]
    actions = [[] for _ in range(B)]
    old_logps = [[] for _ in range(B)]
    old_values = [[] for _ in range(B)]
    logpos = [[list(pos[b])] for b in range(B)]   # positions_log (start first)
    arrival = [[None] * n for _ in range(B)]
    done = [False] * B
    collided = [False] * B
    makespan = [None] * B

    for t in range(1, max_steps + 1):
        active = [b for b in range(B) if not done[b]]
        if not active:
            break
        occ_batch, hist_batch = [], []
        for b in active:
            occ_np = _occ(pos[b], H, W)
            occs[b].append(occ_np)  # append first so the window includes it
            occ_batch.append(occ_np)
            hist_batch.append(occ_history_window(occs[b], len(occs[b]) - 1, history))
        occ_t = torch.from_numpy(np.stack(occ_batch)).to(device)          # (A,H,W)
        hist_t = torch.from_numpy(np.stack(hist_batch)).to(device)        # (A,hist,H,W)
        emb_t = embs[active]                                              # (A,D,H,W)
        logits = model.decode(emb_t, occ_t, hist_t)                      # (A,5,H,W)
        A = len(active)
        rows = torch.tensor([[r for (r, c) in pos[b]] for b in active], device=device)
        cols = torch.tensor([[c for (r, c) in pos[b]] for b in active], device=device)
        ar = torch.arange(A, device=device)[:, None]
        la = logits.permute(0, 2, 3, 1)[ar, rows, cols]                  # (A,N,5)
        dist = Categorical(logits=la)
        samp = dist.sample()                                             # (A,N)
        logp = dist.log_prob(samp).sum(dim=1)                            # (A,)
        values = critic(emb_t, occ_t)                                    # (A,)
        samp_np = samp.cpu().numpy().astype(np.int64)
        logp_np = logp.cpu().numpy()
        val_np = values.cpu().numpy()

        for ai, b in enumerate(active):
            a_np = samp_np[ai]
            cells[b].append(list(pos[b]))
            actions[b].append(a_np)
            old_logps[b].append(float(logp_np[ai]))
            old_values[b].append(float(val_np[ai]))
            nxt = intended_cells(pos[b], a_np)
            if find_collision(gmaps[b], pos[b], a_np, nxt=nxt) is not None:
                collided[b] = True
                done[b] = True
                logpos[b].append(list(pos[b]))   # no move applied
                continue
            pos[b] = nxt
            logpos[b].append(list(pos[b]))
            for i in range(n):
                if pos[b][i] == goals[b][i] and arrival[b][i] is None:
                    arrival[b][i] = t
            if all(pos[b][i] == goals[b][i] for i in range(n)):
                makespan[b] = t
                done[b] = True

    buffers = []
    for b in range(B):
        log = logpos[b]
        gd = goal_dist[b]

        def potential(positions, gd=gd):
            d = sum(float(gd[i][positions[i][0], positions[i][1]]) for i in range(n))
            return -d / (n * norm)

        success = makespan[b] is not None
        n_reached = sum(pos[b][i] == goals[b][i] for i in range(n))
        rewards = _assemble_rewards(log, potential, weights, success=success,
                                    n_reached=n_reached, collided=collided[b], n=n)
        buffers.append(ActionEpisodeBuffer(
            gmaps[b], goals[b], occs[b], cells[b], actions[b],
            np.asarray(old_logps[b], np.float32),
            np.asarray(old_values[b], np.float32),
            rewards, _occ(log[-1], H, W), success, collided[b], n_reached))
    return buffers


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

    # Precompute each episode's fixed replay tensors once (reused every epoch); the
    # per-step timesteps are batched into a single decode below.
    history = model.config["history"]
    prep = [_replay_tensors(b, history, adv_mean, adv_std, device) for b in live]

    stats = defaultdict(float)
    n_updates = 0
    epochs_run = 0
    for _ in range(epochs):
        epoch_kl = []
        epochs_run += 1
        for p in prep:
            T = p["T"]
            emb = model.encode(p["feats"])                    # (1,D,H,W)
            emb_T = emb.expand(T, -1, -1, -1)                 # (T,D,H,W) view, grad->emb
            logits = model.decode(emb_T, p["occ"], p["hist"])  # (T,5,H,W), one forward
            ar = torch.arange(T, device=device)[:, None]
            la = logits.permute(0, 2, 3, 1)[ar, p["rows"], p["cols"]]  # (T,N,5)
            dist = Categorical(logits=la)
            new_logp = dist.log_prob(p["act"]).sum(dim=1)     # (T,)
            entropy = dist.entropy().mean()
            new_value = critic(emb_T.detach(), p["occ"])      # (T,); critic off encoder
            old_logp, adv, ret = p["old_logp"], p["adv"], p["ret"]

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
                          max_steps=256, device="cpu", rng=None, vectorized=True):
    """Collect one batch of episodes (one per instance) and do a PPO update.

    ``instances`` is a list of ``(gmap, starts, goals)``. Collection is
    **vectorized** by default (:func:`collect_batch_action` — all episodes stepped
    in lockstep with one batched decode per timestep); set ``vectorized=False`` for
    the per-episode reference path. Returns a stats dict with update diagnostics
    plus batch reward / success_rate / frac_reached / collision_rate.
    """
    rng = rng or np.random.default_rng()
    if vectorized:
        batch = collect_batch_action(model, critic, instances, weights=weights,
                                     max_steps=max_steps, device=device)
    else:
        batch = [collect_episode_action(model, critic, g, s, gl, weights=weights,
                                        max_steps=max_steps, device=device, rng=rng)
                 for (g, s, gl) in instances]
    ep_rewards, succ, reached, n_ag, collisions = [], 0, 0, 0, 0
    for b in batch:
        compute_gae_action(b, model, critic, gamma, lam, device)
        ep_rewards.append(float(b.rewards.sum()) if len(b) else 0.0)
        succ += b.success
        reached += b.n_reached
        n_ag += len(b.goals)
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
           "collect_episode_action", "collect_batch_action", "compute_gae_action",
           "ppo_update_action", "train_action_ppo_step"]
