"""RL fine-tuning of the priority field (GRPO-style group-baseline REINFORCE).

The action is the whole priority field. For each map we draw K perturbed fields
from a Gaussian around the model's pre-activation, evaluate each on a *shared*
set of start/goal instances (so differences reflect field quality, not luck),
and push the model toward the higher-reward samples via a group-normalized
advantage. Reward rewards success and penalizes makespan.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..envs.grid import GridMap, sample_start_goals
from ..envs.simulator import Simulator
from ..priority.features import build_features
from .reward import DEFAULT_WEIGHTS, RewardWeights


def episode_reward(gmap, samples, field, max_steps, weights=DEFAULT_WEIGHTS):
    """Mean reward of a field over shared start/goal samples.

    Rewards success strongly and, as a dense shaping term, prefers lower makespan
    and flowtime so there is still gradient once success saturates. Shaping
    coefficients come from `weights` (reward_weight.yaml).
    """
    rs = []
    n = len(samples[0][0])
    for starts, goals in samples:
        # training pinned to the legacy beta engine (the existing checkpoints were
        # produced this way); eval/benchmark use the paper-yield default.
        sim = Simulator(gmap, starts, goals, max_steps=max_steps, yield_mode="beta")
        res = sim.run(field)
        rs.append(weights.episode(res.success, res.makespan, res.flowtime, n, max_steps))
    return float(np.mean(rs))


def rl_step(model, maps, opt, device, K=8, sigma=0.5, n_agents=8,
            n_samples=2, max_steps=400, rng=None, anchor=None, anchor_w=0.0,
            pool=None, engine="cpu", weights=DEFAULT_WEIGHTS):
    """One optimization step over a batch of maps. Returns stats dict.

    The B*K episode rollouts are independent. ``engine`` selects how rewards are
    evaluated; the gradient math is identical regardless:
      - "cpu": exact PIBT via episode_reward (optionally over a ``pool``).
      - "vec": GPU-batched VecSim (approximate, non-backtracking -- see rl_vec).
    """
    rng = rng or np.random.default_rng()
    free_masks, feats_list, samples_list = [], [], []
    for gmap in maps:
        feats_list.append(torch.from_numpy(build_features(gmap)))
        free_masks.append(torch.from_numpy((gmap.occ == 0).astype(np.float32)))
        s = [sample_start_goals(gmap, n_agents, rng=rng, min_sep=4)
             for _ in range(n_samples)]
        samples_list.append(s)

    feats = torch.stack(feats_list).to(device)        # (B,C,H,W)
    free = torch.stack(free_masks).to(device)         # (B,H,W)
    logits = model(feats, return_logits=True)         # (B,H,W) pre-activation
    B = len(maps)

    # Sample K perturbed fields per map. Keep the per-map (K,H,W) field arrays for
    # the vec engine, and a flat (b,k) task list for the cpu engine.
    pre_list, fmask_list, fields_list, tasks = [], [], [], []
    for b in range(B):
        a = logits[b]                                 # (H,W) requires grad
        fmask = free[b]
        eps = torch.randn(K, *a.shape, device=device)
        pre = a.detach()[None] + sigma * eps          # (K,H,W) sampled actions
        fields = F.softplus(pre) * fmask              # (K,H,W) >=0
        pre_list.append(pre)
        fmask_list.append(fmask)
        fields_np = fields.cpu().numpy()
        fields_list.append(fields_np)
        for k in range(K):
            tasks.append((maps[b], samples_list[b], fields_np[k], max_steps, weights))

    if engine == "vec":
        from .rl_vec import vec_rewards
        rewards_flat = vec_rewards(maps, samples_list, fields_list,
                                   n_agents, max_steps, device, weights=weights)
    elif pool is not None:
        rewards_flat = np.asarray(pool.starmap(episode_reward, tasks),
                                  dtype=np.float64).reshape(B, K)
    else:
        rewards_flat = np.asarray([episode_reward(*t) for t in tasks],
                                  dtype=np.float64).reshape(B, K)

    total_loss = 0.0
    all_rewards = []
    for b in range(B):
        a = logits[b]
        pre = pre_list[b]
        fmask = fmask_list[b]
        rewards = rewards_flat[b]
        all_rewards.append(rewards.mean())
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
        adv_t = torch.from_numpy(adv.astype(np.float32)).to(device)
        # logprob of sampled action wrt mean a (Gaussian, sigma const):
        #   logp_k = -0.5/sigma^2 * sum_free (pre_k - a)^2
        diff = (pre - a[None]) ** 2 * fmask[None]
        logp = -0.5 / (sigma ** 2) * diff.sum(dim=(-2, -1))   # (K,)
        total_loss = total_loss + -(adv_t * logp).mean()
    loss = total_loss / B

    # Anchor regularization: keep logits close to the imitation init so RL does
    # not catastrophically forget map types it already handles well.
    if anchor is not None and anchor_w > 0:
        with torch.no_grad():
            ref = anchor(feats, return_logits=True)
        loss = loss + anchor_w * F.mse_loss(logits * free, ref * free)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return {"loss": loss.item(), "reward": float(np.mean(all_rewards))}
