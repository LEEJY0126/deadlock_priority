# Embedding priority (`feature/embedding`)

A **dynamic, occupancy-conditioned priority** built on a learned map embedding.
The static `PriorityUNet` field is replaced by two models — a **MapEncoder**
(map → shared embedding, once per map) and a **PriorityDecoder** (embedding +
live occupancy → priority field, every step) — and the priority is trained with
**per-step A2C**. Still the comms-free global-priority + PIBT paradigm of the main
project (NOT the separate `deadlock_embedding` POMDP repo); only the priority
signal changes, from static to dynamic. Branched off `feature/goal-livelock`.

---

## 1. Motivation

The shipped priority field is **static**: one `(H,W)` field per map, an agent's
priority is its cell value. A static field cannot break a symmetric standoff on
its own — which is why the runtime needs either the `beta` stuck-boost or the
`paper` yield to make progress in 1-wide corridors. The hypothesis of this branch:
a priority that is **conditioned on the live agent configuration** can express
context the static field cannot ("in *this* jam, agent A should outrank B"),
targeting deadlock directly rather than relying on a hand-designed booster.

This trades away two properties of the static field, deliberately: the priority
is no longer computable offline (it depends on live occupancy), and — because it
is recomputed per step from occupancy — the comms-free invariant now requires all
agents to observe the same occupancy (true under the project's global-observation
assumption; would break under FOV, an explicit future concern).

## 2. Architecture (as implemented)

**MapEncoder** (`src/priority/model_embedding.py`). Map features `[C,H,W]`
(the existing `build_features`, now 4 channels: free/clearance/row/col — the
goal-heatmap channel was removed, so the field is purely structural and never
sees goal positions) → shared embedding `[emb,H,W]`. Reuses the
`model_transformer` trunk (conv stem +
`sinusoidal_pe_2d` + `TransformerEncoder`) minus the scalar head, fusing the
global-attended tokens with the local stem skip. **Map-only input**, so the
embedding is identical for every agent — the comms-free invariant is preserved —
and it is meant to run **once per map**.

**PriorityDecoder** (`model_embedding.py`). Embedding `[emb,H,W]` + live
occupancy `[H,W]` (1 = an agent stands here) + **occupancy history**
`[history,H,W]` → priority **field** `[H,W]`. Occupancy is lifted with a small
conv (a cell sees *nearby* agents); the history is encoded by a `HistoryEncoder`
(conv stem over the `history` frames + shallow spatial transformer →
`[hist_dim,H,W]`) and projected; both are fused with the embedding, then a
shallow transformer + per-cell softplus head. Cheap; runs **per step**.

The history gives the decoder *temporal* signal — motion direction and how long
a cell has been occupied (stuck-time) — the very information `beta` hand-codes,
now learnable. It is **stateful in the rollout**: `embedding_field_fn` keeps a
per-episode occupancy buffer and `occ_history_window(occs, t, history)` builds
the window ending at step `t`, front-padded with the start frame for early steps.
The *same* window helper is used at PPO replay time (over the stored `occs`), so
collection and update see identical inputs (verified: replay `|ratio-1| ~ 1e-7`).
The critic stays occupancy-only — the value baseline does not need history.

**Field-then-index.** Each agent reads its scalar priority by indexing the field
at its own cell (`agent_priorities`). This was chosen over a raw `[N]` head: a
binary occupancy map cannot attribute a `[N]`-vector slot to a specific agent, and
a fixed `[N]` head breaks when the agent count varies. Indexing a shared field is
agnostic to count/order and stays a drop-in for the PIBT consumer, which already
indexes a field per agent. The occupancy input is what makes the field dynamic.

**Simulator integration** (`src/envs/simulator.py`). `Simulator.run` gained an
optional `field_fn(positions) -> (H,W)` callback: when present, the field is
recomputed and re-normalized (`_normalize_field`, same zero-mean/unit-std as the
static path) **once per step** from the current occupancy. The simulator stays
torch-free — the closure lives in `model_embedding.embedding_field_fn`, which
encodes the map once and runs only the cheap decoder per call. Runs under
`yield_mode="paper"`.

**Resolution mode: `paper`, not `beta`** (deliberate). `beta` bakes anti-starvation
*into* the priority (`prio += beta·stuck`), so the decoder would need stuck-time /
history to replace it. `paper` decouples liveness (right-hand rule + explicit
yield, Alg. 3) from priority, so the decoder only has to learn contextual
**ordering** and the liveness guarantee stays in verified code. Cost: the priority
has less leverage under `paper` (the resolution does much of the work), so expect
smaller margins — and, empirically, a weaker training signal (§5).

## 3. Training — per-step A2C ("option B")

Chosen over the existing field-bandit (`src/train/rl.py`), which treats the whole
static field as one action scored by a scalar episode reward. For a dynamic,
per-step field that framing does not fit, and the reward design below wants
genuine per-timestep credit.

**Action (per step).** A per-agent priority. The decoder field read at the agent
cells is the Gaussian mean; the sampled priorities (`mean + σ·ε`, detached) are
written into the field the sim consumes, so PIBT orders on them. The paper-mode
resolution is treated as fixed, non-differentiable environment dynamics.

**Reward (`StepRewardWeights` in `src/train/reward.py`).**
- `progress`: potential-based shaping, `progress·(Φ_t − Φ_{t−1})`, `Φ = −mean
  normalized team distance-to-goal`. Potential-based ⇒ policy-invariant (Ng et al.
  1999). Design notes that fed this choice: cumulative progress *telescopes* to
  (init−final) distance, so it is constant among successes (cancels in the
  advantage) and only bites in the **failure** regime; use **team Σ-distance**,
  never "all agents progress" (that would punish yielding, the whole point of
  priority); a per-step time penalty **is** the makespan term — don't double-count.
- `time_penalty`: small negative per step.
- terminal `success` + `reached` (fraction of agents at goal — partial credit).

**Critic** (`src/train/rl_embedding.py`). A state-value head over the shared
embedding + occupancy, giving the advantage baseline. It uses a **detached**
embedding: value regression's loss (~30) dwarfs the policy loss (~0.4), so an
un-detached critic dominates the shared encoder; detaching lets the encoder be
shaped purely by the policy gradient.

**A2C update (baseline path).** Monte-Carlo discounted returns, advantage =
return − value (normalized per episode), `policy_loss + value_coef·value_loss`,
grad-clipped. Single episode per update. This is `train_embedding_episode`;
kept as `--algo a2c`. It is mechanically correct but did not learn (§5).

**PPO + GAE (default path).** The variance/trust-region upgrade that does learn:
- **Collection is grad-free** (`collect_episode`): records per step the occupancy,
  agent cells, sampled action, and behavior log-prob/value — everything needed to
  *replay* the episode later.
- **`compute_gae`**: GAE(λ) advantages with a value **bootstrap** at truncation
  (0 if solved, else V(final)), so max-step cutoffs are unbiased.
- **`ppo_update`**: several epochs over the batch, re-encoding each map (grad) and
  replaying stored states to recompute log-probs/values, with the clipped
  surrogate `min(ratio·A, clip(ratio,1±ε)·A)` + value loss. Batch-normalized
  advantages; reports `approx_kl` / `clip_frac`.
- **σ-annealing** (driver): exploration std decays linearly (broad early, refine
  late). A **batch of episodes** per update further cuts variance.

Driver: `scripts/train_embedding_rl.py` (`--algo ppo` default; balanced random
maps, periodic greedy benchmark vs MST, saves an `arch="embedding"` ckpt that
`evaluate.py --ckpt` auto-routes). **Always run with `python3 -u`** — buffered
stdout hid all metrics on the first attempt.

## 4. Implementation history (this session)

Core committed as `581b7a8` (`bcd1374` added the sim script + docs) on
`feature/embedding`; the goal-channel removal (step 11) is a later change. The
order in which the design settled and the fixes that mattered.

| step | change | why |
|------|--------|-----|
| 1 | split `PriorityUNet` → `MapEncoder` + `PriorityDecoder`, reusing the transformer trunk; `build_model("embedding")` | separate once-per-map map understanding from the per-step decision |
| 2 | **field-then-index** output (`[H,W]` field, index per agent) instead of a raw `[N]` head | a binary occupancy map can't attribute `[N]` slots to agents; indexing is count/order-agnostic and a PIBT drop-in |
| 3 | `Simulator.run(field_fn=...)` recomputes the field per step; `embedding_field_fn` closure (encode once, decode per step) | make priority dynamic without pulling torch into the simulator |
| 4 | chose **`paper`** resolution over `beta` | decouples liveness from priority so the decoder only learns ordering; keeps the guarantee in verified code |
| 5 | wired eval: `benchmark.evaluate_embedding` + `evaluate.py --embedding`/auto-route | apples-to-apples A/B vs MST on the same instances |
| 6 | **bug fix:** `run` called `field_fn` twice on step 1 (bootstrap + first loop iter) → now once per step; `prev_base` bootstrapped lazily | RL needs action/step counts to align 1:1 (verified `call[t] == log[t]`) |
| 7 | per-step A2C: `StepRewardWeights`, `Critic`, `train_embedding_episode`, driver | option B — genuine per-timestep credit for the dynamic field |
| 8 | **fix:** critic uses a **detached** embedding | value loss ≫ policy loss was dominating the shared encoder |
| 9 | A2C didn't learn (flat / regressing) → **PPO + GAE**: `collect_episode`, `compute_gae`, `ppo_update`, `train_embedding_ppo_step`, σ-anneal | variance reduction (GAE + batched episodes) + a trust region (clipping) — the two things A2C lacked |
| 10 | **observability fix:** run training with `python3 -u` / `flush=True` | buffered stdout hid every metric on the first PPO run — was flying blind |
| 11 | **removed the goal-heatmap feature channel** (`N_CHANNELS` 5→4; free/clearance/row/col only) | test whether dynamic priority needs to see goals; the field is now purely structural. Breaking for 5-channel checkpoints. |
| 12 | **occupancy-history input** to the decoder: `HistoryEncoder` (`[history,H,W]`→`[hist_dim,H,W]`) + stateful `occ_history_window`; threaded through field_fn / collect / replay | give the decoder temporal signal (motion, stuck-time) that a single snapshot lacks — the info `beta` hand-codes, now learnable. Replay reuses the same window (`\|ratio-1\|~1e-7`). |

## 5. Results

> **The numbers below predate step 11 (goal-channel removal).** They were
> measured with the 5-channel features (goal heatmap on). Re-run after retraining
> on the 4-channel features to get current numbers.

**Untrained embedding vs baselines** (n=24: n_per_kind=6 × n_inst=4, 8 agents,
21×21, `paper`). Success rate:

| map | PIBT (elapsed) | MST | Embedding (untrained) |
|-----|:--------------:|:---:|:---------------------:|
| forest | 100.0 | 95.8 | 100.0 |
| wide   | 95.8  | 91.7 | 91.7 |
| narrow | 87.5  | 45.8 | 50.0 |

An **untrained** (random) field ≈ MST — which mostly confirms that `paper`
resolution carries the load and a normalized random field is a roughly neutral
prior, **not** that the architecture is good. n=24 is noisy (use ≥12/kind).

**A2C: mechanically correct, did not learn.** Grads reach encoder/decoder/critic,
params move, checkpoints round-trip — but no learning win:

| regime | greedy success before → after | note |
|--------|:-----------------------------:|------|
| narrow (has headroom) | 50 → 50 | no improvement in ~300 CPU episodes |
| forest+wide | 100 → 83 | already saturated; exploration noise *degraded* a good policy |

High variance (single episode/update) + no trust region ⇒ updates random-walk the
policy.

**PPO + GAE: healthy and learning.** Overfitting a fixed set of hard narrow
instances (6 × 13×13, 6 agents, 45 PPO updates, batch 6, epochs 4, σ 0.5→0.15,
`paper`, dim 32 — a light *observable* diagnostic, CPU):

| metric | start | end (peak) |
|--------|:-----:|:----------:|
| greedy success | 0.33 | **0.50** (0.67 @ it30) |
| greedy frac-reached | 0.72 | **0.89** |
| value loss | 4.9 | ~1.0 |

Trust region behaved: `approx_kl` ~0.01 early rising to ~0.05–0.10 late,
`clip_frac` 0.22 → ~0.5. The critic converged and greedy success climbed off the
floor — the win A2C could not get. Caveats: the greedy metric is quantized/noisy
on a 6-instance set (peak 0.67 @ it30, settled 0.50 → **keep best-by-eval, not
last**); rising `clip_frac`/`|kl|` late says the step size is high (lower LR / 2–3
epochs / KL early-stop for longer runs); and this is *overfitting* to prove
learnability — a real result needs training on fresh maps with held-out eval on GPU.

## 6. Limitations / open work

- **Learning shown only on a fixed set (overfit).** PPO+GAE learns (§5), but
  generalization is untested — the open item is a longer GPU run on **fresh maps**
  with held-out eval. (Two of the earlier gaps are now closed: the driver saves a
  **best-by-eval** `*_best.pt` alongside the last checkpoint, and PPO has a
  `--target_kl` early-stop that caps the late-training over-stepping —
  `clip_frac`→0.5 / `|kl|`→0.1 — by cutting epochs once an epoch's mean |kl|
  exceeds the target.)
- **`paper` leverage.** Because resolution does much of the work, the ceiling on
  how much a learned priority can move the numbers may be modest (same caveat that
  made `paper` the right *safety* choice makes it a weaker *training* signal).
- **No imitation warm-start.** The static-field oracle produces no per-frame
  labels, so there is no cheap IL bootstrap; RL starts cold.
- **Dynamic-ordering oscillation.** A field that flips ordering frame-to-frame can
  interact with the yield/oscillation detection — watch it (the `goal-livelock`
  theme).
- **Global-observation only.** Dynamic priority from occupancy is comms-free only
  if all agents share the same occupancy; FOV would break it.

## 7. Reproduce

Full flag reference for the branch's scripts + checkpoint/reward-YAML formats:
[`embedding-scripts_usage.md`](embedding-scripts_usage.md).

```bash
# tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_model_embedding.py tests/test_rl_embedding.py -q

# untrained embedding vs MST vs PIBT
python3 scripts/evaluate.py --embedding --n_per_kind 6 --n_inst 4

# train (PPO+GAE by default); saves runs/rl_embedding.pt + runs/rl_embedding_best.pt,
# auto-routed by evaluate.py. Use `python3 -u` so the log flushes live.
python3 -u scripts/train_embedding_rl.py --iters 600 --device cuda > runs/train_embedding.log 2>&1 &
python3 scripts/evaluate.py --ckpt runs/rl_embedding_best.pt --n_per_kind 12 --n_inst 5
```

## 8. Action-map variant (`embedding_action`)

A follow-on takes this one step further: instead of a scalar priority field
consumed by PIBT, the model emits a **per-cell action distribution** `[5, H, W]`
over `[UP, DOWN, LEFT, RIGHT, STAY]` and moves agents **directly** — PIBT is gone,
so the policy itself must avoid collisions (a collision ends the episode as a
failure with a penalty). It reuses this branch's trunk (`MapEncoder` /
`HistoryEncoder`) and PPO machinery unchanged; only the decoder head and the
executor differ.

That variant has its own branch (`feature/embedding_action`) and dedicated docs:
**[`embedding_action.md`](embedding_action.md)** (design) and
**[`embedding_action-script_usage.md`](embedding_action-script_usage.md)** (scripts,
flags, checkpoint format).
