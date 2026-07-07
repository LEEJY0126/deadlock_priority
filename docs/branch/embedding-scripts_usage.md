# Scripts Usage — `feature/embedding`

Companion to [`docs/scripts_usage.md`](../scripts_usage.md) for the **dynamic
embedding priority** branch (see [`embedding.md`](embedding.md) for the design).
Only what is new or different on this branch is documented here; shared pieces
(metrics, the mapf-IR elapsed-PIBT baseline, `--oracle` modes) are unchanged —
see the main doc.

The embedding pipeline is RL-only (there is no per-frame imitation label for a
*dynamic* field, so no `gen_dataset`/`train_imitation` stage):

```
train_embedding_rl  →  evaluate --embedding / --ckpt <embedding ckpt>
```

`--device` defaults to `cuda` when a GPU is available, otherwise `cpu`. Always
launch training with `python -u` (unbuffered) so metrics flush live to the log.

**Input features are structural only.** `build_features` produces **4 channels**
(`N_CHANNELS=4`): free mask, clearance, row coord, col coord. The goal-heatmap
channel was removed, so the priority field never sees goal positions — agents
still reach their goals via the simulator's per-agent goal-distance fields. (This
is a breaking change: 5-channel checkpoints from before the removal won't load.)

---

## `train_embedding_rl.py`

Per-step RL for the `EmbeddingPriorityModel` (`MapEncoder` + `PriorityDecoder`)
with a state-value `Critic`. Default algorithm is **PPO + GAE**; `--algo a2c`
falls back to the single-episode A2C step. Trains on fresh random maps under
`paper` resolution, periodically benchmarks the **greedy** policy vs MST, and
saves a checkpoint that `evaluate.py`/`load_model` pick up automatically.

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--init` | str | `None` | Embedding checkpoint to warm-start from (must be `arch="embedding"`) |
| `--out` | str | `runs/rl_embedding.pt` | Last-iterate checkpoint path; best-by-eval is saved to `<out>_best.pt` |
| `--algo` | str | `ppo` | `ppo` (clipped, GAE, batched) or `a2c` (single-episode baseline) |
| `--iters` | int | `1000` | PPO updates (each collects `--batch_episodes` episodes) |
| `--batch_episodes` | int | `8` | Episodes collected per PPO update (variance reduction) |
| `--epochs` | int | `4` | PPO epochs per batch (may stop early via `--target_kl`) |
| `--clip` | float | `0.2` | PPO clip epsilon |
| `--target_kl` | float | `0.03` | Stop epochs once an epoch's mean \|approx_kl\| exceeds this (trust-region guard; `≤0` disables) |
| `--lam` | float | `0.95` | GAE lambda |
| `--gamma` | float | `0.99` | Discount |
| `--sigma` | float | `0.5` | **Initial** exploration std over per-agent priorities |
| `--sigma_final` | float | `0.1` | Exploration std at the last iter (linear anneal from `--sigma`) |
| `--value_coef` | float | `0.5` | Weight of the value loss |
| `--lr` | float | `3e-4` | Adam learning rate (encoder + decoder + critic) |
| `--size` | int | `17` | Grid side length (square) |
| `--n_agents` | int | `8` | Agents per episode |
| `--max_steps` | int | `256` | Episode step cap |
| `--dim` | int | `128` | Embedding dim (cold start; inherited from `--init`) |
| `--enc_depth` | int | `4` | MapEncoder transformer layers (cold start) |
| `--dec_depth` | int | `2` | PriorityDecoder transformer layers (cold start) |
| `--history` | int | `8` | Occupancy-history frames fed to the decoder (temporal context) |
| `--hist_dim` | int | `64` | HistoryEncoder embedding dim |
| `--hist_depth` | int | `2` | HistoryEncoder transformer layers |
| `--oracle` | str | `paper` | PIBT resolution for the rollouts (`paper` recommended — see design doc) |
| `--reward_weights` | str | `None` | YAML of **per-step** reward weights (see below); default weights if omitted |
| `--eval_every` | int | `200` | Greedy benchmark + checkpoint cadence (iters) |
| `--eval_maps_per_kind` | int | `6` | Held-out eval maps per kind (forest/wide/narrow) |
| `--eval_inst_per_map` | int | `3` | Start/goal instances per eval map |
| `--log_every` | int | `50` | Log cadence (iters) |
| `--seed` | int | `0` | RNG seed |
| `--device` | str | `cuda`/`cpu` | Compute device |

```bash
# generalization run on fresh maps (unbuffered so the log flushes live)
python -u scripts/train_embedding_rl.py --device cuda --iters 600 \
    --size 21 --n_agents 8 --batch_episodes 8 --epochs 4 --target_kl 0.03 \
    --out runs/rl_embedding.pt > runs/train_embedding.log 2>&1 &

# then benchmark the best checkpoint at a larger eval
python scripts/evaluate.py --ckpt runs/rl_embedding_best.pt --n_per_kind 12 --n_inst 5
```

> **What the "action" is.** Each step the decoder emits a priority *field*; the
> per-agent priorities (field indexed at the agent cells) are the Gaussian mean,
> and the sampled priorities are what PIBT orders on. Liveness comes from `paper`
> resolution (fixed environment dynamics), so the policy only learns contextual
> ordering. Reward is per step (progress + time) plus a terminal success/reached
> bonus.

> **`best.pt` is best-by-eval, not last.** Every `--eval_every` the greedy policy
> is scored on held-out maps; `<out>_best.pt` is overwritten when the **mean
> embedding success across kinds** improves, while `<out>` always holds the last
> iterate. Greedy success is noisy/quantized on the small in-loop eval — re-score
> the best checkpoint with `evaluate.py` at a larger `--n_per_kind` before trusting
> a number.

> **Trust-region logging.** The log prints `kl`, `clip` (clip fraction), and
> `ep=<run>/<epochs>` per update. Healthy PPO keeps `kl` small (~0.01–0.03) and
> `clip` well under ~0.5; `ep` dropping below `--epochs` means `--target_kl`
> trimmed the update. Rising `clip`/`kl` late in training → lower `--lr`, fewer
> `--epochs`, or a tighter `--target_kl`.

---

## `evaluate.py` (embedding additions)

The existing `evaluate.py` (see the main doc for the full arg table, the
elapsed-PIBT baseline, and the metric definitions) gains one flag and one
auto-routing behavior on this branch.

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--embedding` | flag | off | Evaluate the dynamic embedding model. With `--ckpt` it loads that model; without `--ckpt` it builds a **fresh (untrained)** embedding model for a baseline number |

**Auto-routing.** When `--ckpt` points at an `arch="embedding"` checkpoint,
`evaluate.py` detects it (`isinstance(model, EmbeddingPriorityModel)`) and drives
`benchmark.evaluate_embedding` — a per-episode `field_fn` recomputed each step
from live occupancy (a *dynamic* field, not one static field) instead of the
static field-provider path. All other checkpoints and the MST/elapsed-PIBT
baselines are unchanged, so the report is directly comparable across methods.

**Inference latency.** For the embedding model, eval also prints a mean
encoder/decoder latency line (CUDA-synchronized), e.g.
`latency: encoder 7.60 ms/map (n=18), decoder 6.19 ms/step (n=1041)` — the
encoder runs once per map, the decoder once per step, so this quantifies the
once-per-map / per-step cost split. The same line appears in `train_embedding_rl.py`
at every periodic eval.

```bash
# untrained embedding vs MST vs elapsed-PIBT (sanity baseline)
python scripts/evaluate.py --embedding --n_per_kind 12 --n_inst 5

# a trained embedding checkpoint (auto-routed; --embedding not required)
python scripts/evaluate.py --ckpt runs/rl_embedding_best.pt --n_per_kind 12 --n_inst 5
```

---

## `simulate_embedding.py`

Animated side-by-side episode (the `simulate.py` analogue for this branch): the
**same** start/goal instance under the static MST field and the **dynamic**
embedding field. The MST panel's background is static; the embedding panel's
background **animates** — you watch the priority field morph each step as the
model re-decides who-yields. Agents are colored dots with fading trails, goals
are stars, obstacles grey. Saves a GIF (or `--live` window).

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--ckpt` | str | `runs/rl_embedding_best.pt` | Embedding checkpoint (a non-embedding ckpt falls back to a static learned panel) |
| `--map` | str | `narrow` | `forest` / `wide` / `narrow` |
| `--size` | int | `21` | Grid side length |
| `--n_agents` | int | `8` | Number of agents |
| `--max_steps` | int | `200` | Episode step cap (also caps animation length) |
| `--seed` | int | `0` | RNG seed for map + start/goals |
| `--out` | str | `runs/sim_embedding.gif` | Output GIF path |
| `--fps` | int | `5` | Animation frames per second |
| `--trail` | int | `8` | Trail length in steps (0 = off) |
| `--raw` | flag | off | Add a top row of raw-priority maps (values labeled; the embedding row's values animate too) |
| `--live` | flag | off | Show a window instead of saving |
| `--oracle` | str | `paper` | PIBT resolution mode for the animated episodes |
| `--device` | str | `cuda`/`cpu` | Compute device |

```bash
# watch the dynamic field evolve next to the static MST field
python scripts/simulate_embedding.py --ckpt runs/rl_embedding_best.pt --map narrow --seed 3

# with the raw per-cell priority values (animated for the embedding panel)
python scripts/simulate_embedding.py --ckpt runs/rl_embedding_best.pt --map narrow --seed 3 --raw
```

The per-step fields are captured during the rollout (a recording wrapper around
`embedding_field_fn`), aligned 1:1 with the position log, so frame `t` shows the
field the sim actually used at step `t`. Backgrounds are per-map z-scored with a
fixed color range so the dynamic panel does not flicker. Both panels print final
`success / makespan / flowtime`.

## Per-step reward weights

The embedding model uses **per-step** shaping (`StepRewardWeights` in
`src/train/reward.py`), distinct from the episode-level `RewardWeights` used by
`train_rl.py`. Pass a YAML via `--reward_weights`; any omitted key keeps its
default:

```yaml
progress: 1.0       # weight on team distance-to-goal potential shaping
time_penalty: 0.01  # small negative every step (the makespan primitive)
success: 5.0        # terminal bonus for solving the whole instance
reached: 1.0        # terminal bonus x (fraction of agents at goal)
```

Per-step reward = `progress·(Φ_t − Φ_{t−1}) − time_penalty`, with `Φ =
−mean_agent_distance` (normalized); the terminal step adds `success·solved +
reached·(n_reached/n)`. Potential-based progress is policy-invariant (Ng et al.
1999) and mainly helps the failure-heavy narrow maps; see
[`embedding.md`](embedding.md) §3 for the reasoning (why *team* Σ-distance, why
time-penalty is not double-counted with makespan).

---

## Data Formats

### Embedding checkpoint (`runs/rl_embedding*.pt`)

Same loader as the other archs (`src.priority.model.load_model` →
`build_model("embedding", **config)`), with an embedding-specific `config` and an
extra `critic` key:

| key | type | meaning |
| --- | --- | --- |
| `arch` | `str` | `"embedding"` — routes `build_model`/`evaluate.py` to the dynamic path |
| `model` | `state_dict` | `EmbeddingPriorityModel` weights (MapEncoder + PriorityDecoder) |
| `critic` | `state_dict` | `Critic` weights (value baseline; unused at eval, kept for resuming) |
| `config` | `dict` | Constructor kwargs: `dim`, `enc_depth`, `dec_depth`, `heads`, `mlp_ratio`, `dropout`, `occ_dim`, `cin`, `history`, `hist_dim`, `hist_depth` |

`load_model` reads `arch` and rebuilds the matching model automatically; the
`critic` key is ignored by eval and only needed to resume training. Unlike the
static-field archs, this checkpoint is **not** consumed via `predict_field` (the
field depends on live occupancy) — eval builds a per-step `field_fn` through
`embedding_field_fn`.
