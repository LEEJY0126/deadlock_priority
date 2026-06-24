# Scripts Usage

All scripts live in `scripts/` and are run from the project root, e.g.
`python scripts/gen_dataset.py --n_maps 150`. The typical end-to-end order is:
`smoke_test` → `gen_dataset` → `train_imitation` → `train_rl` → `evaluate` /
`visualize`.

`--device` defaults to `cuda` when a GPU is available, otherwise `cpu`.

---

## `smoke_test.py`
End-to-end sanity check: builds forest / wide-maze / narrow-maze maps and runs
the MST baseline through PIBT, reporting success/makespan/flowtime.

*No arguments.*

```bash
python scripts/smoke_test.py
```

---

## `gen_dataset.py`
Generate the imitation dataset: per map, the oracle searches a candidate bank for
the best map-level priority field and caches `(occupancy, label field)`.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--out` | str | `data/imitation.npz` | Output `.npz` path |
| `--n_maps` | int | `120` | Number of maps to generate (cycles forest/wide/narrow) |
| `--size` | int | `21` | Grid side length (square map) |
| `--n_agents` | int | `8` | Agents per evaluation instance |
| `--n_samples` | int | `4` | Start/goal instances used to score each candidate field |
| `--seed` | int | `0` | RNG seed |

```bash
python scripts/gen_dataset.py --out data/imitation.npz --n_maps 150
```

---

## `train_imitation.py`
Imitation pretraining: regress the CNN onto oracle-selected priority fields
(loss is ordering-only, standardized per map).

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data` | str | `data/imitation.npz` | Dataset produced by `gen_dataset.py` |
| `--out` | str | `runs/imitation.pt` | Checkpoint output path (best val) |
| `--epochs` | int | `200` | Training epochs |
| `--bs` | int | `16` | Batch size |
| `--lr` | float | `1e-3` | Adam learning rate |
| `--device` | str | `cuda`/`cpu` | Compute device |

```bash
python scripts/train_imitation.py --data data/imitation.npz --out runs/imitation.pt
```

---

## `train_rl.py`
RL fine-tuning (GRPO-style group-baseline REINFORCE over whole-field actions).
Optionally warm-starts from an imitation checkpoint and anchors to it to prevent
catastrophic forgetting. Periodically benchmarks against the MST baseline.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--init` | str | `None` | Imitation checkpoint to warm-start from (also used as the anchor) |
| `--out` | str | `runs/rl.pt` | Checkpoint output path |
| `--iters` | int | `300` | Optimization iterations |
| `--batch_maps` | int | `4` | Maps per iteration (balanced across kinds) |
| `--K` | int | `8` | Field samples drawn per map (group size) |
| `--sigma` | float | `0.5` | Std of the Gaussian field perturbation |
| `--lr` | float | `1e-4` | Adam learning rate |
| `--anchor_w` | float | `0.5` | Weight of the L2 anchor toward the init logits (0 disables) |
| `--size` | int | `21` | Grid side length |
| `--n_agents` | int | `8` | Agents per episode |
| `--eval_every` | int | `25` | Benchmark + checkpoint cadence (iterations) |
| `--workers` | int | `0` | Parallel rollout workers (0/1 = serial; ~3× faster at 8; cpu engine only) |
| `--engine` | str | `cpu` | Rollout engine: `cpu` (exact PIBT) or `vec` (GPU-batched approx; `GPU_vectorized` branch) |
| `--reward_weights` | str | `reward_weight.yaml` | YAML of reward shaping weights (snapshotted into the run dir) |
| `--device` | str | `cuda`/`cpu` | Compute device |

> **Rollout parallelism.** The `batch_maps × K` episode rollouts each step are
> independent and CPU-bound (the NN runs on GPU, the PIBT sim on CPU). `--workers
> N` spreads them over a process pool — ~3.2× wall-clock at `--workers 8` for the
> default batch (≈30 min → ≈9 min). Going beyond ~8 gives little extra since there
> are only `batch_maps × K` tasks (default 32). Results are independent of worker
> count: all randomness happens before dispatch.

```bash
python scripts/train_rl.py --init runs/imitation.pt --out runs/rl.pt --iters 150
```

---

## `evaluate.py`
Benchmark the MST baseline against a learned checkpoint on identical held-out
instances (per-kind success rate, makespan, flowtime).

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--ckpt` | str | `None` | Learned checkpoint to compare (omit to report baseline only) |
| `--n_per_kind` | int | `10` | Eval maps per kind (use ≥12 — small evals are noisy) |
| `--n_inst` | int | `4` | Start/goal instances per map |
| `--n_agents` | int | `8` | Agents per instance |
| `--device` | str | `cuda`/`cpu` | Compute device |

```bash
python scripts/evaluate.py --ckpt runs/rl.pt --n_per_kind 12 --n_inst 5
```

### Metrics: success / makespan / flowtime

Reported by both `evaluate.py` and `bench_vec.py` (defined in
`EpisodeResult`, `src/envs/simulator.py`). An episode runs until all agents reach
their goals or a `max_steps` cap is hit.

| metric | meaning | units / range | direction |
|--------|---------|---------------|-----------|
| **success** | All agents are simultaneously at their goals before `max_steps`. Reported as the **rate** (% of episodes solved). | 0–100% | higher better |
| **makespan** | The step at which the **last** agent arrives (all-at-goal time). Failed episodes count as `max_steps`. | steps | lower better |
| **flowtime** | Sum over agents of each agent's **first-arrival** step (sum-of-costs). Agents that never arrive contribute `max_steps`. | agent·steps | lower better |

Notes:
- **success** is the headline number — it measures deadlock resolution. makespan
  and flowtime are *efficiency* measures, only meaningful among solved episodes.
- **makespan vs flowtime:** makespan is the slowest single agent (team finish
  time); flowtime is the total/average effort across all agents. A field can
  improve one while worsening the other (e.g. making one agent wait to unblock
  the rest lowers makespan but can raise flowtime).
- In `evaluate.py` / training logs, makespan and flowtime are **averaged over the
  evaluated instances** of each map kind.

---

## `visualize.py`
For a forest / wide / narrow map, render three columns — the original obstacle
map, the MST priority field, and the learned priority field — and save a PNG.
Each priority cell is annotated with its raw priority value; cell color is
per-map normalized (z-score) so the MST and learned patterns are comparable
despite different scales.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--ckpt` | str | `runs/rl.pt` | Learned checkpoint (map + MST only if missing) |
| `--out` | str | `runs/fields.png` | Output PNG path |
| `--size` | int | `21` | Grid side length (smaller → larger, more legible numbers) |
| `--no_numbers` | flag | off | Skip the per-cell priority labels |
| `--device` | str | `cuda`/`cpu` | Compute device |

```bash
python scripts/visualize.py --ckpt runs/rl.pt --out runs/fields.png
python scripts/visualize.py --ckpt runs/rl.pt --size 13   # bigger, readable numbers
```

---

## `simulate.py`

Animated side-by-side episode: runs the **same** start/goal instance under the
MST field and the learned field and animates the agents, so you can watch *how*
the learned priority changes who-yields and the resulting trajectories. Priority
field as background (viridis), obstacles grey, agents as colored dots with fading
trails, goals as matching-color stars. Saves a GIF (or shows a live window).

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--ckpt` | str | `runs/rl.pt` | Learned checkpoint (MST-only panel if missing) |
| `--map` | str | `narrow` | `forest` / `wide` / `narrow` |
| `--size` | int | `21` | Grid side length |
| `--n_agents` | int | `8` | Number of agents |
| `--max_steps` | int | `200` | Episode step cap (also caps animation length) |
| `--seed` | int | `0` | RNG seed for map + start/goals |
| `--out` | str | `runs/sim.gif` | Output GIF path |
| `--fps` | int | `5` | Animation frames per second |
| `--trail` | int | `8` | Trail length in steps (0 = off) |
| `--live` | flag | off | Show a window instead of saving |
| `--device` | str | `cuda`/`cpu` | Compute device |

```bash
# narrow-maze case where MST deadlocks but the learned field solves it:
python scripts/simulate.py --ckpt runs/rl.pt --map narrow --seed 10 --max_steps 60
```

Both panels print final `success / makespan / flowtime`. Tip: use `evaluate.py`
or a quick scan to find a seed that contrasts the two methods.

---

## `bench_vec.py` (GPU_vectorized branch only)

Throughput benchmark for the experimental GPU-vectorized simulator
(`src/envs/vec_sim.py`) vs the CPU `Simulator`. Builds E identical episodes,
runs them both ways, and reports episodes/second plus an open-map sanity check.
See `report_GPU-vectorized.md` and the tracked `GPU_VECTORIZED.md` for findings.
**Only exists on the `GPU_vectorized` branch.**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--device` | str | `cuda`/`cpu` | Compute device |
| `--size` | int | `21` | Grid side length |
| `--n_agents` | int | `8` | Agents per episode |
| `--max_steps` | int | `256` | Episode step cap |
| `--counts` | str | `32,128,512,2048` | Comma-separated batch sizes (E) to sweep |
| `--cpu_max` | int | `512` | Skip the CPU run above this E (its eps/s is ~constant) |

```bash
python scripts/bench_vec.py --device cuda --max_steps 128 \
    --counts 32,128,512,2048 --cpu_max 512
```

Columns: `CPU eps/s` (end-to-end CPU sim), `GPU eps/s` (`VecSim.run` stepping
only), `GPU build/s` (`build_batch` setup only — per-episode BFS, CPU-bound),
`step speedup` (CPU/GPU run time).

---

# Experiment Tracking

Both `train_imitation.py` and `train_rl.py` create a per-run directory
`logs/{script_name}_{timestamp}/` (gitignored) via `src/utils/experiment.py`.
Each run dir contains:

| file | contents |
|------|----------|
| `config.yaml` | run hyperparameters (see below) |
| `train.log` | human-readable progress (also echoed to stdout) |
| `events.out.tfevents.*` | TensorBoard scalars |
| `model.py`, `features.py` | snapshots of the model architecture + input feature definitions used for the run |
| `*.pt` | checkpoints — `best.pt` (imitation), `checkpoint.pt`/`final.pt` (RL). Also copied to `--out`. |
| `reward_weight.yaml` | (RL only) snapshot of the reward weights used |

**`config.yaml` fields**
- `train_imitation`: `total_epochs`, `batch_size`, `learning_rate` (+ `data`, `no_pool`, `device`, `best_val_loss`).
- `train_rl`: `total_iters`, `sigma`, `learning_rate`, `n_agents` (+ `batch_maps`, `K`, `anchor_w`, `engine`, `init`, `reward_weights`).

**TensorBoard scalars**
- `train_imitation`: `loss/train`, `loss/val`.
- `train_rl`: `reward/step`, `reward/ema`, `loss`, and per map kind
  `eval_success/{kind}`, `eval_makespan/{kind}`, `eval_flowtime/{kind}`.

View with: `tensorboard --logdir logs`

## Reward weights (`reward_weight.yaml`)

RL reward shaping is configured in a tracked `reward_weight.yaml` at the repo
root (override with `train_rl.py --reward_weights path.yaml`):

```yaml
success: 2.0     # weight on solving (all agents reach goals)
makespan: 0.5    # penalty on team finish time (normalized by max_steps)
flowtime: 0.5    # penalty on total effort (normalized by n_agents*max_steps)
```

Per-episode reward = `success*w_success − makespan_norm*w_makespan −
flowtime_norm*w_flowtime` (see `src/train/reward.py`). The resolved weights are
snapshotted into each RL run dir for reproducibility. Raise `success` to push
harder on deadlock resolution; raise `makespan`/`flowtime` to favor speed.

---

# Data Formats

## `data/imitation.npz` (dataset)

A compressed NumPy archive (`np.savez_compressed`) written by `gen_dataset.py`,
holding three **row-aligned** arrays — index `i` refers to the same map in all
three (stacked in generation order, cycling forest → wide → narrow).

| key | shape | dtype | meaning |
|-----|-------|-------|---------|
| `occ` | `(N, S, S)` | `uint8` | Occupancy grids. `0` = free cell, `1` = obstacle. |
| `label` | `(N, S, S)` | `float32` | Oracle-selected priority field per cell — the imitation target. |
| `kind` | `(N,)` | `<U6` (str) | Map type per row: `forest` / `wide` / `narrow`. |

`N` = number of maps (`--n_maps`), `S` = grid size (`--size`, e.g. 21).

`label` values are **priority levels** (higher = higher priority): integer-like,
small per map (e.g. 0–3), but up to ~47 across the dataset because deep narrow
mazes grow taller MST priority trees. Obstacle cells are `0`.

**Not stored:** the model input features. `train_imitation.py:load()` recomputes
them at load time via `build_features(GridMap(occ))` and derives the free mask as
`occ == 0`. This keeps the archive tiny (~27 KB) and lets you change the feature
set (`src/priority/features.py`) and retrain **without** regenerating the
dataset. Rerun `gen_dataset.py` only if you change the maps, the oracle, or the
grid size.

## `runs/*.pt` (checkpoints)

A `torch.save` dict written by `train_imitation.py` / `train_rl.py`:

| key | type | meaning |
|-----|------|---------|
| `model` | `state_dict` | `PriorityUNet` weights. |
| `no_pool` | `bool` | Architecture flag — `True` if the model was trained with `--no_pool` (full-res, no MaxPool). |

Load with `src.priority.model.load_model(path, device)`, which reads `no_pool`
and rebuilds the matching architecture automatically. Legacy checkpoints without
the `no_pool` key default to the pooled architecture.
