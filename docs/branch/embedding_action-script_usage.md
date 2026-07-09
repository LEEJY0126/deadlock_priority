# Scripts Usage — `feature/embedding_action`

Companion to [`embedding_action.md`](embedding_action.md) (the design) and to the
main [`docs/scripts_usage.md`](../scripts_usage.md). Only what is new or different
for the **dynamic action-map policy** is documented here; shared pieces (metrics,
the MST baseline, the elapsed-PIBT baseline) are unchanged — see the main doc.

The action pipeline is RL-only (there is no per-frame action oracle, so no
imitation stage):

```
train_embedding_action_rl  →  evaluate --action / --ckpt <embedding_action ckpt>
```

`--device` defaults to `cuda` when a GPU is available, otherwise `cpu`. Always
launch training with `python -u` (unbuffered) so metrics flush live to the log.

**Input features are structural only** — the same 4-channel `build_features`
(`N_CHANNELS=4`: free mask, clearance, row coord, col coord) as the priority model.
The policy never sees goal positions in the field; agents reach goals via the
progress reward. The **output** is the difference: a `[5, H, W]` action-logit map
over `[UP, DOWN, LEFT, RIGHT, STAY]`, read per agent at its own cell.

---

## `train_embedding_action_rl.py`

Per-step PPO for the `EmbeddingActionModel` (`MapEncoder` + `ActionDecoder`) with a
state-value `Critic`. The model emits a per-agent move directly (no PIBT); a
collision ends the episode (failed) with a collision penalty. Trains on fresh
random maps, periodically benchmarks the **greedy** policy (success + collision
rate) against MST, and saves a checkpoint that `evaluate.py`/`load_model` pick up
automatically (`arch="embedding_action"`).

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--init` | str | `None` | Checkpoint to warm-start from (must be `arch="embedding_action"`) |
| `--out` | str | `runs/rl_action.pt` | Last-iterate checkpoint path; best-by-eval saved to `<out>_best.pt` |
| `--iters` | int | `1000` | PPO updates (each collects `--batch_episodes` episodes) |
| `--batch_episodes` | int | `8` | Episodes collected per PPO update (variance reduction) |
| `--epochs` | int | `4` | PPO epochs per batch (may stop early via `--target_kl`) |
| `--clip` | float | `0.2` | PPO clip epsilon |
| `--target_kl` | float | `0.03` | Stop epochs once an epoch's mean \|approx_kl\| exceeds this (`≤0` disables) |
| `--lam` | float | `0.95` | GAE lambda |
| `--gamma` | float | `0.99` | Discount |
| `--entropy_coef` | float | `0.01` | Entropy-bonus weight (exploration; **replaces** the priority model's `--sigma`) |
| `--value_coef` | float | `0.5` | Weight of the value loss |
| `--lr` | float | `3e-4` | Adam learning rate (encoder + decoder + critic) |
| `--size` | int | `17` | Grid side length (square) |
| `--n_agents` | int | `8` | Agents per episode |
| `--max_steps` | int | `256` | Episode step cap |
| `--dim` | int | `128` | Embedding dim (cold start; inherited from `--init`) |
| `--enc_depth` | int | `4` | MapEncoder transformer layers (cold start) |
| `--dec_depth` | int | `2` | ActionDecoder transformer layers (cold start) |
| `--history` | int | `8` | Occupancy-history frames fed to the decoder |
| `--hist_dim` | int | `64` | HistoryEncoder embedding dim |
| `--hist_depth` | int | `2` | HistoryEncoder transformer layers |
| `--reward_weights` | str | `None` | YAML of per-step reward weights incl. `collision` (see below) |
| `--eval_every` | int | `200` | Greedy benchmark + checkpoint cadence (iters) |
| `--eval_maps_per_kind` | int | `6` | Held-out eval maps per kind (forest/wide/narrow) |
| `--eval_inst_per_map` | int | `3` | Start/goal instances per eval map |
| `--log_every` | int | `50` | Log cadence (iters) |
| `--seed` | int | `0` | RNG seed |
| `--device` | str | `cuda`/`cpu` | Compute device |

There is **no** `--oracle`, `--sigma`, or `--sigma_final` (no PIBT resolution, no
Gaussian action noise).

```bash
# generalization run on fresh maps (unbuffered + nohup so it survives logout)
nohup python -u scripts/train_embedding_action_rl.py --device cuda --iters 600 \
    --size 21 --n_agents 8 --batch_episodes 8 --epochs 4 --target_kl 0.03 \
    --entropy_coef 0.01 --out runs/rl_action.pt > runs/train_action.log 2>&1 &

# then benchmark the best checkpoint at a larger eval
python scripts/evaluate.py --ckpt runs/rl_action_best.pt --n_per_kind 12 --n_inst 5
```

> **What the "action" is.** Each step the decoder emits a `[5, H, W]` logit field;
> the per-agent action (field indexed at the agent cell) is a categorical over
> `[UP, DOWN, LEFT, RIGHT, STAY]` — sampled during collection, argmax at eval. A
> collision (wall / vertex / swap — see `src/envs/action_exec.py`) terminates the
> episode as a failure. Reward is per step (progress + time) plus a terminal
> success / reached bonus and a collision penalty.

> **The launch command is echoed** as the first log line (`command: python -u …`,
> from `sys.orig_argv`) so a run is reproducible from its log.

> **`best.pt` is best-by-eval, not last.** Every `--eval_every` the greedy policy is
> scored on held-out maps; `<out>_best.pt` is overwritten when the mean action
> success across kinds improves, while `<out>` always holds the last iterate.
> Greedy success is noisy on the small in-loop eval — re-score with `evaluate.py` at
> a larger `--n_per_kind` before trusting a number.

> **Log line.** Per update: `reward`, `succ`, `reached`, **`coll`** (fraction of
> the batch's episodes that ended on a collision), `pi_loss`, `v_loss`, **`ent`**
> (mean policy entropy), `kl`, `clip`, `ep=<run>/<epochs>`. A healthy run drives
> `coll`↓ and `succ`↑ while `ent` decays from ≈`ln 5 ≈ 1.61`. Rising `clip`/`kl`
> late → lower `--lr`, fewer `--epochs`, or a tighter `--target_kl`.

---

## `evaluate.py` (action additions)

The existing `evaluate.py` (main doc for the full arg table + baselines) gains one
flag and one auto-routing behavior for this branch.

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--action` | flag | off | Evaluate the dynamic action-map policy. With `--ckpt` it loads that model; without `--ckpt` it builds a **fresh (untrained)** one for a baseline. Report adds a per-kind **collision rate** |

**Auto-routing.** When `--ckpt` points at an `arch="embedding_action"` checkpoint,
`evaluate.py` detects it (`isinstance(model, EmbeddingActionModel)`) and drives
`benchmark.evaluate_action` — greedy (argmax) direct-move rollouts via
`run_action_episode`, reporting success/makespan/flowtime **plus collision rate**.
The MST/elapsed-PIBT baselines are unchanged and printed as collision-free
references (not apples-to-apples, since they run through PIBT).

**Inference latency.** As for the priority model, eval prints a mean
encoder/decoder latency line (CUDA-synchronized), e.g.
`latency: encoder 7.60 ms/map (n=18), decoder 6.19 ms/step (n=1041)` — the encoder
runs once per map, the decoder once per step. The same line appears in
`train_embedding_action_rl.py` at every periodic eval.

```bash
# untrained action policy vs MST (collision rate reported alongside success)
python scripts/evaluate.py --action --n_per_kind 12 --n_inst 5

# a trained action checkpoint (auto-routed; --action not required)
python scripts/evaluate.py --ckpt runs/rl_action_best.pt --n_per_kind 12 --n_inst 5
```

---

## Per-step reward weights

The action policy uses the same **per-step** `StepRewardWeights` as the priority
model, with the extra terminal `collision` term. Pass a YAML via
`--reward_weights`; any omitted key keeps its default:

```yaml
progress: 1.0       # weight on team distance-to-goal potential shaping
time_penalty: 0.01  # small negative every step (the makespan primitive)
success: 5.0        # terminal bonus for solving the whole instance
reached: 1.0        # terminal bonus x (fraction of agents at goal)
collision: -5.0     # terminal penalty when the episode ends on a collision
```

Per-step reward = `progress·(Φ_t − Φ_{t−1}) − time_penalty`, with `Φ =
−mean_agent_distance` (normalized); the terminal step adds `success·solved +
reached·(n_reached/n)`, and `collision` when the episode ended on a collision (in
which case `solved = 0`, so no success bonus). Potential-based progress is
policy-invariant (Ng et al. 1999).

---

## Data Formats

### Action checkpoint (`runs/rl_action*.pt`)

Same loader as the other archs (`src.priority.model.load_model` →
`build_model("embedding_action", **config)`), with an extra `critic` key:

| key | type | meaning |
| --- | --- | --- |
| `arch` | `str` | `"embedding_action"` — routes `build_model`/`evaluate.py` to the direct-action path |
| `model` | `state_dict` | `EmbeddingActionModel` weights (MapEncoder + ActionDecoder) |
| `critic` | `state_dict` | `Critic` weights (value baseline; unused at eval, kept for resuming) |
| `config` | `dict` | Constructor kwargs: `dim`, `enc_depth`, `dec_depth`, `heads`, `mlp_ratio`, `dropout`, `occ_dim`, `cin`, `history`, `hist_dim`, `hist_depth`, **`n_actions`** (5) |

`load_model` reads `arch` and rebuilds the matching model automatically; the
`critic` key is ignored by eval and only needed to resume training. Eval drives the
model through `benchmark.evaluate_action` (greedy direct-move rollouts, no
field/PIBT) rather than `predict_field`.
