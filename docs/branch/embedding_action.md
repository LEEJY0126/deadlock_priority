# Dynamic action-map policy (`feature/embedding_action`)

A follow-on to the [embedding priority](embedding.md) model. Instead of emitting a
scalar **priority field** that PIBT turns into collision-free moves, the model
emits a per-cell **action distribution** and moves agents **directly** — PIBT is
gone, so the policy itself is responsible for avoiding collisions.

Scripts/flags/checkpoint reference: [`embedding_action-script_usage.md`](embedding_action-script_usage.md).

## 1. Motivation

The priority path (`embedding.md`) keeps liveness and collision-avoidance in the
*environment*: the model only learns an ordering, and `paper`-mode PIBT guarantees
a legal joint move every step. That is safe but caps how much the learned component
can change — resolution does most of the work.

This variant hands the whole decision to the model: each step it picks, for every
agent, one of `[UP, DOWN, LEFT, RIGHT, STAY]`. There is no PIBT and no resolution
oracle, so nothing prevents two agents from colliding. A collision **ends the
episode immediately** as a failure with a penalty, so the policy has to *learn*
collision-free coordination — the PRIMAL-style setting the decentralized-policy
notes point at, but implemented here on the shared-embedding trunk.

## 2. Architecture

The trunk is **shared and unchanged** from the priority model:

| Stage | Module | Runs | Output |
| --- | --- | --- | --- |
| Map encode | `MapEncoder` | once per map | shared embedding `[D,H,W]` |
| History encode | `HistoryEncoder` | per step | occupancy-history embedding |
| Action decode | **`ActionDecoder`** | per step | **action logits `[5,H,W]`** |

Only the decoder head differs: where `PriorityDecoder` ends in `Linear(dim, 1)` +
softplus (one positive scalar per cell), `ActionDecoder` ends in `Linear(dim, 5)`
— five logits per cell, a categorical over `[UP, DOWN, LEFT, RIGHT, STAY]`. The
fusion body (occupancy conv + projected history + shallow transformer) is identical.

**Field-then-index (unchanged contract).** Each agent reads the 5-vector at *its
own* cell (`agent_action_logits`, the analogue of `agent_priorities`) and forms a
per-agent categorical. As with the priority model this keeps the policy:

- **agent-count / ordering agnostic** — one shared field, read off per agent;
- **communication-free** — the field is a function of the map + shared occupancy
  (+ history), never of agent identity.

Action channel order is fixed to the request `[UP, DOWN, LEFT, RIGHT, STAY]`; the
matching grid deltas are `ACTION_MOVES` in `src/envs/action_exec.py`
(`(-1,0),(1,0),(0,-1),(0,1),(0,0)`). Note this differs from grid's `MOVES`
(`STAY,UP,DOWN,LEFT,RIGHT`), so the action module defines its own order to avoid
confusion.

Files: `src/priority/model_action.py` — `EmbeddingActionModel`, `ActionDecoder`,
`agent_action_logits`, `action_field_fn`, `greedy_action_fn`. Registered as
`arch="embedding_action"` in `src/priority/model.py::build_model`, so `load_model`
and `evaluate.py` route it automatically.

## 3. Execution + collision model

`src/envs/action_exec.py::run_action_episode` applies the joint move each step. A
step is a **collision** if any of:

- **wall** — an agent's next cell is an obstacle or out of bounds;
- **vertex** — two agents' next cells coincide;
- **swap** — two adjacent agents exchange cells (`i→pos[j]` while `j→pos[i]`).

Plain **following** — moving into a cell another agent vacates the *same* step — is
**allowed** (no vertex conflict at the next configuration, no swap). On a collision
the episode stops immediately and is marked failed (the penalty is applied by the
reward, not the executor). `find_collision` returns the reason (`"wall"`,
`"vertex"`, `"swap"`, or `None`).

Design choices (locked with the user):

- **Arrived agents stay in play and remain collidable.** They keep acting (can even
  move off their goal); their cell still blocks others and still participates in
  swap/vertex checks. Success is unchanged: **all agents on goals simultaneously**.
- **No action masking.** Illegal moves (into walls/agents) are *not* filtered out —
  learning to avoid them is the entire point. This is why a random policy collides
  almost immediately.

`run_action_episode` returns an `ActionEpisodeResult` (`success, makespan,
flowtime, n_reached, steps, collided, collision_step, collision_reason,
positions_log`). Its `positions_log` has `steps + 1` entries; on a collision the
final entry repeats the pre-collision config (no move applied), so per-step reward
assembly stays 1:1 with the recorded action calls. One runner backs both eval
(greedy `action_fn`) and RL (a recording, sampling `action_fn`).

## 4. Reward

Reuses the per-step `StepRewardWeights` (`src/train/reward.py`) and adds one
terminal term:

| term | when | meaning |
| --- | --- | --- |
| `progress` (1.0) | per step | potential shaping on team distance-to-goal `Φ = −mean_dist` |
| `time_penalty` (0.01) | per step | small negative each step (makespan primitive) |
| `success` (5.0) | terminal | solved the whole instance |
| `reached` (1.0) | terminal | × fraction of agents at goal (partial credit) |
| **`collision` (−5.0)** | terminal | episode ended on a collision |

`collision` mirrors `success` in magnitude, so a crash roughly cancels a solve —
enough to push hard toward collision-free without paralysing the policy into an
all-STAY optimum. A collided episode gets `success=0`, so it cannot also collect
the success bonus. The `collision` key is ignored by the PIBT priority path (which
is collision-free by construction).

## 5. Training — per-step PPO

`src/train/rl_action.py` mirrors the PPO+GAE path in `rl_embedding.py` and
**reuses** its `Critic`/`make_critic` and `occ_history_window`:

- grad-free **collection** records, per step, occupancy, agent cells, the sampled
  action indices, and the behavior log-prob / value;
- the **update** re-encodes each map and replays the states over several epochs
  with a clipped surrogate + value loss, KL early-stop (`--target_kl`) as the
  trust-region guard.

Differences from the priority path:

- **Categorical, not Gaussian.** The policy is a `Categorical` over the 5 actions;
  the joint log-prob is `Σ log p(aᵢ)`. There is no `sigma`.
- **Exploration = sampling + entropy bonus** (`--entropy_coef`, default 0.01). The
  loss subtracts `entropy_coef · mean entropy`.
- **GAE bootstrap** is 0 when the episode **solved *or* collided** (both are true
  terminals), else `V(final occupancy)` for `max_steps` truncation.
- **No `--oracle`** (there is no PIBT resolution).

The driver `scripts/train_embedding_action_rl.py` logs, per update: `reward`,
`succ`, `reached`, **`coll`** (collision rate), `pi_loss`, `v_loss`, **`ent`**
(policy entropy), `kl`, `clip`, `ep`. A healthy run drives `coll`↓ and `succ`↑
while `ent` decays from ≈`ln 5 ≈ 1.61` as the policy sharpens.

## 6. Evaluation

`scripts/evaluate.py` auto-routes an `arch="embedding_action"` checkpoint to
`benchmark.evaluate_action`: **greedy (argmax)** direct-move rollouts through
`run_action_episode`, reporting the usual success/makespan/flowtime **plus a
per-kind collision rate** (fraction of episodes that ended on a collision) and the
CUDA-synced encoder/decoder latency line. The MST baseline is still printed as a
*collision-free reference* (it runs through PIBT), not an apples-to-apples number.

```bash
python3 scripts/evaluate.py --action --n_per_kind 6 --n_inst 4        # untrained baseline
python3 scripts/evaluate.py --ckpt runs/rl_action_best.pt             # trained (auto-routed)
```

## 7. Limitations / open work

- **Harder RL problem.** Terminate-on-first-collision makes early episodes very
  short (a random policy collides almost immediately), so credit is sparse. This is
  fundamentally harder than the priority path, where PIBT guaranteed liveness.
  Curriculum knobs: `--n_agents`, `--size`, `--entropy_coef`.
- **Convergence run outstanding.** The pipeline (model, executor, RL, eval, tests)
  is wired and smoke-tested end-to-end; a real training run on fresh maps with
  held-out eval is the open item — same status as the priority path.
- **Global-observation only.** Like the priority variant, comms-free holds only if
  all agents share the same occupancy; a limited FOV would break it.
- **No imitation warm-start.** RL starts cold; there is no per-frame action oracle.

## 8. Reproduce

```bash
# tests (model shapes, collision detection, RL mechanics)
python3 -m pytest tests/test_model_action.py tests/test_action_exec.py tests/test_rl_action.py -q

# untrained action policy vs MST (collision rate reported alongside success)
python3 scripts/evaluate.py --action --n_per_kind 6 --n_inst 4

# train (PPO+GAE); saves runs/rl_action.pt + runs/rl_action_best.pt, auto-routed by
# evaluate.py. Use `python3 -u` so the log flushes live; nohup so it survives logout.
nohup python3 -u scripts/train_embedding_action_rl.py --device cuda --iters 600 \
    --size 21 --n_agents 8 --entropy_coef 0.01 \
    --out runs/rl_action.pt > runs/train_action.log 2>&1 &
python3 scripts/evaluate.py --ckpt runs/rl_action_best.pt --n_per_kind 12 --n_inst 5
```
