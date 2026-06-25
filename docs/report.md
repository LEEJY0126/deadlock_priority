# Project vs. Paper: Scope and Differences

This project is **not** a reimplementation of the paper *"Asynchronous
Communication-free Multi-Agent Trajectory Planning and Deadlock Resolution in
Maze-like Environments"* (IEEE T-RO submission 26-0057). It isolates and tests
**one idea** from that paper — position-based priority for deadlock resolution —
and asks whether a **learned** priority field can beat the paper's hand-designed
heuristic. This document records exactly what we kept, what we changed, and why,
so the scope is not misread.

## One-line summary

> The paper is a full asynchronous, communication-free, **continuous-space**
> trajectory planner with collision-avoidance guarantees. This project keeps only
> its *position-priority* concept and replaces the hand-designed priority with a
> **learned CNN field**, evaluated at the discrete **grid + PIBT** abstraction.

## The hypothesis tested here

Given the paper's priority-based deadlock-resolution scheme, *can a learned
priority field outperform the paper's MST priority tree?* We test this where
priority actually has its effect — the grid/MAPF (PIBT) layer — rather than in
the full continuous stack.

## What is kept from the paper

- **Position-based priority field.** A priority that is a deterministic function
  of `(cell, known map)`, shared by all agents. Because it is computed offline
  and identical for everyone, it needs no communication. (Paper Sec. IV-C.1.)
- **MST priority tree (Eq. 12).** Implemented faithfully in
  `src/priority/mst_baseline.py` as the **baseline to beat**.
- **PIBT** as the MAPF solver that consumes priorities, plus the paper's
  "agents at goal yield" and anti-oscillation tie-break ideas (Eq. 13).
- **Deadlock / livelock resolution (Alg. 3).** Both branches are ported to the
  grid in the default `"paper"` yield mode: the **right-hand rule** for deadlock
  (Eq. 14 detection → Eq. 15 sidestep) and the **lowest-priority-neighbour
  back-out** for livelock (Eq. 18). The limited-sensing conflict (Eq. 19) has
  nothing to fire on under the simulator's full observability, so it is omitted.

## What is different

| Aspect | Paper | This project |
|--------|-------|--------------|
| **Priority source** | Hand-designed MST priority tree | **Learned field** (CNN U-Net or Transformer; hybrid imitation → RL) |
| **State space** | Continuous 3D, double-integrator dynamics | **Discrete grid**, agents hop cell-to-cell |
| **Inter-agent collision avoidance** | ABVC + Safe Flight Corridor + QP trajectory optimization (with Theorems 1–2) | **Not modeled** — only PIBT's grid vertex/swap conflict rules |
| **Asynchronous updates** | Core contribution; ABVC guarantees safety under async replanning | **Not modeled** — synchronous, centralized PIBT step |
| **Communication-free** | Achieved via position priority + ABVC | Priority is comms-free by construction (map-only input); the rest of the async/sensing machinery is absent |
| **Limited sensing range** | Explicit constraint (Eq. 11) | **Not modeled** — full observability in the simulator |
| **Deadlock / livelock handling** | Right-hand rule + livelock detection + limited-sensing conflict resolution (Alg. 3) | Right-hand rule (Eq. 14→15) + livelock back-out (Eq. 18) **ported to the grid** in `"paper"` mode; limited-sensing (Eq. 19) N/A under full observability. Legacy `"beta"` anti-starvation boost also available |
| **Validation** | Simulation + 8-quadrotor hardware | Grid simulation only |

## Design choices that depart from the paper (and why)

These are abstraction decisions made so that *priority quality* is the only
variable under study:

1. **Two yield modes (`paper` vs `beta`).** A purely static field cannot resolve
   a 1-wide corridor (agents must move aside — PIBT's reachability needs a
   tie-breaking term). The default `"paper"` mode ports Alg. 3 faithfully
   (right-hand rule for deadlock, lowest-priority back-out for livelock). A legacy
   `"beta"` mode instead adds an anti-starvation boost ∝ stuck-time that *raises*
   a stuck agent's priority so it pushes through; it is more forgiving and is used
   for **training**, while evaluation/visualization use `"paper"`. Either mode is
   applied **identically** to the MST baseline and the learned field, so the field
   remains the only difference. Fields are scale-normalized first so the mechanism
   treats the integer MST field and the softplus learned field equally.
2. **Braided mazes.** A perfect (tree) maze is near-unsolvable for 8 agents (no
   passing places), which would mask any priority differences. We braid mazes to
   add alcoves/loops → hard-but-solvable.
3. **Map-only model input.** The learned field uses only globally-known
   information (occupancy, clearance, coordinates), never live agent positions.
   This preserves the paper's communication-free property by construction and
   avoids the inconsistency a per-agent runtime policy could cause.
4. **Imitation label via candidate-bank search.** The optimal field is
   intractable, so the oracle scores a bank of candidate fields by simulation and
   keeps the best as a cheap proxy label.

## Why the comparison is still fair and meaningful

- Both methods run through the **same** simulator, PIBT solver, deadlock/livelock
  resolution (the same `yield_mode`), maze distribution, and held-out instances.
  Only the priority field differs.
- Priority does not affect collision safety in the paper either (safety comes
  from ABVC/SFC/stop-constraint, Theorems 1–2). So studying priority in isolation
  does not discard a safety property — a worse field can only cause deadlock or
  inefficiency, never a collision.

## Result (held-out, 8 agents, 21×21, 500 instances/kind, `paper` mode)

Default `paper` resolution (right-hand rule + livelock), with two correctness
fixes to the resolution layer (see note). Headline: the learned **Transformer**
field trained **fully in-distribution** (paper-mode oracle labels → paper-mode RL
→ paper-mode eval; `runs/rl_transformer.pt`, `dim=96`, best-by-success) vs the MST
baseline — `succ` (↑) / `mksp` (↓) / `flow` (↓):

| map | MST — succ / mksp / flow | Learned (Transformer, imit→RL) — succ / mksp / flow |
|-----|:------------------------:|:---------------------------------------------------:|
| forest | 93.2% / 27.1 / 128.1 | **94.8%** / 26.2 / 126.4 |
| wide   | 91.0% / 27.5 / 129.2 | **94.2%** / 26.0 / 125.7 |
| narrow | 60.6% / 33.7 / 154.6 | **61.0%** / 30.7 / 146.4 |

The learned field wins on every metric on every map, but by a **modest margin**
(+0.4–3.2pp success; lower makespan/flowtime everywhere). Honest takeaway: once
deadlocks are resolved correctly, the paper's hand-designed **MST baseline is
already strong** (61–93%) and the learned field is a small *consistent*
improvement — on `narrow` the success gap is nearly closed (+0.4pp), though the
learned field still finishes faster (flowtime 146.4 vs 154.6).

> **Corrected baseline (two fixed bugs).** (1) Agents at their goal now get
> `-inf` priority so they always yield (Eq. 13a); priced at `0`, with the per-map
> z-scored field an en-route agent in a low-priority region could go *negative*
> and fail to displace a *finished* agent parked on its path. (2) The livelock
> detector now catches priority-**oscillation** swaps directly and yields by
> *base* priority (the anti-oscillation tie-break otherwise hid who should yield).
> Both depressed every success rate and inflated the apparent learned-vs-MST gap;
> fixing them raises both (MST narrow 42.8%→60.6%) so the true gap is small. A
> residual class — a goal parked in a 1-wide corridor on another agent's only path
> — is unsolvable by any *local* yield under the MST field but is handled by the
> learned field (it is not a deviation from the paper, whose Eq. 14/18 conditions
> also do not fire there; the paper guarantees collision- not deadlock-freeness).

`runs/fields_rl_transformer.png` shows *where* the learned field still helps: the
MST field is piecewise-constant in coarse blocks, while the learned field is a
smooth fine-grained gradient that breaks symmetry more precisely at junctions.

## Ablation: removing the pooling layer

To test whether the U-Net's multiscale receptive field actually matters, we
retrained an identical model with `pool=False` (`--no_pool`): a full-resolution
flat CNN, no `MaxPool2d`. Same dataset, same training schedule; only the
architecture differs.

> **Note.** The ablation table below is the earlier measurement (legacy `beta`
> mode, 60 instances/kind); it has not been rerun under the current `paper`-mode
> right-hand-rule resolution. The qualitative conclusion (pooling matters, removing
> it hurts narrow-corridor reasoning and destabilizes RL) is what to take from it.

Success rate (held-out, 8 agents, 21×21, 60 instances/kind, legacy `beta` mode):

| map | MST baseline | pooled imitation | no-pool imitation | pooled RL | no-pool RL |
|-----|:------------:|:----------------:|:-----------------:|:---------:|:----------:|
| forest | 90.0% | 88.3% | 90.0% | **91.7%** | 78.3% |
| wide   | 81.7% | 86.7% | 85.0% | **90.0%** | 80.0% |
| narrow | 30.0% | 45.0% | 35.0% | **46.7%** | 31.7% |

Findings:

1. **Apples-to-apples (imitation, only arch differs):** removing pooling costs
   **−10pp on narrow mazes** (45.0 → 35.0), −1.7pp on wide, and is noise on
   forest. The damage concentrates exactly where corridor-scale reasoning matters
   — open forest, where local features suffice, is unaffected.
2. **Under RL the no-pool model is unstable:** it degraded *below its own
   imitation init* (forest 90 → 78, narrow 35 → 32) back toward the MST baseline,
   whereas the pooled model fine-tunes cleanly to the best numbers overall.
3. **Conclusion:** the pooling-based multiscale receptive field is doing real
   work; removing it lowers the ceiling and destabilizes RL for no compute saving
   (full-res deep layers are *more* expensive). A principled full-resolution
   alternative would be dilated convolutions, not deleting the downsampling.

The `--no_pool` checkpoints are not shipped in `runs/` (this ablation predates the
current `paper`-mode regeneration). Reproduce with `--no_pool` on
`train_imitation.py` — the arch then propagates through RL/eval/viz automatically
via the checkpoint flag.

## Engineering notes

### Rollout parallelism (RL)

RL wall-clock is dominated by the `batch_maps × K` PIBT episode rollouts per step,
not the network: the CNN runs batched on GPU, but each episode is small, branchy,
CPU-bound NumPy. The rollouts are independent, so `train_rl.py --workers N`
dispatches them over a `spawn` process pool. Measured: **~3.2× wall-clock at
`--workers 8`** for the default batch (≈30 min → ≈9 min); ~8 workers saturates
because there are only `batch_maps × K = 32` tasks. Correctness is worker-count
independent — all randomness (field sampling, start/goals) happens in the parent
before dispatch; the pool only relocates the deterministic reward computation.

**Why not GPU rollouts?** PIBT is the wrong workload for a GPU: it is sequential
within a timestep (a lower-priority agent's move depends on already-placed
higher-priority agents) and uses data-dependent recursive backtracking (warp
divergence under SIMT), with tiny per-step arithmetic. GPU-accelerating a *single*
episode would be slower than CPU. The GPU-appropriate alternative is **batched
vectorized envs** (thousands of episodes as tensor lanes, Isaac-Gym/Brax style),
but (a) PIBT's backtracking does not vectorize cleanly without either heavy
divergence or replacing the solver with a non-backtracking conflict rule (which
drops PIBT's reachability guarantee), and (b) at this scale (32 envs/step) the
lane count is far below where GPU env-sims amortize launch overhead. If env count
is later scaled to thousands, a vectorized GPU stepper becomes worth revisiting;
until then, CPU multiprocessing is the right tool. *(Future work.)*

## What this project does NOT claim

- It does **not** reproduce the paper's continuous-space planner, async safety,
  limited-sensing handling, or collision-avoidance guarantees.
- It does **not** show the learned field improves the *full* system. The intended
  bridge — dropping the learned field into a continuous planner with the ABVC
  collision layer and confirming the gain transfers — is **future work** (see the
  README "Limitations / next steps").
