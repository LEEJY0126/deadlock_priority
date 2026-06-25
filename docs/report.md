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

Success rate, default `paper` resolution (right-hand rule + livelock):

| map | MST baseline (paper) | imitation CNN | hybrid CNN | imitation Transformer | hybrid Transformer |
|-----|:--------------------:|:-------------:|:----------:|:---------------------:|:------------------:|
| forest | 70.0% | 78.2% | **79.6%** | 77.4% | 76.4% |
| wide   | 70.2% | 71.4% | 73.0% | **74.4%** | 72.4% |
| narrow | 42.8% | 46.4% | 45.4% | 46.4% | **48.8%** |

All four learned fields beat the paper's heuristic on all three map types
(forest +6.4–9.6pp, wide +1.2–4.2pp, narrow +2.6–6.0pp at n=500/kind), and are
also slightly more efficient (lower makespan/flowtime than MST throughout). The
CNN U-Net leads on forest (hybrid 79.6%); the Transformer leads on narrow
(48.8%) and wide (74.4%); the two stay within ~1–3pp. Hybrid checkpoints are the
best-by-success iterate (`runs/rl.pt` mirrors `best.pt`). `runs/fields_rl.png`
shows the mechanism: the MST field is piecewise-constant in coarse blocks, while
the learned field is a smooth fine-grained gradient that breaks symmetry more
precisely at junctions.

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
