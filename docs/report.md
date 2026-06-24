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

## What is different

| Aspect | Paper | This project |
|--------|-------|--------------|
| **Priority source** | Hand-designed MST priority tree | **Learned CNN field** (hybrid imitation → RL) |
| **State space** | Continuous 3D, double-integrator dynamics | **Discrete grid**, agents hop cell-to-cell |
| **Inter-agent collision avoidance** | ABVC + Safe Flight Corridor + QP trajectory optimization (with Theorems 1–2) | **Not modeled** — only PIBT's grid vertex/swap conflict rules |
| **Asynchronous updates** | Core contribution; ABVC guarantees safety under async replanning | **Not modeled** — synchronous, centralized PIBT step |
| **Communication-free** | Achieved via position priority + ABVC | Priority is comms-free by construction (map-only input); the rest of the async/sensing machinery is absent |
| **Limited sensing range** | Explicit constraint (Eq. 11) | **Not modeled** — full observability in the simulator |
| **Deadlock / livelock handling** | Right-hand rule + livelock detection + limited-sensing conflict resolution (Alg. 3) | Replaced by the **learned field** + a generic PIBT anti-starvation boost |
| **Validation** | Simulation + 8-quadrotor hardware | Grid simulation only |

## Design choices that depart from the paper (and why)

These are abstraction decisions made so that *priority quality* is the only
variable under study:

1. **Anti-starvation boost (`beta`).** A purely static field cannot resolve a
   1-wide corridor (agents must back out — PIBT's reachability needs a
   starvation-breaking term). We add a small boost ∝ stuck-time, applied
   **identically** to the MST baseline and the learned field, so the field
   remains the only difference. Fields are scale-normalized first so `beta`
   affects the integer MST field and the softplus learned field equally.
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

- Both methods run through the **same** simulator, PIBT solver, anti-starvation
  boost, maze distribution, and held-out instances. Only the priority field
  differs.
- Priority does not affect collision safety in the paper either (safety comes
  from ABVC/SFC/stop-constraint, Theorems 1–2). So studying priority in isolation
  does not discard a safety property — a worse field can only cause deadlock or
  inefficiency, never a collision.

## Result (held-out, 8 agents, 21×21, 60 instances/kind)

| map | MST baseline (paper) | imitation | hybrid (imitation→RL) |
|-----|:--------------------:|:---------:|:---------------------:|
| forest | 90.0% | 88.3% | **91.7%** |
| wide   | 81.7% | 86.7% | **90.0%** |
| narrow | 30.0% | 45.0% | **46.7%** |

The learned field beats the paper's heuristic on all three map types, with the
largest gains where deadlocks dominate. `runs/fields.png` shows the mechanism:
the MST field is piecewise-constant in coarse blocks, while the learned field is
a smooth fine-grained gradient that breaks symmetry more precisely at junctions.

## Ablation: removing the pooling layer

To test whether the U-Net's multiscale receptive field actually matters, we
retrained an identical model with `pool=False` (`--no_pool`): a full-resolution
flat CNN, no `MaxPool2d`. Same dataset, same training schedule; only the
architecture differs.

Success rate (held-out, 8 agents, 21×21, 60 instances/kind):

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

Checkpoints: `runs/imitation_nopool.pt`, `runs/rl_nopool.pt`. Reproduce with
`--no_pool` on `train_imitation.py` (the arch then propagates through RL/eval/viz
automatically via the checkpoint flag).

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
