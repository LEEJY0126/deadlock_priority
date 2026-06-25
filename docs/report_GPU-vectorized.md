# Report: GPU-Vectorized Envs Experiment

Companion to the main `report.md`. Records the design, results, and conclusion of
the `GPU_vectorized` branch experiment — testing whether running the RL rollouts
as one batched GPU computation beats the CPU path. This is the local (gitignored)
long-form report; a condensed, tracked version lives in `GPU_VECTORIZED.md` on
the branch.

## Motivation

RL fine-tuning is dominated by `batch_maps × K` PIBT episode rollouts per step,
which are CPU-bound. Two ways to parallelize them:

1. **CPU multiprocessing** — a process pool over the independent rollouts. Already
   merged on `main` (`train_rl.py --workers`), ~3.2× wall-clock, exact PIBT.
2. **GPU vectorized envs** — simulate thousands of episodes simultaneously as
   tensor lanes (Isaac-Gym/Brax style). This experiment.

## Why this is NOT PIBT

PIBT resolves conflicts with **priority inheritance and backtracking** — a
recursive, data-dependent procedure. On a GPU's SIMT model that means warp
divergence and variable-depth recursion, which do not vectorize cleanly.

`VecSim` (`src/envs/vec_sim.py`) therefore uses a **GPU-friendly approximation**:

- All E episodes are torch tensors stepped in lockstep.
- Each step, agents claim next cells in **priority order** (one priority rank at a
  time, vectorized across all E envs). Vertex conflicts are resolved **exactly**
  (highest-priority claimant wins a cell; others fall back to their next-best
  candidate, ultimately "stay").
- A **head-on-swap fix-up** forces the lower-priority agent of a swapping pair to
  stay.
- **No inheritance, no backtracking** → it drops PIBT's reachability guarantee.

Priority mirrors `simulator.py`: normalized field value at the current cell, a
small index tie-break, a stuck-time boost (β), and yielding once arrived.

It is intended for throughput benchmarking and fast approximate rollouts, not as
a drop-in replacement for the exact CPU solver.

## Benchmark setup

`scripts/bench_vec.py`: builds E identical episodes (same maps/instances/MST
fields), runs them through both the CPU `Simulator` and `VecSim`, and reports
episodes/second. Hardware: RTX 5080, 21×21 maps, 8 agents, `max_steps=128`,
balanced forest/wide/narrow.

## Results

```
sanity (open map, 64 eps): vec success=100%  cpu success=92%

     E    CPU eps/s    GPU eps/s  GPU build/s   step speedup
    32         43.9        206.2         52.3          4.7x
   128         46.4        726.2         51.5         15.6x
   512         44.4       3367.3         53.7         75.9x
  2048       (skip)      11880.7         53.2     ~270x (vs 44)
```

- **CPU eps/s** — end-to-end CPU `Simulator` (≈44, flat in E).
- **GPU eps/s** — `VecSim.run()` stepping only.
- **GPU build/s** — `build_batch()` setup only (per-episode BFS distance fields,
  CPU).

## Analysis

1. **GPU stepping scales strongly with batch size** — 4.7× (E=32) → ~76×
   (E=512) → ~270× extrapolated (E=2048). Confirms the GPU *can* win, but only at
   large batch; small batches don't amortize kernel-launch overhead. At our real
   RL scale (`batch_maps × K ≈ 32` envs/step) the win is just ~4.7×.

2. **The bottleneck shifts to setup.** `build_batch` builds one single-source BFS
   distance field per (env, agent) on the CPU, flat at **~53 eps/s — barely above
   the CPU simulator's 44**. So **end-to-end (build + run), GPU vectorization is
   throttled by CPU-side BFS** and the stepping speedup is largely hidden. At
   E=8192 the BFS setup (~65k BFS calls) dominates outright — that benchmark row
   times out in setup, not stepping.

3. **Mitigations** to make the stepping win show up end-to-end:
   - **Amortize BFS across the K perturbed fields.** Distances depend only on
     map+goals, not the priority field, so the K fields sharing an RL instance
     reuse one BFS (÷K).
   - **Parallelize BFS on CPU** (the existing `--workers` idea).
   - **Vectorized GPU BFS** — iterative grid relaxation over all (env, agent)
     goals at once, moving setup off the CPU critical path.

4. **Solver-quality caveat.** Being non-backtracking, `VecSim` resolves fewer
   deadlocks than real PIBT on hard mazes (the sanity map is easy: 100% vs CPU
   92%). This path trades solver quality for throughput; using it for RL would
   change the optimization target, not just its speed.

## Conclusion / recommendation

| Approach | Scale where it wins | Solver | Effort | Status |
|----------|--------------------|--------|--------|--------|
| CPU multiprocessing (`--workers`) | current (~32 envs/step) | exact PIBT | low | ✅ on `main` |
| GPU vectorized envs (`VecSim`) | thousands of envs/step | approximate | high | experiment (this branch) |

**At the current scale, CPU multiprocessing is the right tool** — exact PIBT,
~3.2×, no solver change. **GPU vectorization pays off only if** you (a) scale to
thousands of envs per step, (b) also move/amortize the BFS setup off the CPU
critical path, and (c) accept the approximate non-backtracking solver. Kept as a
documented experiment on the `GPU_vectorized` branch for if/when env count is
scaled up.

Reproduce: `python scripts/bench_vec.py --device cuda --max_steps 128 --counts 32,128,512,2048 --cpu_max 512`
