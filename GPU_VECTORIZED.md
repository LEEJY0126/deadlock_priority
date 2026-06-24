# GPU-Vectorized Envs (experimental branch)

Branch `GPU_vectorized`. Tests whether running the episode rollouts as a single
batched GPU computation ("vectorized envs", Isaac-Gym/Brax style) beats the CPU
path for our RL rollouts.

## What's here

- `src/envs/vec_sim.py` — `build_batch()` stacks E episodes into GPU tensors;
  `VecSim.run()` steps all E in lockstep.
- `scripts/bench_vec.py` — throughput benchmark vs the CPU `Simulator`, plus an
  open-map sanity check.

## ⚠️ This is NOT PIBT

PIBT's priority **inheritance with backtracking** is recursive and data-dependent
— it does not vectorize cleanly on a GPU (warp divergence, variable recursion
depth). `VecSim` uses a GPU-friendly approximation: a **single-pass,
priority-ordered reservation** (agents claim cells in priority order; vertex
conflicts resolved exactly; a head-on-swap fix-up) — **no inheritance, no
backtracking**, so it drops PIBT's reachability guarantee. It is for throughput
benchmarking and fast approximate rollouts, not a drop-in solver replacement.

## Results (RTX 5080, 21×21, 8 agents, max_steps=128)

```
sanity (open map, 64 eps): vec success=100%  cpu success=92%

     E    CPU eps/s    GPU eps/s  GPU build/s   step speedup
    32         43.9        206.2         52.3          4.7x
   128         46.4        726.2         51.5         15.6x
   512         44.4       3367.3         53.7         75.9x
  2048       (skip)      11880.7         53.2     ~270x (vs 44)
```

- **CPU eps/s** — end-to-end CPU `Simulator` (≈44, flat).
- **GPU eps/s** — `VecSim.run()` only (the stepping).
- **GPU build/s** — `build_batch()` setup only (per-episode BFS distance fields,
  built on CPU).

## Findings

1. **GPU stepping scales strongly with batch size** — 4.7× at E=32, ~76× at
   E=512, ~270× extrapolated at E=2048. The GPU *can* win big, but only at large
   batch (thousands of envs), exactly as predicted: small batches don't amortize
   kernel-launch overhead.

2. **The bottleneck shifts to setup.** `build_batch` (per-episode BFS distance
   fields, on CPU) is flat at **~53 eps/s — barely above the CPU simulator's 44
   eps/s.** So **end-to-end (build + run), GPU vectorization is throttled by
   CPU-side BFS setup**, and the huge stepping speedup is largely hidden. At
   E=8192 the BFS setup alone (~65k single-source BFS) dominates and is why that
   row times out.

3. **Mitigations** (to realize the stepping win end-to-end):
   - **Amortize BFS across the K perturbed fields.** Distances depend only on
     map+goals, not the priority field, so for the K fields sharing an instance
     in an RL step the BFS is computed once and reused (÷K).
   - **Parallelize BFS on CPU** — same idea as the `--workers` pool already on
     `main`.
   - **Vectorized GPU BFS** — iterative grid relaxation over all (env,agent)
     goals at once; moves setup off the CPU critical path entirely.

4. **Solver-quality caveat.** `VecSim` is the non-backtracking approximation, so
   on hard mazes it would resolve fewer deadlocks than real PIBT (the sanity open
   map is easy: 100% vs CPU 92%). This path trades solver quality for throughput.

## Recommendation

At the **current scale (≈32 envs / RL step), CPU multiprocessing (`--workers`,
already on `main`) is the right tool** — simple, exact PIBT, ~3.2× with no
solver change. **GPU vectorization pays off only if** you (a) scale to thousands
of envs per step, (b) also move/amortize the BFS setup off the CPU critical path,
and (c) accept the approximate non-backtracking solver. Kept as a documented
experiment for if/when env count is scaled up.

Reproduce: `python scripts/bench_vec.py --device cuda --max_steps 128 --counts 32,128,512,2048 --cpu_max 512`
