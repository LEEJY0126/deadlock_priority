# Goal-livelock (`feature/goal-livelock`)

An **opt-in, isolated** deadlock-resolution branch for the case where an agent
keeps getting bounced off *its own goal* by through-traffic. Off by default
(`goal_livelock=False`); enable with `--goal_livelock` on `evaluate.py` /
`simulate.py`. Branched off `feature/right-hand-deadlock`.

---

## 1. The problem it targets

When an agent's goal cell sits on another agent's only path (common with the MST
field, whose root/low-priority cells lie in the corridors that traffic uses), the
finished agent is shoved off its goal, returns, is shoved off again — a stable
oscillation that the existing branches don't resolve:

- **Deadlock (Eq. 14/15)** doesn't fire: the oscillating agent isn't stationary,
  and its detector skips arrived blockers.
- **Livelock (Eq. 18)** doesn't fire cleanly: the anti-oscillation tie-break
  inflates whoever was just pushed down, hiding who should yield.
- The **arrived → `-inf`** rule (from `feature/right-hand-deadlock`) makes the
  finished agent always yield, which *enables* the oscillation rather than ending
  it: it is pushed off, regains priority one cell away, pushes back, repeats.

The canonical instance is **MST seed 20**: agent 3's goal `(7,7)` is a 1-wide
corridor cell that agent 0 (goal `(14,5)`) must traverse. They lock into a
`(7,6)↔(7,7)` / `(7,7)↔(7,8)` swap forever (6/8 reached).

## 2. Algorithm (as implemented)

Stateful per-agent **retreat**, evaluated every step (precedence **deadlock →
goal-livelock → livelock**):

**Detection (enter retreat).** Agent `i` was on its goal last step, is pushed
*exactly one cell* off, and `priority(current) > priority(goal)` (it was displaced
*upward* in priority).

**Temp goal.** The nearest **open** cell (`clearance ≥ 2`) reachable by a
**non-increasing-priority BFS** from the agent — i.e. descend the field, crossing
equal-priority plateaus (an open MST region shares one priority), and stop at the
first open cell; fall back to the lowest-priority cell if none is reachable
without going uphill (`_retreat_node`).

**While retreating.**
- Route to the temp goal via PIBT's **`subgoal`** path (staying *allowed*, unlike
  the back-out `cost` path which forces a move), so the agent can hold once it
  arrives.
- Keep **en-route priority even on its own goal cell** (the arrived-`-inf` rule is
  suppressed while retreating) — otherwise it is `-inf` while crossing `(7,7)` and
  gets shoved back before it can pass.

**Exit (return to real goal).** Only once the agent has **reached** the temp goal
*and* no agent that moved on the previous step is within **Manhattan 3** of it
(the traffic by the open cell has passed). Then drop the temp goal and route home.

Implemented in `src/envs/simulator.py`: `_goal_livelock_step`, `_retreat_node`,
`_gll_dist` (cached BFS), the `retreating[i]` argument to `_agent_priorities`, and
the `subgoal` route added to `pibt.step`.

## 3. Development log (what each commit found)

| commit | change | why |
|--------|--------|-----|
| `f223277` | initial retreat: greedy priority descent to clearance ≥ 2; hold by Manhattan-1 temp-goal proximity | first version; small net gain but the greedy descent **stopped at a clearance-1 local min** on the through-path |
| `629f18c` | **option B**: descent = non-increasing-priority **BFS** that crosses plateaus to a real `clearance ≥ 2` cell | the temp goal is now actually open (seed 20: `(7,8)`→`(7,9)`); bigger net gain |
| `57993dc` | **keep priority while retreating** + **exit only after reaching temp goal** + Manhattan 2 | tracing seed 20 showed the agent was `-inf` on its goal cell and got shoved back; and the exit released *before* it even reached the open cell |
| `b9b738a` | exit hold radius **Manhattan 3** | the blocker, pushed up its own column, hadn't cleared the chokepoint at radius ≤ 2; radius 3 waits long enough → **seed 20 solved** |

The decisive insight (raised in review): with retreat-keeping priority, the
agent at `(7,7)` [raw 4] **out-ranks** the blocker at `(7,8)` [raw 3], so it
**displaces** it (not a forbidden swap) and advances to the open cells — the
earlier "head-on, can't pass" reasoning was wrong.

## 4. Results

Held-out, 8 agents, 21×21, **n=100/kind** (20 maps × 5 instances), `paper` mode,
learned model `runs/rl_transformer.pt` (Transformer, dim96). Success rate, goal-
livelock **off → on** (final, Manhattan 3):

| map | MST off → on | Learned off → on |
|-----|:------------:|:----------------:|
| forest | 92 → **94** | 94 → **97** |
| wide   | 87 → **89** | 85 → **92** |
| narrow | 57 → **61** | 65 → **69** |

No regression; a clear gain on wide/narrow. Diagnostic seeds (MST, `paper`,
goal-livelock on):

| seed | result | note |
|------|--------|------|
| 20 | **8/8** (makespan 33) | the motivating case — now solved (was 6/8) |
| 54 | 7/8 (fail) | still structural (see limitations) |
| 56 | 8/8 | |
| 76 | 8/8 | |

(n=100 is noisy; a full n=500 sweep is a follow-up since goal-livelock is off by
default and not part of the headline numbers.)

## 5. Limitations

- **Seed 54 still fails.** Not every "goal on a path" instance has a reachable
  open cell whose hold neighbourhood the blocker passes through; a genuinely
  infeasible 1-wide corridor (no passing place at all) cannot be resolved by any
  local rule.
- **Manhattan 3 / clearance ≥ 2 are tuned, not derived.** They work on these
  braided 21×21 mazes; other map distributions may want different values.
- **Not in training / headline numbers.** Off by default; the shipped checkpoints
  were trained without it, and the README/report tables are goal-livelock-off.

## 6. Reproduce

```bash
# benchmark with / without goal-livelock
python scripts/evaluate.py --ckpt runs/rl_transformer.pt --n_per_kind 100 --n_inst 5
python scripts/evaluate.py --ckpt runs/rl_transformer.pt --n_per_kind 100 --n_inst 5 --goal_livelock

# watch the motivating instance resolve
python scripts/simulate.py --ckpt runs/rl_transformer.pt --map narrow --seed 20 \
    --max_steps 60 --raw --goal_livelock --out runs/gifs/goal_livelock/20.gif
```
