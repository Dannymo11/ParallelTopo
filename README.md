# TopoGraph

Batched GPU simulation for graph-structured reinforcement learning environments
with **variable topology**. CS348K final project, Spring 2026.

## What this is

Existing batched simulators (Madrona, Brax, Isaac Gym) assume fixed environment topology. 
This project asks whether the same playbook of a GPU-resident state, per-environment masking. 
and vmap'd dynamics, extends to environments where each environment in the batch is *building a
different graph* over the course of an episode.

Concrete workload: a slim transit-network construction simulator on synthetic
grid cities (10×10 to 15×15 zones, 15-step episode horizon). Each step adds
one edge subject to a budget; reward is an accessibility-weighted welfare
function computed over batched approximate all-pairs shortest paths.

## Project questions

Three questionsanchor the work, each tied to a quantitative success gate.

**RQ1 — Feasibility.** *Can a graph-structured RL environment with variable
topology be efficiently batched on a single GPU using approximate batched APSP
and edge-existence masking, while preserving simulator semantics?*

> **Success gate.** Mean travel-time error < 2% vs. exact APSP at *K =
> graph diameter* Bellman-Ford iterations; ≥ 10× simulator-only rollouts/sec
> at batch 256 vs. the CPU baseline; fits on a 24 GB GPU at 10×10–15×15
> grids.

**RQ2 — Cost of variable topology.** *How much overhead does variable
topology add vs. a fixed-topology batched simulator on the same workload?*

> **Success gate.** A clean throughput comparison between (a) the
> variable-topology simulator and (b) a fixed-topology variant where the
> edge mask is frozen at episode start, reported as overhead percentage at
> matched batch and graph sizes. The point is to *quantify* the cost of the
> contribution, not to prove it's free — this is the figure that
> distinguishes us from existing batched simulators (Madrona, Brax, Isaac
> Gym).

**RQ3 — End-to-end training throughput.** *Does simulator throughput
translate to RL training throughput, or does the policy / host-transfer
boundary dominate?*

> **Success gate.** Env-steps/sec measured at the training boundary with a
> small MLP policy at batch sizes {1, 16, 64, 256, 1024}. Identify the
> regime where simulator throughput stops being the bottleneck — that is
> the "where does the time go" insight figure.

## Layout

```
src/topograph/
  sim_cpu/      # NumPy/SciPy CPU baseline 
  sim_gpu/      # JAX batched simulator    
  bench/        # rollouts/sec benchmark harness
  policies/     # random / greedy / learned
tests/          # sanity + numerical-equivalence
results/        # benchmark outputs, profiling, figures
docs/           # checkpoint and final writeup
```

## Setup

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev,figures]"
```

## Checkpoint status

The CPU baseline is implemented and the evaluation harness produces real
numbers.

**Throughput floor** (10×10 grid, horizon 15, random policy, 1000 episodes,
scipy-backed Floyd-Warshall):

| Metric | Value |
|---|---|
| Rollouts / sec | 117.8 |
| Env-steps / sec | 1,767.4 |
| Mean episode wall-time | 8.49 ms |
| Episode return (random) | 204.2 ± 4.59, range [193.9, 227.4] |

The APSP backend is `scipy.sparse.csgraph.floyd_warshall` — chosen
deliberately to give the CPU baseline a fair shot. An earlier
hand-vectorized NumPy FW ran at 66.4 rollouts/sec (15.07 ms/episode); the
swap to scipy delivered a 1.78× end-to-end speedup with bit-identical
output. Quoting the slower number as the "CPU baseline" would have
artificially inflated the M3 GPU speedup figure, which the systems-paper
write-up cannot afford. The hand-vectorized version is retained as
`compute_travel_times_numpy` for the M3 GPU-port correctness cross-check
and as the input to `scripts/apsp_baseline_comparison.py`.

**Where the time goes** (10×10, horizon 15, 1500 step samples,
scipy backend):

| Component | Mean (µs/step) | Fraction |
|---|---:|---:|
| `compute_travel_times` (Floyd-Warshall APSP) | 417.9 | **74.9%** |
| `compute_demand` | 70.7 | 12.7% |
| `compute_accessibility` (×2) | 56.9 | 10.2% |
| `update_activity` | 8.7 | 1.6% |
| `compute_welfare` | 2.0 | 0.4% |
| `bookkeeping` + `apply_action` | 1.7 | 0.3% |

APSP is decisively the hot spot.

**Sanity tests** distinguish "simulator wired up" from "simulator returns
zero on every input":

* *Walking-only floor* — no-op rollouts have non-zero return because
  walking accessibility itself contributes welfare.
* *Edges help* — a hand-built central corridor strictly beats an empty
  rollout on the same world.
* *Greedy beats random* — one-step greedy strictly outperforms uniform
  random when averaged over 5 seeds.

73 tests in total; all passing.

## Checkpoint 2 — Batched APSP feasibility (2026-05-22)

The make-or-break technical risk for the whole project — whether batched
shortest paths can be vectorized across environments with *different*
active edge sets at acceptable accuracy — landed decisively today. The
throughput and memory halves closed on 2026-05-31 with a GPU run on a
Modal A100 — **all three M2 gates now pass.**

### M2 throughput + memory — closed (2026-05-31)

GPU run on a Modal A100 (40 GB); batched APSP, K=5, fp32, horizon 15.
Reproduce: `modal run scripts/modal_m2_throughput.py` (or `python
scripts/m2_apsp_throughput.py` on any CUDA box). Artifact:
`results/m2_apsp_throughput/m2_throughput_20260531T191537Z.{json,csv}`.

**Throughput gate — PASS.** 10×10/horizon-15 reference workload,
simulator-only, batch 256: **5,994 rollouts/sec = 50.9× the scipy CPU
baseline** (117.8 rollouts/sec). Gate was ≥1,150 (≥10×); cleared by ~5×.
Throughput is flat-to-rising once the GPU saturates; the sweet spot is
broad (batch 64–1024).

| grid | batch | rollouts/sec | speedup | peak MB |
|---|---:|---:|---:|---:|
| 10×10 | 1 | 727.5 | 6.2× | 0 |
| 10×10 | 16 | 4,822.6 | 40.9× | 4 |
| 10×10 | 64 | 6,284.5 | 53.3× | 15 |
| 10×10 | 256 | **5,994.1** | **50.9×** | 59 |
| 10×10 | 1024 | 7,263.3 | 61.7× | 234 |
| 15×15 | 256 | 1,160.2 | — | 297 |
| 15×15 | 1024 | 1,158.9 | — | 1,187 |

**Memory gate — PASS, by a wide margin.** Peak device memory is 297 MB at
15×15/batch 256 and ≤1.2 GB across the entire sweep — far under 24 GB. The
naive (B, N, N, N) min-plus bound (11.7 GB at 15×15/256, 46 GB at
15×15/1024) never materializes: XLA fuses the broadcast-add-and-min into a
single reduce kernel, so peak memory scales as O(B·N²), not O(B·N³). The
cubic memory wall is pushed well past N=225 — bf16 / K-axis tiling are not
needed at M1-spec sizes.

**Note on 15×15 throughput.** The gate is defined on the 10×10 reference
workload (where the 117.8 baseline was measured); 15×15 is run for the
memory gate and the M5 graph-size figure. Comparing 15×15 GPU throughput
to the *10×10* CPU baseline understates it — a like-for-like 15×15 CPU
rollout is ~3–6 rollouts/sec, so 15×15 GPU at 1,160 rollouts/sec is
~200–370×.

### Evaluation framework

The four figures the writeup will contain, with current status:

| # | Figure | Status | What it shows |
|---|---|---|---|
| 1 | Throughput vs. batch size on fixed grid | **done** | Full-simulator rollouts/sec at batch {1, 16, 64, 256, 1024} on 10×10/horizon-15. GPU verdict 2026-06-01: **60.0× the scipy CPU baseline at batch 256** (7,068 rollouts/sec, full on-device `rollout_random`); the APSP-only kernel alone was 50.9× (M2). Scipy FW @ 117.8 rollouts/sec as the horizontal reference. End-to-end-training curve pending M4. |
| 2 | Throughput vs. graph size at fixed batch | **done** | Batch 256/1024, N over {25…900} (5×5…30×30). GPU 2026-06-01: throughput falls 6,252→20 rollouts/sec (10×10→30×30) ~O(N³); peak memory O(B·N²), 30×30@1024 = 12.7 GB (no OOM). Compute, not memory, is the ceiling. `scripts/m5_fig2_graphsize.py`. |
| 3 | Where-does-time-go per-component breakdown | **done** | CPU vs GPU at matched 10×10. Shortest-path (APSP) share rises 74.9%→~94% on GPU; the dynamics collapse from ~25% to ~6%. `results/profiling/{cpu,gpu}_breakdown_*`. |
| 4 | Variable-topology overhead vs. fixed-topology | **done** | Production vs frozen-mask, matched workload. GPU 2026-06-01: **mean 0.25%, max 1.24% overhead** → variable topology is effectively free vs a fixed-topology baseline. Contribution-distinguishing experiment vs. Madrona / Brax / Isaac Gym. `scripts/m5_fig4_overhead.py`. |

### What's been answered (real intermediate results)

**M2 accuracy gate — passed cleanly (today).** Across a sweep of 3 grid
shapes × 5 mask densities × 3 seeds × 2 candidate algorithms × 7 K values
(630 cells total), the worst mean relative error at the M2-targeted K is
**0.003%**. At convergence-K the vast majority of cells are bit-identical
to scipy FW. Algorithm choice for M3: matrix-squaring (`D ← min(D, D ⊕ D)`)
over Bellman-Ford-style (`D ← min(D, D ⊕ A)`) — same answer, ~4× fewer
iterations. Reproduction: `python scripts/batched_apsp_accuracy.py`;
results: `results/m2_apsp_accuracy/m2_accuracy_20260522T212737Z.json`.

| Grid | First fully-converged K (Bellman-Ford) | First fully-converged K (matrix-squaring) |
|---|---:|---:|
| 5×5 | 4 | 3 |
| 10×10 | 16 | 4 |
| 15×15 | 16 | 4 |

K=5 is the recommended single value across all M1-spec grid sizes (one
safety iteration past the empirical convergence point; 2⁵ = 32 hops
covers the ≤28-hop walking-floor diameter of a 15×15 grid with margin).

**Writeup framing shift the accuracy data supports.** At K=5 on M1-spec
grids the relaxation produces *exact* shortest paths, not approximate
ones — the walking floor bounds the hop-diameter low enough that
fixed-iteration matrix-squaring converges to bit-identical scipy-FW
output. The contribution reframes from "approximate batched APSP with
bounded error" to "**exact** batched APSP via fixed-iteration min-plus
matrix-squaring, with K chosen from graph diameter." Stronger story —
no accuracy/throughput trade-off curve to defend, just a fixed O(log D)
compute cost per step.

**JAX kernel correctness — smoke test passed.** The `apsp_batched`
kernel runs end-to-end under `jax.jit` + `jax.vmap` at all
laptop-feasible batch sizes; the test suite (`tests/test_apsp_jax.py`)
asserts agreement with scipy FW within fp32 round-off across the same
(grid × density) regime as the accuracy sweep. The smoke test confirms
the GPU port is correctly wired; the throughput verdict is the only
remaining M2 question.

### What's still unanswered

| Question | Required run | Target / when |
|---|---|---|
| ~~M2 throughput gate~~ | **CLOSED 2026-05-31** | 5,994 rollouts/sec @ batch 256 = 50.9× scipy (gate ≥10×) |
| ~~M2 memory gate~~ | **CLOSED 2026-05-31** | 297 MB @ 15×15/256, ≤1.2 GB peak (≪24 GB) |
| ~~M3 — full vectorized simulator on GPU~~ | **DONE 2026-06-01** | all components vmap'd + assembled step + on-device `rollout_random`; equivalence-tested vs CPU at fp32 (130 tests pass) |
| ~~Figure 1: throughput vs. batch size on GPU~~ | **DONE 2026-06-01** | full simulator 60.0× scipy @ batch 256 (7,068 rollouts/sec); `scripts/m3_throughput.py` |
| ~~Figure 2: throughput vs. graph size~~ | **DONE 2026-06-01** | O(N³) throughput drop, O(B·N²) memory, no OOM through 30×30; `scripts/m5_fig2_graphsize.py` |
| ~~Figure 3 (GPU half): per-component breakdown on GPU~~ | **DONE 2026-06-01** | shortest-path share 74.9%→~94% (CPU→GPU) at 10×10; `python -m topograph.bench.gpu_profiling` |
| ~~Figure 4: variable-topology overhead~~ | **DONE 2026-06-01** | mean 0.25% / max 1.24% overhead; `scripts/m5_fig4_overhead.py` |
| **M4** — end-to-end RL training throughput | wire batched simulator into SB3 / PureJaxRL | most likely scope cut if M3 slips; see below |

### Plan to the presentation

The plan from here to the 2026-06-02 presentation:

| Date | Work | Why this date |
|---|---|---|
| 2026-05-31 *(done)* | GPU run on Modal A100; closed M2 throughput + memory bars (50.9×, ≤1.2 GB) | — |
| 2026-05-25 — 2026-05-29 | M3 — vmap the rest of the simulator | ~5 days for a compressed M3; the architectural pattern is set by `apsp_batched` |
| 2026-05-30 — 2026-05-31 | M5 figures (1–4) and M6 writeup | Reproduction scripts already exist for Figs 1 and 3 (CPU); GPU runs are quick once M3 is in |
| 2026-06-01 | Final figure polish + presentation rehearsal | Buffer day |
| 2026-06-02 | CS348K final presentation | Hard deadline |

**Most likely scope cut if M3 slips: M4 (end-to-end RL training
integration / RQ3).** The systems contribution can be presented on
simulator-only throughput (Figs 1, 2, 3) plus the variable-topology
overhead figure (Fig 4), which is the contribution-distinguishing
experiment. M4 becomes a "future work" line in the writeup. RQ3 (does
simulator throughput translate to training throughput) is interesting
but secondary to the core systems claim, and reviewers of this style of
paper accept "training integration left for follow-up" when the
simulator results stand on their own.

## Checkpoint 3 — Full GPU simulator (2026-06-01)

M3 is complete: every per-step component is ported to JAX and the whole
simulator runs on-device. `sim_gpu/dynamics.py` has `compute_accessibility`,
`update_activity`, `compute_demand`, `apply_action`, `compute_welfare`
(single-env + vmapped batched); `sim_gpu/step.py` assembles the vmapped
`step` over an ECS-like `GPUState` with on-device direct-distance
construction and the `freeze_mask` ablation flag; `sim_gpu/policy.py` adds
the uniform-random-legal policy and a `lax.scan` `rollout_random` driver.
Correctness is pinned by equivalence tests against the CPU baseline at fp32
(`tests/test_dynamics_jax.py`, `test_step_jax.py`, `test_policy_jax.py`) —
including a B=1 full-rollout reward+final-state match and aggregate reward
parity (GPU random mean return 204.1 vs CPU 203.4). 130 tests pass.

**Full-simulator throughput (the M5 Figure 1 headline).** A100 run via
`scripts/m3_throughput.py` (reproduce: `modal run
scripts/modal_m3_throughput.py`). Artifact:
`results/m3_throughput/m3_throughput_20260601T045732Z.{json,csv}`.

| grid | batch | rollouts/sec | speedup | peak MB |
|---|---:|---:|---:|---:|
| 10×10 | 1 | 503.5 | 4.3× | 0 |
| 10×10 | 16 | 3,546.4 | 30.1× | 3 |
| 10×10 | 64 | 5,560.0 | 47.2× | 10 |
| 10×10 | 256 | **7,068.2** | **60.0×** | 40 |
| 10×10 | 1024 | 6,973.6 | 59.2× | 158 |
| 15×15 | 256 | 1,149.8 | — | 199 |
| 15×15 | 1024 | 1,146.3 | — | 796 |

**Throughput gate — PASS at 60.0×** (7,068 rollouts/sec at batch 256 on the
10×10 reference workload, vs the ≥10× / ≥1,150 gate). This is the *full
simulator* against the full scipy CPU baseline (117.8) — a like-for-like
number, and the honest Figure 1 headline.

**The full simulator is faster than APSP-only.** The M2 APSP-only bench hit
5,994 rollouts/sec at batch 256; the full simulator hits 7,068 — more work
per step, yet ~18% faster. The cause is dispatch amortization: M2 issued 15
separate `jit` dispatches of `apsp_batched` over a static matrix, while M3
fuses the entire 15-step episode (APSP + all dynamics + policy) into one
`lax.scan` under a single dispatch, eliminating per-step launch overhead and
host round-trips and letting XLA fuse across steps. The systems takeaway:
the win is not only vectorization across environments but fusing the whole
rollout into one compiled program. (At batch 1 the full sim is slower — 503
vs 727 — where fixed per-rollout cost dominates; the curves cross as the
batch fills the GPU.)

**Memory stays tiny.** Peak ≤796 MB across the whole sweep (40 MB at the
10×10/256 gate cell), confirming again that XLA fuses the (B, N, N, N) APSP
intermediate away — the full simulator keeps the same O(B·N²) profile as the
kernel alone.

## M5 figures — graph-size scaling & variable-topology overhead (2026-06-01)

Both run on a Modal A100. Reproduce: `modal run
scripts/modal_m5_fig2_graphsize.py` and `modal run
scripts/modal_m5_fig4_overhead.py`. Artifacts in
`results/m5_fig2_graphsize/` and `results/m5_fig4_overhead/`.

### Figure 2 — throughput vs. graph size at fixed batch

Full-simulator rollouts/sec as N grows, at batch 256 (batch 1024 is within
noise of this for N≥100 — the simulator is compute-bound, not
occupancy-bound, above tiny grids):

| grid | N | rollouts/sec | env-steps/sec | peak MB |
|---|---:|---:|---:|---:|
| 5×5 | 25 | 138,530 | 2,077,949 | 1 |
| 10×10 | 100 | 6,252 | 93,787 | 40 |
| 15×15 | 225 | 1,152 | 17,281 | 199 |
| 20×20 | 400 | 248 | 3,720 | 628 |
| 25×25 | 625 | 47 | 710 | 1,531 |
| 30×30 | 900 | 20 | 295 | 3,172 |

Two findings. (1) **The memory wall never bites** — peak memory tracks
O(B·N²), not the naive O(B·N³); even 30×30 at batch 1024 peaks at 12.7 GB,
far under 40 GB. The XLA-fusion result from M2/M3 holds at scale. (2) **The
ceiling is compute** — in the N≥225 regime per-rollout time scales ~O(N³)
(the K=5 APSP), so throughput falls 6,252→20 rollouts/sec from 10×10 to
30×30 and is compute-impractical well before it OOMs. Finding the
compute ceiling, not a memory ceiling, is the stronger result. (The memory
column is the clean batch-256 sweep; the cumulative peak counter makes
mid-range batch-1024 readings stale.)

### Figure 3 — where-does-time-go, CPU vs. GPU

GPU per-component breakdown via `python -m topograph.bench.gpu_profiling` (run
on the A100; artifacts `results/profiling/gpu_breakdown_*`). At the matched
10×10 operating point, against the CPU profile (`cpu_breakdown_*`):

| | CPU (scipy, 10×10) | GPU (10×10, batch 256) |
|---|---:|---:|
| shortest paths (direct-distance build + APSP) | 74.9% | ~94% |
| all other dynamics (demand, accessibility ×2, update, welfare, action) | ~25% | ~6% |

Read the GPU step by its ground truth, not the isolated per-component table:
the fused whole-step is 3.34 ms, and the shortest-path work (APSP 2.64 ms +
direct-distance build 0.51 ms) is 3.15 ms of it — **~94%**. The dynamics that
cost ~23% on the CPU (demand 12.7% + accessibility 10.2%) collapse to a few
percent on the GPU: the dense matmuls and outer products parallelize and fuse
almost for free, while the O(N³) min-plus APSP stays memory-bandwidth-bound
and remains the wall. (The isolated-op table reads 64.6% for shortest paths
only because each op pays a fixed ~0.8 ms kernel-launch overhead — visible as
the spurious 14% "bookkeeping" for a `step+1` — which the `sum/fused = 1.49×`
gap quantifies and which disappears under fusion.)

**The bottleneck does not move off APSP — it intensifies** (75% → ~94%), and
it grows with N: at 15×15 the shortest-path share is ~97% (`gpu_breakdown_step_15x15_*`),
tracking Figure 2's O(N³) scaling. Direct-distance construction is a roughly
constant ~0.5 ms, so its share shrinks from ~15% at 10×10 to ~3% at 15×15. The
takeaway: the GPU makes everything *except* shortest paths nearly free, so the
60× speedup is batching + fusing the APSP across environments — exactly the
component the fixed-iteration matrix-squaring kernel was built to accelerate.

### Figure 4 — variable-topology overhead vs. fixed-topology baseline

The contribution-distinguishing experiment: identical workload,
`apply_action`'s per-step mask scatter on (production) vs off (frozen
topology), measured per-step so both run all 15 APSPs (a fused `lax.scan`
would let XLA hoist the frozen APSP and confound the measurement — see
`scripts/m5_fig4_overhead.py`):

| grid | batch | overhead |
|---|---:|---:|
| 10×10 | 64 | 1.24% |
| 10×10 | 256 | 0.20% |
| 10×10 | 1024 | 0.04% |
| 15×15 | 64 | −0.01% |
| 15×15 | 256 | −0.03% |
| 15×15 | 1024 | 0.05% |

**Variable topology costs ≤1.2% (0.25% mean)** over a fixed-topology
baseline at matched batch and graph size, falling to noise as batch/grid
grow — the per-step mask scatter is negligible against the K=5 APSP. This is
the number that distinguishes TopoGraph from fixed-topology batched
simulators (Madrona / Brax / Isaac Gym): the variable-topology capability is
effectively free.

## M4 — End-to-end training throughput (scaffolded 2026-06-01)

RQ3 asks whether the simulator's throughput survives once a policy and a
gradient update are in the loop, or whether the policy / host boundary
dominates. The on-device scaffolding for the answer is built and tested;
the GPU numbers are one Modal run away.

**What landed (items 1–3 of the M4 plan).**

* `sim_gpu/mlp_policy.py` — a hand-rolled 64-unit MLP (no flax/optax, so the
  proven `jax[cuda12]==0.5.3` Modal image is unchanged), a bounded state
  featurizer (`activity ⧺ edge_mask ⧺ budget-frac ⧺ step-frac`), masked
  categorical sampling, and three drivers, all under one `jit`/`lax.scan`
  (PureJaxRL-style, so the policy and the simulator never leave the device):
  `rollout_mlp` (simulator + MLP inference) and `train_step` (simulator +
  inference + REINFORCE backward + an SGD update), plus `reinforce_loss`.
* `scripts/m4_train_throughput.py` — times **sim-only** (`rollout_random`)
  vs **policy-inference** (`rollout_mlp`) vs **train-step** across batch
  {1, 16, 64, 256, 1024} on 10×10, reports env-steps/sec and the
  `infer/sim` and `train/sim` ratios, and prints the RQ3 verdict (throughput
  retained under the policy + the batch where the train-step curve flattens).
  Writes `results/m4_train_throughput/m4_train_throughput_<ts>.{json,csv}`.
* `scripts/modal_m4_train_throughput.py` — A100 runner (a copy of the M3
  runner pointed at the new script; no new image deps).
* `scripts/plot_m4_train_throughput.py` — draws the three-driver
  env-steps/sec-vs-batch chart (Figure 1b) from the JSON, titled with the
  % of simulator throughput the train step retains at the sweet spot.
* `tests/test_mlp_policy_jax.py` — legality of every sampled action,
  key-determinism, shapes, and a `train_step` test that asserts a finite
  loss and that **every** parameter moves (gradients flow through the
  scanned rollout). 48 GPU-path tests pass on CPU JAX.

**Reproduce (GPU):**

```bash
modal run scripts/modal_m4_train_throughput.py     # writes results/m4_train_throughput/
python scripts/plot_m4_train_throughput.py          # renders Figure 1b from the latest JSON
```

The agent is deliberately not trained to do anything useful (an explicit
M4 non-goal): the policy exists to consume the simulator at a realistic
rate so the boundary cost is real. Correctness here means legality,
determinism, and gradient flow — not policy quality.

**Still open (item 4):** the GPU-vs-CPU end-to-end *training* speedup needs
a realistic CPU baseline — a multi-process SB3 `SubprocVecEnv` running PPO
on the same task. That is sketched in `scripts/m4_cpu_baseline_sb3.py`
(Gymnasium wrapper around `sim_cpu` with the same observation features as
the GPU policy) but not run here, because it pulls in `stable-baselines3`,
`gymnasium`, and `torch` — kept out of the GPU image and meant for a CPU
box. The GPU `train_step` env-steps/sec is the upper bound that baseline is
measured against; until it lands, RQ3 is reported as the on-device retained
throughput plus the simulator-only upper bound.

## Planned experiments

1. **Throughput vs. batch size sweep.** Rollouts/sec at batch sizes
   {1, 16, 64, 256, 1024, 4096} on a fixed 10×10 grid. Both simulator-only
   and end-to-end-training numbers. CPU baseline as a horizontal line.
   *Success:* see RQ1 gate above.
2. **Throughput vs. graph size sweep.** Fix batch size at the sweet spot
   from (1); vary zones over {25, 100, 225, 400}. *Success:* identifies
   where the approach breaks down (memory, APSP cost) — finding a
   ceiling is a successful outcome, not a failure.
3. **Where-does-the-time-go breakdown, GPU edition.** Re-run the
   instrumented step on the GPU at one well-chosen operating point and
   compare CPU vs. GPU per-component fractions side-by-side. *Success:*
   APSP fraction drops dramatically; the new hot spot becomes
   policy-network or host-device transfer.
4. **Variable-topology overhead.** Compare against a fixed-topology
   batched simulator (no edge addition during episodes) on the same
   workload. *Success:* a clean overhead-percentage figure at matched
   batch and graph sizes — the contribution-distinguishing experiment
   vs. Madrona / Brax / Isaac Gym (per RQ2).

## Reproducing the checkpoint numbers

```bash
# 1. Test suite (73 tests including 3 sanity tests that reject trivial outputs)
pytest

# 2. Throughput floor
python -m topograph.bench \
    --grid 10 --horizon 15 --policy random \
    --episodes 1000 --warmup 10
# Writes: results/bench/cpu_baseline_random_10x10_h15_<ts>.json

# 3. Per-component time breakdown
python -m topograph.bench.profiling \
    --grid 10 --horizon 15 --episodes 100 --warmup 5
# Writes: results/profiling/cpu_breakdown_random_10x10_h15_<ts>.{json,csv,png}
```

## Milestones

* **M1** *(done)* — CPU baseline simulator + throughput floor + profiling.
* **M2** *(done 2026-05-31)* — Batched APSP feasibility study; all three gates passed (accuracy, throughput 50.9×, memory ≤1.2 GB).
* **M3** *(done 2026-06-01)* — Full vectorized simulator on GPU; full-simulator throughput 60.0× scipy @ batch 256 (7,068 rollouts/sec), equivalence-tested vs CPU at fp32.
* **M4** *(scaffolded 2026-06-01)* — End-to-end training throughput. On-device MLP policy + REINFORCE `train_step`, the three-driver throughput harness, and equivalence/grad tests have landed (items 1–3); GPU numbers pending a Modal run. The SB3 CPU training baseline (item 4) is sketched. See below.
* **M5** — Throughput study and ablations.
* **M6** — Writeup and CS348K final presentation.

## Generalization — transit_learning APSP (2026-06-01)

A check that the M2/M3 contribution isn't specific to our own grid cities: we
applied the batched APSP kernel (fixed-iteration min-plus matrix-squaring) to an
unrelated external transit-optimization codebase — Holliday & Dudek's
`transit_learning`, which uses a vectorized Floyd-Warshall on its shortest-path
hot path — and benchmarked it against that codebase's own FW on the real Mandl
and Mumford city graphs (N = 15–127). Harness, data loaders, Modal runner, and
figure live in [`transit_learning/`](transit_learning/) (PyTorch port, so the
comparison runs in one runtime).

On a Modal A100-40GB the kernel is **bit-exact with Floyd-Warshall at K = 5** and
gives **20–117× speedups in the single-graph / small-batch regime** (it needs
only `⌈log2 N⌉` sequential steps vs FW's `N`), narrowing to ~1.5–2× when hundreds
of large graphs are batched — where, under PyTorch eager, it also hits an
O(B·N³) memory wall that XLA fusion eliminated in our JAX simulator. The
systems takeaway reinforces M5 Figure 3: the kernel's batch-scaling on GPU
depends on fusion, not the algorithm alone, and its clearest win is
latency-bound low-batch shortest-path evaluation on large graphs. This is
write-up material for the Discussion section (Linear `TOP-31`); see
[`transit_learning/README.md`](transit_learning/README.md) for the full tables
and reproduction.
