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
throughput half of the M2 gate is wired and correctness-tested but
requires a GPU run to verdict; that run is scheduled this weekend.

### Evaluation framework

The four figures the writeup will contain, with current status:

| # | Figure | Status | What it shows |
|---|---|---|---|
| 1 | Throughput vs. batch size on fixed grid | **partial** | Rollouts/sec at batch {1, 16, 64, 256, 1024} on 10×10/horizon-15 reference workload. CPU JAX baseline ran end-to-end (smoke test passed); GPU verdict pending. CPU baseline (scipy FW @ 117.8 rollouts/sec) as horizontal line. |
| 2 | Throughput vs. graph size at fixed batch | **stub** | Fix batch at sweet spot from Fig 1; vary N over {25, 100, 225, 400}. Identifies where the approach breaks down (memory, APSP cost). |
| 3 | Where-does-time-go per-component breakdown | **partial** | Per-component step time at one operating point. CPU figure ready (`results/profiling/cpu_breakdown_*.png`). GPU breakdown produced after M3 lands. |
| 4 | Variable-topology overhead vs. fixed-topology | **stub** | Re-run the simulator with the edge mask frozen at episode start; report overhead percentage at matched batch and graph size. Contribution-distinguishing experiment vs. Madrona / Brax / Isaac Gym. |

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
| **M2 throughput gate**: ≥1,178 rollouts/sec at batch 256 on 10×10/horizon-15 simulator-only (10× the scipy CPU baseline) | `python scripts/m2_apsp_throughput.py` on GPU | this weekend, 2026-05-23/24 |
| **M2 memory gate**: peak device memory at batch 256 fits comfortably on a 24 GB GPU (naive bound: 11.7 GB temp tensor at N=225) | same run | this weekend |
| **M3** — full vectorized simulator on GPU (demand, accessibility, land-use update, action application, reward — all vmap'd) | port from `sim_cpu` to `sim_gpu` | week of 2026-05-25 |
| **Figure 1**: throughput vs. batch size on GPU | M3 + final benchmark sweep | week of 2026-05-25 |
| **Figure 2**: throughput vs. graph size | M3 + sweep over N ∈ {25, 100, 225, 400} | M5, week of 2026-05-25 |
| **Figure 3 (GPU half)**: per-component time breakdown on GPU | M3 + instrumented step | M5 |
| **Figure 4**: variable-topology overhead vs. frozen-mask baseline | M3 + ablation run | M5 |
| **M4** — end-to-end RL training throughput | wire batched simulator into SB3 / PureJaxRL | most likely scope cut if M3 slips; see below |

### Plan and revised schedule

The original schedule had M2 closing 2026-05-10 and M4 closing today
(2026-05-22). The schedule slipped by ~11 days. The honest revised plan
to the 2026-06-02 presentation:

| Date | Work | Why this date |
|---|---|---|
| 2026-05-23/24 | GPU run; close M2 throughput + memory bars | GPU access lined up for the weekend |
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

See the [Linear project](https://linear.app/topograph-stanford/project/cs348k-topograph-604634a13675)
for full milestone descriptions and target dates.

* **M1** *(done)* — CPU baseline simulator + throughput floor + profiling.
* **M2** — Batched APSP feasibility study (decision gate for the rest of the project).
* **M3** — Full vectorized simulator on GPU.
* **M4** — End-to-end RL training throughput integration.
* **M5** — Throughput study and ablations.
* **M6** — Writeup and CS348K final presentation.
