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
