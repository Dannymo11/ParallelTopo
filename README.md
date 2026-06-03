# TopoGraph

Batched GPU simulation for graph-structured reinforcement learning environments with **variable topology**.

Existing batched simulators (Madrona, Brax, Isaac Gym) assume fixed environment topology. TopoGraph asks whether the same playbook — GPU-resident state, per-environment masking, and `vmap`'d dynamics — extends to environments where each environment in the batch is *building a different graph* over the course of an episode.

The workload is a slim transit-network construction simulator on synthetic grid cities (10×10 to 15×15 zones, 15-step episode horizon). Each step adds one edge subject to a budget; reward is an accessibility-weighted welfare function computed over batched all-pairs shortest paths (APSP). Because APSP dominates the step cost, the core contribution is an **exact, fixed-iteration min-plus matrix-squaring APSP kernel** that vectorizes across environments with different active edge sets.

## Approach

The simulator keeps all per-environment state on the GPU and expresses one rollout step as a single compiled program. Variable topology is handled by an edge-existence mask updated per step, rather than by branching, so the whole batch runs through the same kernels. Shortest paths are computed with fixed-iteration min-plus matrix squaring (`D ← min(D, D ⊕ D)`), which converges to exact shortest paths in `⌈log₂ D⌉` steps for the graph diameters in this workload — no accuracy/throughput trade-off to tune. The entire episode (APSP + dynamics + policy) is fused into one `lax.scan` so XLA can fuse across steps and eliminate per-step launch overhead.

## Results

Detailed methodology, figures, and full benchmark tables are in the writeup: [`docs/writeup.pdf`](docs/writeup.pdf).

Headline numbers (Modal A100-40GB, 10×10 grid, horizon 15, against a SciPy Floyd-Warshall CPU baseline):

- **60× full-simulator speedup** at batch 256 (7,068 rollouts/sec vs. 117.8 on the CPU baseline).
- **Variable topology is effectively free** — ≤1.2% overhead (0.25% mean) vs. a frozen-topology baseline at matched batch and graph sizes.
- **APSP is the wall, and the GPU intensifies it** — the shortest-path share of step time rises from ~75% (CPU) to ~94% (GPU); peak memory tracks O(B·N²), so the cubic-memory wall never bites at these sizes.

## Repository layout

```
src/topograph/
  sim_cpu/      # NumPy/SciPy CPU baseline (reference semantics)
  sim_gpu/      # JAX batched simulator (APSP, dynamics, step, policies)
  bench/        # rollouts/sec + per-component profiling harness
  policies/     # random / greedy / scripted policies
tests/          # sanity + CPU/GPU numerical-equivalence tests
scripts/        # benchmark drivers + Modal runners for each experiment
results/        # benchmark outputs, profiling, figures
docs/           # writeup
transit_learning/   # generalization study: APSP kernel on an external transit codebase
```

## Setup

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev,figures]"
```

## Quickstart

```bash
# Run the test suite (sanity + CPU/GPU equivalence)
pytest

# CPU baseline throughput
python -m topograph.bench --grid 10 --horizon 15 --policy random --episodes 1000 --warmup 10

# Per-component time breakdown (CPU)
python -m topograph.bench.profiling --grid 10 --horizon 15 --episodes 100 --warmup 5
```

The GPU benchmarks run on a single CUDA device. Each experiment has a local driver in `scripts/` and a matching `modal_*.py` runner for reproducing the published numbers on a Modal A100:

```bash
modal run scripts/modal_m3_throughput.py        # full-simulator throughput vs. batch size
modal run scripts/modal_m5_fig2_graphsize.py    # throughput vs. graph size
modal run scripts/modal_m5_fig4_overhead.py     # variable-topology overhead
modal run scripts/modal_m4_train_throughput.py  # end-to-end training throughput
```

Artifacts (JSON/CSV/figures) are written under `results/`.

## Generalization study

To check that the APSP kernel isn't specific to these grid cities, `transit_learning/` applies it to an external transit-optimization codebase (Holliday & Dudek's `transit_learning`) on the real Mandl and Mumford city graphs. See [`transit_learning/README.md`](transit_learning/README.md) for the harness, tables, and reproduction.
