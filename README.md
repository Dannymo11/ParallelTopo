# TopoGraph

Batched GPU simulation for graph-structured reinforcement learning environments
with **variable topology**. CS348K final project, Spring 2026.

## What this is

A systems study, not an RL paper. Existing batched simulators (Madrona, Brax,
Isaac Gym) assume fixed environment topology. This project asks whether the
same playbook — GPU-resident state, per-environment masking, vmap'd dynamics —
extends to environments where each environment in the batch is *building a
different graph* over the course of an episode.

Concrete workload: a slim transit-network construction simulator on synthetic
grid cities (10×10 to 15×15 zones, 15-step episode horizon). Each step adds
one edge subject to a budget; reward is an accessibility-weighted welfare
function computed over batched approximate all-pairs shortest paths.

## Layout

```
src/topograph/
  sim_cpu/      # NumPy/SciPy CPU baseline (M1)
  sim_gpu/      # JAX batched simulator    (M3)
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
uv pip install -e ".[dev]"
```

## Status

Pre-M1. See the project plan in Linear for milestone targets and success gates.
