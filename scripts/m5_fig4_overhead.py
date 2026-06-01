"""M5 Figure 4: variable-topology overhead vs. a fixed-topology baseline.

The contribution-distinguishing experiment. Madrona / Brax / Isaac Gym
assume fixed environment topology; TopoGraph's claim is that *variable*
topology (each env building a different edge set during the episode) can be
handled with bounded overhead. This script quantifies that overhead.

**Method.** Advance the identical workload `HORIZON` steps twice, calling
`step_batched` in a **Python loop** (one dispatch per step), with:

  (a) production  — `freeze_mask=False`: `apply_action` runs each step, so
                    the edge mask + budget update (the variable-topology
                    machinery).
  (b) ablation    — `freeze_mask=True`: `apply_action` is skipped; the mask
                    stays fixed (the fixed-topology baseline).

Both run the full K=5 APSP + all dynamics every step. The min-plus APSP is
dense and its cost is independent of how many edges are active, so the
*only* compute difference is `apply_action`'s per-step mask scatter + budget
update. The reported number is the pure cost of supporting variable
topology:

    overhead % = (t_production - t_frozen) / t_frozen * 100

Expectation (per the M2/M3 findings): single-digit percent. The mask scatter
is one `.at[idx].set(True)` per env per step against a kernel that does
O(N^3) work; it should be in the noise.

**Why a Python loop and not `lax.scan` / `rollout_random`?** Under a fused
scan, a frozen mask is loop-invariant, so XLA hoists the APSP out of the
loop and computes it *once* instead of `HORIZON` times — which would make
the ablation ~10x faster and report a meaningless triple-digit "overhead".
That hoist is a real fixed-topology *benefit* but it is a different figure;
isolating the mask-scatter cost (the milestone's Figure 4) requires holding
the APSP work equal, which the per-step Python loop does (each step is its
own dispatch, so the APSP is recomputed every step in both modes). The
scan-fused rollout is what Figure 1 measures.

Note: the frozen baseline freezes at the *empty* network (reset state). The
overhead is independent of mask contents — the scatter cost is the same
whether the frozen mask is empty or pre-populated — so empty-frozen cleanly
isolates the masking machinery's cost.

Outputs:
  results/m5_fig4_overhead/m5_fig4_overhead_<ts>.json
  results/m5_fig4_overhead/m5_fig4_overhead_<ts>.csv

Run on GPU:  python scripts/m5_fig4_overhead.py
             (or via scripts/modal_m5_fig4_overhead.py)
"""

from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

try:
    import jax
    import jax.numpy as jnp
except ImportError as e:
    print("ERROR: JAX not installed. Install the gpu extras:")
    print('  uv pip install -e ".[gpu]"')
    print(f"\nUnderlying error: {e}")
    sys.exit(1)

from topograph.sim_cpu import GridCityConfig, make_world
from topograph.sim_gpu import (
    DEFAULT_K_ITERATIONS,
    reset_batched,
    step_batched,
    world_arrays_from_config,
)

HORIZON: int = 15
# Operating points: the Figure-1 sweet spot and its neighbors, on both M1
# grid sizes. Overhead is reported per (grid, batch) cell.
GRID_SHAPES: list[tuple[int, int]] = [(10, 10), (15, 15)]
BATCH_SIZES: list[int] = [64, 256, 1024]

WARMUP_ROLLOUTS: int = 2  # per mode — freeze_mask is static, so each compiles separately
TIMED_ROLLOUTS: int = 7   # median over this many; odd count for a clean median
SEED: int = 0

CPU_TEMP_BUDGET_GB: float = 8.0


def predicted_temp_gb(grid_shape: tuple[int, int], batch_size: int) -> float:
    n_zones = grid_shape[0] * grid_shape[1]
    return batch_size * (n_zones ** 3) * 4 / (1024 ** 3)


@dataclass(frozen=True)
class OverheadResult:
    grid_shape: tuple[int, int]
    n_zones: int
    batch_size: int
    median_production_s: float
    median_frozen_s: float
    production_rollouts_per_sec: float
    frozen_rollouts_per_sec: float
    overhead_pct: float
    production_edges_built: int
    frozen_edges_built: int

    def as_dict(self) -> dict:
        return {
            "grid_shape": list(self.grid_shape),
            "n_zones": self.n_zones,
            "batch_size": self.batch_size,
            "median_production_s": self.median_production_s,
            "median_frozen_s": self.median_frozen_s,
            "production_rollouts_per_sec": self.production_rollouts_per_sec,
            "frozen_rollouts_per_sec": self.frozen_rollouts_per_sec,
            "overhead_pct": self.overhead_pct,
            "production_edges_built": self.production_edges_built,
            "frozen_edges_built": self.frozen_edges_built,
        }


def _run_loop(state, actions, wa, freeze_mask: bool):
    """Advance HORIZON steps via a Python loop of step_batched (no scan).

    One dispatch per step keeps the APSP recompute in BOTH modes — see the
    module docstring on why scan would hoist the frozen APSP. Returns the
    final state (its last reward block_until_ready'd by the caller).
    """
    st = state
    reward = None
    for t in range(HORIZON):
        st, reward = step_batched(
            st, actions[t], wa, horizon=HORIZON, freeze_mask=freeze_mask
        )
    reward.block_until_ready()
    return st


def _time_mode(state, actions, wa, freeze_mask: bool) -> tuple[float, int]:
    """Median full-episode (Python-loop) time + edges active in the final state."""
    for _ in range(WARMUP_ROLLOUTS):
        _run_loop(state, actions, wa, freeze_mask)

    timings = []
    for _ in range(TIMED_ROLLOUTS):
        t0 = time.perf_counter()
        _run_loop(state, actions, wa, freeze_mask)
        timings.append(time.perf_counter() - t0)

    final = _run_loop(state, actions, wa, freeze_mask)
    edges_built = int(np.asarray(final.edge_mask).sum())
    return float(np.median(timings)), edges_built


def benchmark(grid_shape: tuple[int, int], batch_size: int) -> OverheadResult:
    world = make_world(SEED, GridCityConfig(grid_shape=grid_shape, horizon=HORIZON))
    wa = world_arrays_from_config(world)
    state = reset_batched(
        wa, batch_size,
        jnp.asarray(world.initial_activity, dtype=jnp.float32),
        world.initial_budget,
    )

    # Fixed action schedule (HORIZON, B): env e at step t tries a distinct
    # candidate edge, so production activates a spread of edges. Values that
    # are already-active/unaffordable degrade to no-ops at identical cost
    # (apply_action's scatter runs unconditionally via jnp.where), and the
    # APSP cost is independent of mask contents, so this schedule is a fair
    # stand-in for a policy for the purpose of timing the scatter overhead.
    n_candidate = world.n_candidate_edges
    t_idx = np.arange(HORIZON)[:, None]
    e_idx = np.arange(batch_size)[None, :]
    actions = jnp.asarray(((t_idx * batch_size + e_idx) % n_candidate).astype(np.int32))

    t_prod, prod_edges = _time_mode(state, actions, wa, freeze_mask=False)
    t_frozen, frozen_edges = _time_mode(state, actions, wa, freeze_mask=True)

    overhead_pct = (t_prod - t_frozen) / t_frozen * 100.0
    return OverheadResult(
        grid_shape=grid_shape,
        n_zones=int(world.n_zones),
        batch_size=batch_size,
        median_production_s=t_prod,
        median_frozen_s=t_frozen,
        production_rollouts_per_sec=batch_size / t_prod,
        frozen_rollouts_per_sec=batch_size / t_frozen,
        overhead_pct=overhead_pct,
        production_edges_built=prod_edges,
        frozen_edges_built=frozen_edges,
    )


def print_results(results: list[OverheadResult]) -> None:
    print(f"\n{'=' * 92}")
    print(f"  Variable-topology overhead  (K={DEFAULT_K_ITERATIONS}, fp32, "
          f"HORIZON={HORIZON}, random-legal policy)")
    print("  production = mask updated per step;  frozen = mask fixed at episode start")
    print("=" * 92)
    print(f"\n  {'grid':<8}{'N':>5}  {'batch':>7}  {'prod ms':>10}  "
          f"{'frozen ms':>10}  {'overhead':>10}  {'prod edges':>11}  {'froz edges':>11}")
    print(f"  {'-' * 86}")
    for r in results:
        print(
            f"  {str(r.grid_shape):<8}{r.n_zones:>5}  {r.batch_size:>7}  "
            f"{r.median_production_s * 1000:>10.2f}  "
            f"{r.median_frozen_s * 1000:>10.2f}  "
            f"{r.overhead_pct:>9.2f}%  "
            f"{r.production_edges_built:>11,}  {r.frozen_edges_built:>11,}"
        )


def main() -> None:
    devices = jax.devices()
    print("=" * 92)
    print("M5 Figure 4: variable-topology overhead vs. fixed-topology baseline")
    print("=" * 92)
    print(f"  JAX devices: {devices}")
    print(f"  Default backend: {jax.default_backend()}")
    if jax.default_backend() == "cpu":
        print("\n  WARNING: running on CPU. The overhead ratio is GPU-specific; CPU")
        print("  numbers validate the harness but are not the Figure-4 result.")

    is_cpu = jax.default_backend() == "cpu"
    results: list[OverheadResult] = []
    skipped = []
    for grid_shape in GRID_SHAPES:
        for batch_size in BATCH_SIZES:
            if is_cpu and predicted_temp_gb(grid_shape, batch_size) > CPU_TEMP_BUDGET_GB:
                skipped.append((grid_shape, batch_size))
                continue
            try:
                results.append(benchmark(grid_shape, batch_size))
            except Exception as e:  # noqa: BLE001
                print(f"\n  FAILED grid={grid_shape} batch={batch_size}: {e}")

    if skipped:
        print(f"\n  Skipped {len(skipped)} cell(s) over the {CPU_TEMP_BUDGET_GB:.1f} GB CPU budget: {skipped}")

    print_results(results)

    # Sanity: production must build edges; frozen must build none.
    for r in results:
        assert r.frozen_edges_built == 0, f"frozen run unexpectedly built edges at {r.grid_shape}/{r.batch_size}"
    if results and all(r.production_edges_built > 0 for r in results):
        print("\n  Sanity OK: production builds networks; frozen topology stays empty.")

    summary = {
        "mean_overhead_pct": float(np.mean([r.overhead_pct for r in results])) if results else None,
        "max_overhead_pct": float(np.max([r.overhead_pct for r in results])) if results else None,
    }
    print(f"\n  Overhead across all cells: "
          f"mean {summary['mean_overhead_pct']:.2f}%, max {summary['max_overhead_pct']:.2f}%"
          if results else "\n  No cells run.")

    out_dir = REPO_ROOT / "results" / "m5_fig4_overhead"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    json_path = out_dir / f"m5_fig4_overhead_{ts}.json"
    json_path.write_text(json.dumps({
        "config": {
            "grid_shapes": GRID_SHAPES,
            "batch_sizes": BATCH_SIZES,
            "horizon": HORIZON,
            "k_iterations": DEFAULT_K_ITERATIONS,
            "warmup_rollouts": WARMUP_ROLLOUTS,
            "timed_rollouts": TIMED_ROLLOUTS,
            "backend": jax.default_backend(),
            "devices": [str(d) for d in devices],
            "method": "production (freeze_mask=False) vs ablation (freeze_mask=True), matched workload",
        },
        "results": [r.as_dict() for r in results],
        "summary": summary,
    }, indent=2))

    csv_path = out_dir / f"m5_fig4_overhead_{ts}.csv"
    if results:
        fieldnames = list(results[0].as_dict().keys())
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                row = r.as_dict()
                row["grid_shape"] = f"{r.grid_shape[0]}x{r.grid_shape[1]}"
                writer.writerow(row)

    print(f"\n  Wrote: {json_path.relative_to(REPO_ROOT)}")
    if results:
        print(f"  Wrote: {csv_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
