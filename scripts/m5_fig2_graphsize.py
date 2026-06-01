"""M5 Figure 2: full-simulator throughput vs. graph size, at fixed batch.

Fixes the batch at the Figure-1 sweet spot and sweeps the number of zones N
to show how throughput scales with graph size and where it breaks down.
Per the milestone, *finding the ceiling* is the successful outcome of this
figure, not avoiding it.

**Sweep.** Grids {5x5, 10x10, 15x15, 20x20, 25x25, 30x30} -> N in
{25, 100, 225, 400, 625, 900}, at batch {256, 1024}. The first four are the
M1-spec range; 25x25 and 30x30 push past it to locate the actual ceiling.

**What to expect (and the finding).** The originally-feared ceiling was
memory: the naive (B, N, N, N) APSP intermediate is ~65 GB at N=400/B=256
and grows as N^3. But the M2/M3 runs showed XLA fuses that term away, so
peak memory scales as O(B * N^2), not O(B * N^3) — at N=400/B=256 that is
only ~1-2 GB. So memory does *not* bite in the M1 range; the real ceiling is
**compute** (K=5 min-plus matrix-squaring is O(N^3) per step). Throughput is
therefore expected to fall ~O(N^3) per environment while peak memory grows
only ~O(N^2). Documenting that divergence — and the N/batch at which a 40 GB
card finally OOMs (if it does within the sweep) — is the figure.

**Methodology notes.**
* Grids are swept in ASCENDING N (and ascending batch within a grid) so the
  cumulative peak-memory high-water mark reported per row approximates that
  row's own footprint (the device peak counter is not reset between cells;
  see m2_apsp_throughput.py).
* Throughput is the full on-device simulator via `rollout_random` (same as
  the M3 / Figure-1 measure), not APSP-only.
* A cell that OOMs is caught and recorded as a gap — that gap IS the ceiling.

Outputs:
  results/m5_fig2_graphsize/m5_fig2_graphsize_<ts>.json
  results/m5_fig2_graphsize/m5_fig2_graphsize_<ts>.csv

Run on GPU:  python scripts/m5_fig2_graphsize.py
             (or via scripts/modal_m5_fig2_graphsize.py; use A100-80GB for
              the largest cells)
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
    rollout_random,
    world_arrays_from_config,
)

HORIZON: int = 15
# Ascending N so the cumulative peak-memory reading approximates per-row.
GRID_SHAPES: list[tuple[int, int]] = [
    (5, 5), (10, 10), (15, 15), (20, 20), (25, 25), (30, 30)
]
BATCH_SIZES: list[int] = [256, 1024]

SCIPY_CPU_BASELINE_ROLLOUTS_PER_SEC: float = 117.8  # 10x10 reference (informational)
WARMUP_ROLLOUTS: int = 2
TIMED_ROLLOUTS: int = 5
SEED: int = 0

CPU_TEMP_BUDGET_GB: float = 8.0


def predicted_temp_gb(grid_shape: tuple[int, int], batch_size: int) -> float:
    n_zones = grid_shape[0] * grid_shape[1]
    return batch_size * (n_zones ** 3) * 4 / (1024 ** 3)


@dataclass(frozen=True)
class GraphSizeResult:
    grid_shape: tuple[int, int]
    n_zones: int
    batch_size: int
    median_rollout_s: float
    rollouts_per_sec: float
    env_steps_per_sec: float
    speedup_vs_scipy_baseline: float
    peak_device_mb: float | None

    def as_dict(self) -> dict:
        return {
            "grid_shape": list(self.grid_shape),
            "n_zones": self.n_zones,
            "batch_size": self.batch_size,
            "median_rollout_s": self.median_rollout_s,
            "rollouts_per_sec": self.rollouts_per_sec,
            "env_steps_per_sec": self.env_steps_per_sec,
            "speedup_vs_scipy_baseline": self.speedup_vs_scipy_baseline,
            "peak_device_mb": self.peak_device_mb,
        }


def read_peak_memory_mb(device: jax.Device) -> float | None:
    stats_fn = getattr(device, "memory_stats", None)
    if stats_fn is None:
        return None
    try:
        stats = stats_fn()
    except Exception:
        return None
    if not stats:
        return None
    peak = stats.get("peak_bytes_in_use") or stats.get("bytes_in_use") or 0
    return peak / (1024 ** 2)


def time_one_rollout(state, key, world, k_iterations: int) -> float:
    t0 = time.perf_counter()
    _f, rewards, _a = rollout_random(state, key, world, horizon=HORIZON, k_iterations=k_iterations)
    rewards.block_until_ready()
    return time.perf_counter() - t0


def benchmark(grid_shape: tuple[int, int], batch_size: int) -> GraphSizeResult:
    world = make_world(SEED, GridCityConfig(grid_shape=grid_shape, horizon=HORIZON))
    wa = world_arrays_from_config(world)
    state = reset_batched(
        wa, batch_size,
        jnp.asarray(world.initial_activity, dtype=jnp.float32),
        world.initial_budget,
    )
    key = jax.random.PRNGKey(SEED)
    device = jax.devices()[0]

    for _ in range(WARMUP_ROLLOUTS):
        time_one_rollout(state, key, wa, DEFAULT_K_ITERATIONS)

    timings = [time_one_rollout(state, key, wa, DEFAULT_K_ITERATIONS) for _ in range(TIMED_ROLLOUTS)]
    median_t = float(np.median(timings))
    rps = batch_size / median_t

    return GraphSizeResult(
        grid_shape=grid_shape,
        n_zones=int(world.n_zones),
        batch_size=batch_size,
        median_rollout_s=median_t,
        rollouts_per_sec=rps,
        env_steps_per_sec=rps * HORIZON,
        speedup_vs_scipy_baseline=rps / SCIPY_CPU_BASELINE_ROLLOUTS_PER_SEC,
        peak_device_mb=read_peak_memory_mb(device),
    )


def print_results(results: list[GraphSizeResult]) -> None:
    print(f"\n{'=' * 96}")
    print(f"  Full-simulator throughput vs. graph size  (K={DEFAULT_K_ITERATIONS}, "
          f"fp32, HORIZON={HORIZON}, random-legal policy)")
    print("=" * 96)
    print(f"\n  {'grid':<8}{'N':>6}  {'batch':>7}  {'ms/roll':>10}  "
          f"{'rollouts/s':>14}  {'env-steps/s':>15}  {'peak MB':>10}")
    print(f"  {'-' * 90}")
    for r in results:
        peak_str = f"{r.peak_device_mb:.0f}" if r.peak_device_mb is not None else "  --  "
        print(
            f"  {str(r.grid_shape):<8}{r.n_zones:>6}  {r.batch_size:>7}  "
            f"{r.median_rollout_s * 1000:>10.2f}  "
            f"{r.rollouts_per_sec:>14,.1f}  "
            f"{r.env_steps_per_sec:>15,.0f}  "
            f"{peak_str:>10}"
        )


def main() -> None:
    devices = jax.devices()
    print("=" * 96)
    print("M5 Figure 2: throughput vs. graph size at fixed batch")
    print("=" * 96)
    print(f"  JAX devices: {devices}")
    print(f"  Default backend: {jax.default_backend()}")
    if jax.default_backend() == "cpu":
        print("\n  WARNING: running on CPU. Validates the harness; the scaling curve")
        print("  and ceiling are GPU-specific.")
    print(f"  Grid shapes:  {GRID_SHAPES}")
    print(f"  Batch sizes:  {BATCH_SIZES}")

    is_cpu = jax.default_backend() == "cpu"
    results: list[GraphSizeResult] = []
    skipped, oomed = [], []
    # Batch outer, grid (N) inner ascending — keeps each batch's curve
    # contiguous and N monotonic for the memory high-water reading.
    for batch_size in BATCH_SIZES:
        for grid_shape in GRID_SHAPES:
            if is_cpu and predicted_temp_gb(grid_shape, batch_size) > CPU_TEMP_BUDGET_GB:
                skipped.append((grid_shape, batch_size))
                continue
            try:
                results.append(benchmark(grid_shape, batch_size))
            except Exception as e:  # noqa: BLE001 — an OOM here is the ceiling, not a bug
                oomed.append((grid_shape, batch_size, str(e).splitlines()[0][:80]))
                print(f"\n  CEILING grid={grid_shape} batch={batch_size}: {str(e).splitlines()[0][:80]}")

    if skipped:
        print(f"\n  Skipped {len(skipped)} cell(s) over the {CPU_TEMP_BUDGET_GB:.1f} GB CPU budget: {skipped}")

    print_results(results)

    if oomed:
        print(f"\n  Ceiling located — {len(oomed)} cell(s) failed (likely OOM):")
        for grid_shape, batch_size, msg in oomed:
            print(f"    grid={grid_shape} batch={batch_size}: {msg}")

    out_dir = REPO_ROOT / "results" / "m5_fig2_graphsize"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    json_path = out_dir / f"m5_fig2_graphsize_{ts}.json"
    json_path.write_text(json.dumps({
        "config": {
            "grid_shapes": GRID_SHAPES,
            "batch_sizes": BATCH_SIZES,
            "horizon": HORIZON,
            "k_iterations": DEFAULT_K_ITERATIONS,
            "policy": "uniform-random-legal (rollout_random)",
            "scipy_cpu_baseline_rollouts_per_sec": SCIPY_CPU_BASELINE_ROLLOUTS_PER_SEC,
            "backend": jax.default_backend(),
            "devices": [str(d) for d in devices],
        },
        "results": [r.as_dict() for r in results],
        "ceiling_failures": [
            {"grid_shape": list(g), "batch_size": b, "error": m} for g, b, m in oomed
        ],
    }, indent=2))

    csv_path = out_dir / f"m5_fig2_graphsize_{ts}.csv"
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
