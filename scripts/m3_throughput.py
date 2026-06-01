"""M3 throughput study: FULL on-device simulator rollouts/sec.

The M5 Figure 1 headline. Where `scripts/m2_apsp_throughput.py` timed the
APSP kernel in isolation (the M2 feasibility gate), this script times the
*entire* simulator: `rollout_random` advances a batch of environments for a
full `HORIZON`-step episode on-device — action sampling, edge-mask scatter,
direct-distance construction, K=5 APSP, accessibility (x2), land-use
update, and welfare — with nothing transferred to the host mid-rollout.

A single `rollout_random` call produces B full episodes, so

    rollouts/sec = B / wall_time_of_one_rollout_call.

This is directly comparable to the scipy CPU baseline (117.8 rollouts/sec
on the 10x10/horizon-15 reference workload) — a true full-simulator vs
full-simulator number, the honest speedup for the writeup.

The throughput gate is the same one M2 used and is defined on the 10x10
reference workload only (see REFERENCE_GRID_SHAPE); 15x15 is reported for
the graph-size scaling story and the memory read.

Outputs:
  results/m3_throughput/m3_throughput_<ts>.json
  results/m3_throughput/m3_throughput_<ts>.csv

Run on GPU:  python scripts/m3_throughput.py   (or via scripts/modal_m3_throughput.py)
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

# Workload — matches the canonical M1 reference workload.
HORIZON: int = 15
GRID_SHAPES: list[tuple[int, int]] = [(10, 10), (15, 15)]
BATCH_SIZES: list[int] = [1, 16, 64, 256, 1024]

# Gate constants (identical to the M2 throughput study).
SCIPY_CPU_BASELINE_ROLLOUTS_PER_SEC: float = 117.8
THROUGHPUT_GATE_FACTOR: float = 10.0
THROUGHPUT_GATE_AT_BATCH_256: float = (
    SCIPY_CPU_BASELINE_ROLLOUTS_PER_SEC * THROUGHPUT_GATE_FACTOR  # = 1178
)
# Throughput is gated on the reference workload only; 15x15 is informational
# (a 15x15 CPU rollout is ~3-6 rollouts/sec, so comparing it to the 10x10
# baseline understates it). See m2_apsp_throughput.py for the full rationale.
REFERENCE_GRID_SHAPE: tuple[int, int] = (10, 10)

WARMUP_ROLLOUTS: int = 2  # first call JIT-compiles for this (grid, batch) shape
TIMED_ROLLOUTS: int = 5   # median over this many
SEED: int = 0

# On CPU (laptop, not the verdict) the (B, N, N, N) APSP intermediate still
# dominates allocation; skip cells whose naive bound exceeds this to avoid
# OOMing the host. Disabled on GPU.
CPU_TEMP_BUDGET_GB: float = 8.0


def predicted_temp_gb(grid_shape: tuple[int, int], batch_size: int) -> float:
    """(B, N, N, N) fp32 tensor size in GB — the dominant naive memory term.

    NB: the M2 GPU run showed XLA fuses this away (actual peak is O(B*N^2)),
    so this is a conservative upper bound used only to gate CPU-host runs.
    """
    n_zones = grid_shape[0] * grid_shape[1]
    return batch_size * (n_zones ** 3) * 4 / (1024 ** 3)


@dataclass(frozen=True)
class BenchResult:
    grid_shape: tuple[int, int]
    n_zones: int
    batch_size: int
    k_iterations: int
    median_rollout_s: float
    rollouts_per_sec: float
    speedup_vs_scipy_baseline: float
    peak_device_mb: float | None

    def as_dict(self) -> dict:
        return {
            "grid_shape": list(self.grid_shape),
            "n_zones": self.n_zones,
            "batch_size": self.batch_size,
            "k_iterations": self.k_iterations,
            "median_rollout_s": self.median_rollout_s,
            "rollouts_per_sec": self.rollouts_per_sec,
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
    """Wall time of a single full B-env episode rollout on-device."""
    t0 = time.perf_counter()
    _final, rewards, _actions = rollout_random(
        state, key, world, horizon=HORIZON, k_iterations=k_iterations
    )
    # Force the dispatch queue to flush so we time GPU compute, not queuing.
    rewards.block_until_ready()
    return time.perf_counter() - t0


def benchmark(
    grid_shape: tuple[int, int],
    batch_size: int,
    k_iterations: int = DEFAULT_K_ITERATIONS,
) -> BenchResult:
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
        time_one_rollout(state, key, wa, k_iterations)

    timings = [time_one_rollout(state, key, wa, k_iterations) for _ in range(TIMED_ROLLOUTS)]
    median_t = float(np.median(timings))
    rollouts_per_sec = batch_size / median_t

    return BenchResult(
        grid_shape=grid_shape,
        n_zones=int(world.n_zones),
        batch_size=batch_size,
        k_iterations=k_iterations,
        median_rollout_s=median_t,
        rollouts_per_sec=rollouts_per_sec,
        speedup_vs_scipy_baseline=rollouts_per_sec / SCIPY_CPU_BASELINE_ROLLOUTS_PER_SEC,
        peak_device_mb=read_peak_memory_mb(device),
    )


def print_results(results: list[BenchResult]) -> None:
    print(f"\n{'=' * 90}")
    print(f"  FULL simulator throughput  (K={DEFAULT_K_ITERATIONS}, fp32, "
          f"HORIZON={HORIZON} steps per rollout, random-legal policy)")
    print(f"  Scipy CPU baseline: {SCIPY_CPU_BASELINE_ROLLOUTS_PER_SEC:.1f} rollouts/sec")
    print(f"  Throughput gate at batch 256: >= {THROUGHPUT_GATE_AT_BATCH_256:.0f} rollouts/sec")
    print("=" * 90)
    print(f"\n  {'grid':<8}{'N':>5}  {'batch':>7}  {'ms/roll':>10}  "
          f"{'rollouts/s':>14}  {'speedup':>10}  {'peak MB':>10}")
    print(f"  {'-' * 84}")
    for r in results:
        peak_str = f"{r.peak_device_mb:.0f}" if r.peak_device_mb is not None else "  --  "
        marker = ""
        if r.batch_size == 256 and r.grid_shape == REFERENCE_GRID_SHAPE:
            marker = " *PASS" if r.rollouts_per_sec >= THROUGHPUT_GATE_AT_BATCH_256 else " FAIL"
        print(
            f"  {str(r.grid_shape):<8}{r.n_zones:>5}  {r.batch_size:>7}  "
            f"{r.median_rollout_s * 1000:>10.2f}  "
            f"{r.rollouts_per_sec:>14,.1f}  "
            f"{r.speedup_vs_scipy_baseline:>9.2f}x  "
            f"{peak_str:>10}{marker}"
        )


def m3_verdict(results: list[BenchResult]) -> dict:
    per_grid = {}
    for grid_shape in {r.grid_shape for r in results}:
        gate_result = next(
            (r for r in results if r.grid_shape == grid_shape and r.batch_size == 256),
            None,
        )
        if gate_result is None:
            continue
        is_reference = grid_shape == REFERENCE_GRID_SHAPE
        per_grid[str(grid_shape)] = {
            "rollouts_per_sec_at_batch_256": gate_result.rollouts_per_sec,
            "speedup_vs_scipy": gate_result.speedup_vs_scipy_baseline,
            "is_reference_workload": is_reference,
            "throughput_gate_applies": is_reference,
            "throughput_gate_passes": (
                gate_result.rollouts_per_sec >= THROUGHPUT_GATE_AT_BATCH_256
                if is_reference
                else None
            ),
            "peak_device_mb_at_batch_256": gate_result.peak_device_mb,
        }
    return per_grid


def main() -> None:
    devices = jax.devices()
    print("=" * 90)
    print("M3 throughput study: FULL on-device simulator rollouts/sec")
    print("=" * 90)
    print(f"  JAX devices: {devices}")
    print(f"  Default backend: {jax.default_backend()}")
    if jax.default_backend() == "cpu":
        print("\n  WARNING: running on CPU. Fine for correctness, but the throughput")
        print("  gate is defined against a GPU. Numbers below are not the verdict.")
    print(f"  Grid shapes:  {GRID_SHAPES}")
    print(f"  Batch sizes:  {BATCH_SIZES}")
    print(f"  K iterations: {DEFAULT_K_ITERATIONS}")

    is_cpu = jax.default_backend() == "cpu"
    results: list[BenchResult] = []
    skipped: list[tuple[tuple[int, int], int, float]] = []
    # Ascending batch within a fixed grid amortizes JIT recompiles.
    for grid_shape in GRID_SHAPES:
        for batch_size in BATCH_SIZES:
            temp_gb = predicted_temp_gb(grid_shape, batch_size)
            if is_cpu and temp_gb > CPU_TEMP_BUDGET_GB:
                skipped.append((grid_shape, batch_size, temp_gb))
                continue
            try:
                results.append(benchmark(grid_shape, batch_size))
            except Exception as e:  # noqa: BLE001 — partial data is still useful
                print(f"\n  FAILED grid={grid_shape} batch={batch_size}: {e}")

    if skipped:
        print(f"\n  Skipped {len(skipped)} cell(s) over the {CPU_TEMP_BUDGET_GB:.1f} GB CPU budget:")
        for grid_shape, batch_size, temp_gb in skipped:
            print(f"    grid={grid_shape} batch={batch_size}  ~{temp_gb:.1f} GB naive temp")

    print_results(results)
    verdict = m3_verdict(results)

    print(f"\n{'=' * 90}")
    print("  M3 full-simulator throughput verdict")
    print("=" * 90)
    for grid_str, v in verdict.items():
        print(f"\n  Grid {grid_str}:")
        if v["throughput_gate_applies"]:
            thr = "PASS" if v["throughput_gate_passes"] else "FAIL"
            print(f"    Throughput @ batch 256: {v['rollouts_per_sec_at_batch_256']:,.1f} "
                  f"rollouts/sec ({v['speedup_vs_scipy']:.2f}x scipy)   [{thr}]  "
                  "<- M3 throughput gate (reference workload)")
        else:
            print(f"    Throughput @ batch 256: {v['rollouts_per_sec_at_batch_256']:,.1f} "
                  f"rollouts/sec ({v['speedup_vs_scipy']:.2f}x the 10x10 CPU baseline, "
                  "informational)   [not gated]")
        peak = v["peak_device_mb_at_batch_256"]
        print(f"    Peak device memory @ batch 256: "
              f"{f'{peak:.0f} MB' if peak is not None else 'unknown'}")

    out_dir = REPO_ROOT / "results" / "m3_throughput"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    json_path = out_dir / f"m3_throughput_{ts}.json"
    json_path.write_text(json.dumps({
        "config": {
            "grid_shapes": GRID_SHAPES,
            "batch_sizes": BATCH_SIZES,
            "horizon": HORIZON,
            "k_iterations": DEFAULT_K_ITERATIONS,
            "policy": "uniform-random-legal (rollout_random)",
            "measures": "full on-device simulator step (all dynamics), not APSP-only",
            "warmup_rollouts": WARMUP_ROLLOUTS,
            "timed_rollouts": TIMED_ROLLOUTS,
            "scipy_cpu_baseline_rollouts_per_sec": SCIPY_CPU_BASELINE_ROLLOUTS_PER_SEC,
            "throughput_gate_at_batch_256": THROUGHPUT_GATE_AT_BATCH_256,
            "reference_grid_shape": list(REFERENCE_GRID_SHAPE),
            "backend": jax.default_backend(),
            "devices": [str(d) for d in devices],
        },
        "results": [r.as_dict() for r in results],
        "verdict": verdict,
    }, indent=2))

    csv_path = out_dir / f"m3_throughput_{ts}.csv"
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
