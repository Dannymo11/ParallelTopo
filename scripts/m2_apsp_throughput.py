"""M2 throughput study: batched APSP rollouts/sec on GPU vs scipy CPU baseline.

Second half of the M2 decision gate (the accuracy half was settled by
`scripts/batched_apsp_accuracy.py` on 2026-05-22). This script answers
the throughput and memory bars:

  THROUGHPUT GATE   simulator-only rollouts/sec at batch 256 on
                    10x10/horizon-15 reference workload
                    target:  >=1,150  (=10x scipy baseline of 117.8)

  MEMORY GATE       peak device memory at batch 256 on 15x15
                    target:  fits comfortably on a 24 GB GPU

**Methodology.** A CPU rollout does 15 sequential APSPs (one per simulator
step). To compare like-for-like, we run 15 sequential `apsp_batched` calls
on a batch of B environments — same total APSP work as B sequential CPU
rollouts. Wall time is divided by B to get rollouts/sec.

We don't time direct-distance construction or host-device transfer here —
those are M3 concerns (in M3 everything lives on the device, so transfer
is amortized to zero). M2's job is to confirm the APSP itself is fast
enough; if it isn't, no amount of M3 work fixes that.

**Outputs.**
  results/m2_apsp_throughput/m2_throughput_<ts>.json
  results/m2_apsp_throughput/m2_throughput_<ts>.csv

The script prints a clear pass/fail verdict at the end.
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
    print("  (and the appropriate jax CUDA wheel for your hardware)")
    print(f"\nUnderlying error: {e}")
    sys.exit(1)

from topograph.sim_cpu import GridCityConfig, make_world
from topograph.sim_cpu.dynamics import _build_direct_distance_matrix
from topograph.sim_gpu import DEFAULT_K_ITERATIONS, apsp_batched


# Workload parameters: matches the canonical M1 reference workload.
HORIZON: int = 15  # steps per "rollout"
GRID_SHAPES: list[tuple[int, int]] = [(10, 10), (15, 15)]
BATCH_SIZES: list[int] = [1, 16, 64, 256, 1024]

# M2 gate constants.
SCIPY_CPU_BASELINE_ROLLOUTS_PER_SEC: float = 117.8  # M1 reference
THROUGHPUT_GATE_FACTOR: float = 10.0
THROUGHPUT_GATE_AT_BATCH_256: float = (
    SCIPY_CPU_BASELINE_ROLLOUTS_PER_SEC * THROUGHPUT_GATE_FACTOR  # = 1178
)

# The throughput gate is defined on the canonical M1 reference workload
# (10x10 / horizon 15) — that is where the 117.8 rollouts/sec scipy baseline
# was measured. Larger grids (15x15) are run for the MEMORY gate and for the
# graph-size scaling story (M5 Fig 2). Applying the 10x10-relative threshold
# to 15x15 compares GPU throughput against the *wrong* CPU baseline (a 15x15
# CPU rollout is ~3-6 rollouts/sec, not 117.8), so the verdict only gates
# throughput on the reference grid. 15x15 reports memory + an informational
# speedup-vs-10x10-baseline figure, never a throughput pass/fail.
REFERENCE_GRID_SHAPE: tuple[int, int] = (10, 10)

# Bench loop hygiene.
WARMUP_EPISODES: int = 2  # first call triggers JIT compile; never time it
TIMED_EPISODES: int = 5   # take median over this many for stability
SEED: int = 0

# When running on CPU (= laptop, not the M2 verdict), cap predicted temp
# tensor size to avoid OOMing the host. The (B, N, N, N) min-plus matmul
# intermediate is the dominant allocation; at B=256 / N=225 it's already
# 11.7 GB and at B=1024 / N=225 it's 46 GB. On GPU we have VRAM headroom;
# on a laptop we don't. This bound is intentionally generous — adjust if
# your laptop has less than 8 GB free RAM.
CPU_TEMP_BUDGET_GB: float = 8.0


def predicted_temp_gb(grid_shape: tuple[int, int], batch_size: int) -> float:
    """(B, N, N, N) fp32 tensor size in GB. Dominant memory cost of the kernel."""
    n_zones = grid_shape[0] * grid_shape[1]
    bytes_ = batch_size * (n_zones ** 3) * 4  # fp32 = 4 bytes
    return bytes_ / (1024 ** 3)


# ---------------------------------------------------------------------------
# Building the batched input
# ---------------------------------------------------------------------------


def build_direct_distance_batch(
    grid_shape: tuple[int, int],
    batch_size: int,
    seed: int = SEED,
) -> np.ndarray:
    """Construct a (B, N, N) stack of direct-distance matrices, one per env.

    Each env gets a random mid-episode edge mask at a random density in
    [0.1, 0.8] — covers the realistic regime of a partially-built network
    mid-rollout. World geometry is shared across envs (single `make_world`)
    so we're only varying the edge mask; this matches what the GPU
    simulator will look like at runtime (one world config, B masks).
    """
    rng = np.random.default_rng(seed)
    world = make_world(0, GridCityConfig(grid_shape=grid_shape))
    n_candidate = world.n_candidate_edges

    matrices: list[np.ndarray] = []
    for _ in range(batch_size):
        density = float(rng.uniform(0.1, 0.8))
        mask = np.zeros(n_candidate, dtype=np.bool_)
        k = max(1, int(round(density * n_candidate)))
        mask[rng.choice(n_candidate, size=k, replace=False)] = True
        matrices.append(_build_direct_distance_matrix(world, mask).astype(np.float32))

    return np.stack(matrices, axis=0)


# ---------------------------------------------------------------------------
# Bench loop
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchResult:
    grid_shape: tuple[int, int]
    n_zones: int
    batch_size: int
    k_iterations: int
    median_episode_s: float
    rollouts_per_sec: float
    speedup_vs_scipy_baseline: float
    peak_device_mb: float | None

    def as_dict(self) -> dict:
        return {
            "grid_shape": list(self.grid_shape),
            "n_zones": self.n_zones,
            "batch_size": self.batch_size,
            "k_iterations": self.k_iterations,
            "median_episode_s": self.median_episode_s,
            "rollouts_per_sec": self.rollouts_per_sec,
            "speedup_vs_scipy_baseline": self.speedup_vs_scipy_baseline,
            "peak_device_mb": self.peak_device_mb,
        }


def reset_peak_memory(device: jax.Device) -> None:
    """Best-effort reset of the device's peak-memory counter."""
    stats = getattr(device, "memory_stats", None)
    if stats is None:
        return
    # Newer JAX exposes a counter-reset method; older versions don't. The
    # bench reports peak between resets, so we just record the baseline.
    try:
        device.memory_stats()  # touch to ensure the counter exists
    except Exception:
        pass


def read_peak_memory_mb(device: jax.Device) -> float | None:
    """Return device peak memory in MB, or None if unavailable.

    Several reasons this can be None: device has no `memory_stats` method
    (very old JAX); CPU backend (returns None instead of raising on some
    versions); call raises (returns None on others). Memory tracking on
    CPU is not meaningful anyway — the M2 memory gate is GPU-only.
    """
    stats_fn = getattr(device, "memory_stats", None)
    if stats_fn is None:
        return None
    try:
        stats = stats_fn()
    except Exception:
        return None
    if not stats:  # None or empty dict
        return None
    peak = stats.get("peak_bytes_in_use") or stats.get("bytes_in_use") or 0
    return peak / (1024 ** 2)


def time_one_episode(A_batch_device: jax.Array, k_iterations: int) -> float:
    """Run HORIZON sequential batched APSPs on the device, return wall time.

    The 15 calls model a 15-step rollout where each step recomputes APSP
    on the current edge-mask state. We re-use the same direct-distance
    matrix for all 15 calls — the cost we're measuring is the APSP itself,
    not the matrix-construction step (which is M3's responsibility to
    bring on-device).
    """
    t0 = time.perf_counter()
    D = A_batch_device
    for _ in range(HORIZON):
        D = apsp_batched(D, k_iterations=k_iterations)
    # block_until_ready forces the dispatch queue to flush; without this
    # we'd measure host-side queuing time, not GPU compute time.
    D.block_until_ready()
    return time.perf_counter() - t0


def benchmark(
    grid_shape: tuple[int, int],
    batch_size: int,
    k_iterations: int = DEFAULT_K_ITERATIONS,
) -> BenchResult:
    A_batch = build_direct_distance_batch(grid_shape, batch_size)
    A_batch_device = jnp.asarray(A_batch)
    device = jax.devices()[0]

    # Warmup: first call triggers JIT compile; subsequent calls are the
    # real measurements. We do >=2 warmups so JIT is fully out of the way.
    for _ in range(WARMUP_EPISODES):
        time_one_episode(A_batch_device, k_iterations)

    reset_peak_memory(device)

    timings = [time_one_episode(A_batch_device, k_iterations) for _ in range(TIMED_EPISODES)]
    median_t = float(np.median(timings))
    rollouts_per_sec = batch_size / median_t
    peak_mb = read_peak_memory_mb(device)

    return BenchResult(
        grid_shape=grid_shape,
        n_zones=int(A_batch.shape[1]),
        batch_size=batch_size,
        k_iterations=k_iterations,
        median_episode_s=median_t,
        rollouts_per_sec=rollouts_per_sec,
        speedup_vs_scipy_baseline=rollouts_per_sec / SCIPY_CPU_BASELINE_ROLLOUTS_PER_SEC,
        peak_device_mb=peak_mb,
    )


# ---------------------------------------------------------------------------
# Reporting + verdict
# ---------------------------------------------------------------------------


def print_results(results: list[BenchResult]) -> None:
    print(f"\n{'=' * 90}")
    print(f"  Batched APSP throughput  (K={DEFAULT_K_ITERATIONS}, fp32, "
          f"HORIZON={HORIZON} APSPs per rollout)")
    print(f"  Scipy CPU baseline: {SCIPY_CPU_BASELINE_ROLLOUTS_PER_SEC:.1f} rollouts/sec")
    print(f"  Throughput gate at batch 256: >= {THROUGHPUT_GATE_AT_BATCH_256:.0f} rollouts/sec")
    print("=" * 90)
    print(f"\n  {'grid':<8}{'N':>5}  {'batch':>7}  {'ms/ep':>10}  "
          f"{'rollouts/s':>14}  {'speedup':>10}  {'peak MB':>10}")
    print(f"  {'-' * 84}")
    for r in results:
        peak_str = f"{r.peak_device_mb:.0f}" if r.peak_device_mb is not None else "  --  "
        marker = ""
        # The throughput pass/fail marker only applies to the reference grid;
        # 15x15 at batch 256 is a memory-gate / scaling row, not a throughput
        # gate, so it gets no PASS/FAIL flag here.
        if r.batch_size == 256 and r.grid_shape == REFERENCE_GRID_SHAPE:
            marker = " *PASS" if r.rollouts_per_sec >= THROUGHPUT_GATE_AT_BATCH_256 else " FAIL"
        print(
            f"  {str(r.grid_shape):<8}{r.n_zones:>5}  {r.batch_size:>7}  "
            f"{r.median_episode_s * 1000:>10.2f}  "
            f"{r.rollouts_per_sec:>14,.1f}  "
            f"{r.speedup_vs_scipy_baseline:>9.2f}x  "
            f"{peak_str:>10}{marker}"
        )


def m2_verdict(results: list[BenchResult]) -> dict:
    per_grid = {}
    for grid_shape in {r.grid_shape for r in results}:
        grid_results = [r for r in results if r.grid_shape == grid_shape]
        gate_result = next((r for r in grid_results if r.batch_size == 256), None)
        if gate_result is None:
            continue
        is_reference = grid_shape == REFERENCE_GRID_SHAPE
        per_grid[str(grid_shape)] = {
            "rollouts_per_sec_at_batch_256": gate_result.rollouts_per_sec,
            "speedup_vs_scipy": gate_result.speedup_vs_scipy_baseline,
            "is_reference_workload": is_reference,
            # Throughput is only gated on the reference grid (see
            # REFERENCE_GRID_SHAPE). For non-reference grids this is None:
            # the row exists for the memory gate / graph-size scaling, and the
            # 10x10-relative speedup above is informational only.
            "throughput_gate_applies": is_reference,
            "throughput_gate_passes": (
                gate_result.rollouts_per_sec >= THROUGHPUT_GATE_AT_BATCH_256
                if is_reference
                else None
            ),
            "peak_device_mb_at_batch_256": gate_result.peak_device_mb,
            "memory_gate_passes": (
                gate_result.peak_device_mb is not None
                and gate_result.peak_device_mb < 24 * 1024  # 24 GB
            ),
        }
    return per_grid


def main() -> None:
    # Print device info upfront so the result is self-documenting.
    devices = jax.devices()
    print("=" * 90)
    print("M2 throughput study: batched APSP rollouts/sec")
    print("=" * 90)
    print(f"  JAX devices: {devices}")
    print(f"  Default backend: {jax.default_backend()}")
    if jax.default_backend() == "cpu":
        print("\n  WARNING: running on CPU. This is fine for correctness but the throughput")
        print("  gate is defined against a GPU. Numbers below are not the M2 verdict.")
    print(f"  Grid shapes:  {GRID_SHAPES}")
    print(f"  Batch sizes:  {BATCH_SIZES}")
    print(f"  K iterations: {DEFAULT_K_ITERATIONS}")

    is_cpu = jax.default_backend() == "cpu"
    if is_cpu:
        print(f"  CPU temp-tensor budget: {CPU_TEMP_BUDGET_GB:.1f} GB "
              "(cells above this are skipped to avoid OOM)")

    results: list[BenchResult] = []
    skipped: list[tuple[tuple[int, int], int, float]] = []
    for grid_shape in GRID_SHAPES:
        for batch_size in BATCH_SIZES:
            temp_gb = predicted_temp_gb(grid_shape, batch_size)
            if is_cpu and temp_gb > CPU_TEMP_BUDGET_GB:
                skipped.append((grid_shape, batch_size, temp_gb))
                continue
            try:
                r = benchmark(grid_shape, batch_size)
                results.append(r)
            except Exception as e:
                print(f"\n  FAILED grid={grid_shape} batch={batch_size}: {e}")
                # Don't bail — partial data is still informative.

    if skipped:
        print(f"\n  Skipped {len(skipped)} cell(s) over the {CPU_TEMP_BUDGET_GB:.1f} GB CPU budget:")
        for grid_shape, batch_size, temp_gb in skipped:
            print(f"    grid={grid_shape} batch={batch_size}  "
                  f"would allocate {temp_gb:.1f} GB temp tensor")

    print_results(results)
    verdict = m2_verdict(results)

    print(f"\n{'=' * 90}")
    print("  M2 throughput + memory gate verdict")
    print("=" * 90)
    for grid_str, v in verdict.items():
        mem_pass = "PASS" if v["memory_gate_passes"] else "FAIL"
        print(f"\n  Grid {grid_str}:")
        if v["throughput_gate_applies"]:
            thr_pass = "PASS" if v["throughput_gate_passes"] else "FAIL"
            print(f"    Throughput @ batch 256: {v['rollouts_per_sec_at_batch_256']:,.1f} rollouts/sec "
                  f"({v['speedup_vs_scipy']:.2f}x scipy)   [{thr_pass}]  "
                  "<- M2 throughput gate (reference workload)")
        else:
            print(f"    Throughput @ batch 256: {v['rollouts_per_sec_at_batch_256']:,.1f} rollouts/sec "
                  f"({v['speedup_vs_scipy']:.2f}x the 10x10 CPU baseline, informational)   "
                  "[not gated — run for memory + graph-size scaling]")
        peak = v["peak_device_mb_at_batch_256"]
        peak_str = f"{peak:.0f} MB" if peak is not None else "unknown"
        print(f"    Peak device memory @ batch 256: {peak_str}   [{mem_pass}]")

    # Write outputs.
    out_dir = REPO_ROOT / "results" / "m2_apsp_throughput"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    json_path = out_dir / f"m2_throughput_{ts}.json"
    json_path.write_text(json.dumps({
        "config": {
            "grid_shapes": GRID_SHAPES,
            "batch_sizes": BATCH_SIZES,
            "horizon": HORIZON,
            "k_iterations": DEFAULT_K_ITERATIONS,
            "warmup_episodes": WARMUP_EPISODES,
            "timed_episodes": TIMED_EPISODES,
            "scipy_cpu_baseline_rollouts_per_sec": SCIPY_CPU_BASELINE_ROLLOUTS_PER_SEC,
            "throughput_gate_at_batch_256": THROUGHPUT_GATE_AT_BATCH_256,
            "backend": jax.default_backend(),
            "devices": [str(d) for d in devices],
        },
        "results": [r.as_dict() for r in results],
        "verdict": verdict,
    }, indent=2))

    csv_path = out_dir / f"m2_throughput_{ts}.csv"
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
