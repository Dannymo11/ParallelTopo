"""Compare our hand-vectorized Floyd-Warshall against scipy's C-implemented FW.

The motivating question: is the CPU baseline's throughput floor (66 rollouts/sec)
artificially low because we're calling FW from Python? If scipy is materially
faster, the projected GPU speedup needs to be measured against scipy's CPU,
not ours. The risk we're checking: that the 109x slowdown vs the stub is "we
used naive NumPy FW", not "the dynamics are genuinely heavy."

Three measurements:

  (1) Per-call APSP cost on identical inputs:
        - numpy_fw  : our hand-vectorized loop
        - scipy_fw  : scipy.sparse.csgraph.floyd_warshall on the same matrix
      Result: median / mean / std over N calls.

  (2) Numerical equivalence: max abs error between the two. If non-zero we have
      a bug; both should compute exact APSP.

  (3) End-to-end throughput under each: monkey-patch dynamics.compute_travel_times
      with the scipy variant and rerun run_benchmark on the canonical
      10x10/horizon-15/random-policy/200-episode workload. Report rollouts/sec
      with each FW backend and the implied per-step cost from the bench.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from topograph.bench.runner import BenchConfig, run_benchmark
from topograph.sim_cpu import GridCityConfig, make_world, reset
from topograph.sim_cpu import api as _api
from topograph.sim_cpu import dynamics as _dynamics

# Both implementations live in the production code path; this script just
# swaps which one `step` calls so the comparison is exactly what production
# would do. `compute_travel_times` is the scipy default (since the swap
# landed); `compute_travel_times_numpy` is the hand-vectorized reference
# that the GPU port (M3) cross-checks correctness against.
scipy_fw = _dynamics.compute_travel_times
numpy_fw = _dynamics.compute_travel_times_numpy


def time_per_call(fn, world, edge_mask, n_calls=200, warmup=10):
    for _ in range(warmup):
        fn(world, edge_mask)
    samples = np.empty(n_calls, dtype=np.float64)
    for i in range(n_calls):
        t0 = time.perf_counter()
        fn(world, edge_mask)
        samples[i] = time.perf_counter() - t0
    return samples


def fmt_us(s_samples):
    us = s_samples * 1e6
    return f"mean={us.mean():.1f} µs  median={np.median(us):.1f} µs  std={us.std(ddof=1):.1f} µs"


def main():
    print("=" * 78)
    print("APSP baseline comparison: hand-NumPy FW  vs  scipy FW")
    print("=" * 78)

    for grid_n in (5, 10, 15):
        world = make_world(0, GridCityConfig(grid_shape=(grid_n, grid_n)))
        # Test on both empty mask and a half-filled mask (mid-episode shape).
        for mask_kind, mask in (
            ("empty",       np.zeros(world.n_candidate_edges, dtype=np.bool_)),
            ("half_active", _half_active_mask(world.n_candidate_edges)),
        ):
            t_numpy = time_per_call(numpy_fw, world, mask)
            t_scipy = time_per_call(scipy_fw, world, mask)

            r_numpy = numpy_fw(world, mask)
            r_scipy = scipy_fw(world, mask)
            max_err = float(np.max(np.abs(r_numpy - r_scipy)))
            rel_err = max_err / float(np.max(r_numpy)) if r_numpy.max() > 0 else 0.0

            print(f"\n--- grid={grid_n}x{grid_n} (N={world.n_zones}), mask={mask_kind} ---")
            print(f"  numpy_fw : {fmt_us(t_numpy)}")
            print(f"  scipy_fw : {fmt_us(t_scipy)}")
            speedup = float(np.median(t_numpy) / np.median(t_scipy))
            print(f"  speedup  : {speedup:.2f}x  (scipy faster)")
            print(f"  max abs error vs numpy: {max_err:.3e}  (rel {rel_err:.2e})")

    # End-to-end throughput comparison on the canonical workload.
    print("\n" + "=" * 78)
    print("End-to-end bench: 10x10 / horizon 15 / random / 200 episodes")
    print("=" * 78)

    cfg = BenchConfig(
        grid_shape=(10, 10),
        horizon=15,
        candidate_k_nearest=8,
        initial_budget=50.0,
        policy="random",
        n_episodes=200,
        warmup_episodes=5,
    )

    # Baseline (NumPy)
    _dynamics.compute_travel_times = numpy_fw
    _api.compute_travel_times = numpy_fw
    result_np = run_benchmark(cfg)
    rps_np = result_np["summary"]["rollouts_per_sec"]
    eps_np = result_np["summary"]["env_steps_per_sec"]
    ms_np = result_np["summary"]["mean_episode_s"] * 1e3

    # Swapped (scipy)
    _dynamics.compute_travel_times = scipy_fw
    _api.compute_travel_times = scipy_fw
    result_sp = run_benchmark(cfg)
    rps_sp = result_sp["summary"]["rollouts_per_sec"]
    eps_sp = result_sp["summary"]["env_steps_per_sec"]
    ms_sp = result_sp["summary"]["mean_episode_s"] * 1e3

    # Restore (clean up monkey-patch).
    _dynamics.compute_travel_times = numpy_fw
    _api.compute_travel_times = numpy_fw

    print(f"  numpy_fw : {rps_np:>8,.1f} rollouts/sec   "
          f"{eps_np:>10,.1f} env_steps/sec   {ms_np:>6.2f} ms/episode")
    print(f"  scipy_fw : {rps_sp:>8,.1f} rollouts/sec   "
          f"{eps_sp:>10,.1f} env_steps/sec   {ms_sp:>6.2f} ms/episode")
    print(f"  end-to-end speedup: {rps_sp / rps_np:.2f}x")
    print(f"\n  Returns sanity:  numpy mean={result_np['returns_summary']['mean']:.3f}  "
          f"scipy mean={result_sp['returns_summary']['mean']:.3f}  "
          f"(should match — same RNG, identical APSP outputs)")


def _half_active_mask(n_candidate, seed=0):
    rng = np.random.default_rng(seed)
    mask = np.zeros(n_candidate, dtype=np.bool_)
    mask[rng.choice(n_candidate, size=n_candidate // 2, replace=False)] = True
    return mask


if __name__ == "__main__":
    main()
