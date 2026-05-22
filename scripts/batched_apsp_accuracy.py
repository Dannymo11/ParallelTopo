"""M2 accuracy study: does K-round parallel min-plus relaxation hit <2% error?

This is the first half of the M2 decision gate, run on CPU before any JAX or
GPU work. The M2 gate has three bars (accuracy, throughput, memory); the
accuracy bar is independent of where the code runs, so we answer it cheapest
first. If the answer is "yes, K iterations of parallel relaxation give <2%
mean error against exact APSP," the project proceeds to the JAX port. If it's
"no," the algorithm changes (more iterations, soft / log-sum-exp variant,
different scheme) *before* anything gets ported.

Two relaxation schemes are tested side-by-side, since both are GPU-friendly
and we should pick the better one for M3:

  (1) **Bellman-Ford-style.** Starting from the direct-distance matrix A,
      iterate D <- min(D, D x A) where 'x' is min-plus matmul. After K
      iterations, D contains shortest paths using <= K+1 hops. Requires
      K ~ graph diameter (in hops) for convergence on weighted graphs.

  (2) **Matrix-squaring.** Iterate D <- min(D, D x D). After K iterations,
      D contains shortest paths using <= 2^K hops. Converges in
      O(log diameter) iterations. Costs ~the same per iteration as (1)
      but needs far fewer iterations; the trade-off is that intermediate
      states are denser, which might matter on GPU memory for larger
      graphs.

For each (grid_size, mask_density) world, we compare both schemes against
the scipy FW ground truth at a sweep of K values and report mean / p95 /
max relative error.

Ground truth: `compute_travel_times` (scipy.sparse.csgraph.floyd_warshall).
Cross-backend equivalence with the hand-NumPy version is asserted by
`tests/test_dynamics.py::test_travel_times_scipy_matches_numpy_reference`,
so scipy is a clean reference.

Outputs:
  results/m2_apsp_accuracy/m2_accuracy_<ts>.json
  results/m2_apsp_accuracy/m2_accuracy_<ts>.csv

Reproduce:
  python scripts/batched_apsp_accuracy.py

The script ends with an explicit M2-gate verdict so the result is unambiguous.
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

from topograph.sim_cpu import GridCityConfig, compute_travel_times, make_world
from topograph.sim_cpu.dynamics import _build_direct_distance_matrix


# ---------------------------------------------------------------------------
# Relaxation schemes (both are what M3 will vmap across the batch)
# ---------------------------------------------------------------------------


def min_plus_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """(A x B)[i, j] = min_k (A[i, k] + B[k, j]).

    The natural GPU implementation uses a 3D broadcast. For N=225 the
    intermediate tensor is 225**3 * 8 bytes ~ 91 MB, which is fine on CPU
    and well within budget for a single-graph reference. On GPU under JAX
    this is the kernel that gets vmapped over the batch axis.
    """
    return (A[:, :, None] + B[None, :, :]).min(axis=1)


def apsp_bellman_ford(A: np.ndarray, k_iterations: int) -> np.ndarray:
    """D_0 = A; D_{t+1} = min(D_t, D_t x A). Paths with <= k+1 hops."""
    D = A.copy()
    for _ in range(k_iterations):
        D = np.minimum(D, min_plus_matmul(D, A))
    return D


def apsp_matrix_squaring(A: np.ndarray, k_iterations: int) -> np.ndarray:
    """D_0 = A; D_{t+1} = min(D_t, D_t x D_t). Paths with <= 2^k hops."""
    D = A.copy()
    for _ in range(k_iterations):
        D = np.minimum(D, min_plus_matmul(D, D))
    return D


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ErrorStats:
    mean_rel: float
    p95_rel: float
    max_rel: float
    max_abs: float

    def as_dict(self) -> dict:
        return {
            "mean_rel": self.mean_rel,
            "p95_rel": self.p95_rel,
            "max_rel": self.max_rel,
            "max_abs": self.max_abs,
        }


def relative_error_stats(approx: np.ndarray, exact: np.ndarray) -> ErrorStats:
    """Off-diagonal relative error.

    We ignore the zero diagonal (no path-to-self) and any exact-zero entry
    (would divide by zero). All off-diagonal exact distances are positive
    here because the walking floor guarantees every off-diagonal pair has
    a strictly positive travel time.
    """
    N = exact.shape[0]
    off_diag = ~np.eye(N, dtype=bool)
    valid = off_diag & (exact > 0)
    abs_err = np.abs(approx - exact)
    rel_err = abs_err[valid] / exact[valid]
    return ErrorStats(
        mean_rel=float(rel_err.mean()),
        p95_rel=float(np.percentile(rel_err, 95)),
        max_rel=float(rel_err.max()),
        max_abs=float(abs_err[off_diag].max()),
    )


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def make_mask(n_candidate: int, density: float, seed: int) -> np.ndarray:
    """Mid-episode mask shape: a uniformly random subset of candidate edges."""
    rng = np.random.default_rng(seed)
    mask = np.zeros(n_candidate, dtype=np.bool_)
    if density > 0:
        k = max(1, int(round(density * n_candidate)))
        mask[rng.choice(n_candidate, size=k, replace=False)] = True
    return mask


def graph_hop_diameter(A: np.ndarray) -> int:
    """Unweighted diameter of the *active-transit* graph, used to pick a
    sensible BF K range. Since the walking floor makes every pair reachable
    in one weighted hop, this is really 'how many transit edges might the
    longest shortest path chain'. Bounded above by N-1; in practice small.

    We approximate with the unweighted diameter assuming each cell is one
    hop from its 4 grid neighbors (true for a grid city's walk graph)."""
    N = A.shape[0]
    # Treat anything cheaper than half the max walk time as a "hop".
    # This is a heuristic to bound BF iterations sensibly; the exact bound
    # would come from the unweighted version of the active-edge graph.
    return int(min(N - 1, 2 * int(np.sqrt(N))))  # 2*sqrt(N) >= grid diameter


SWEEP_GRID_SHAPES: list[tuple[int, int]] = [(5, 5), (10, 10), (15, 15)]
SWEEP_DENSITIES: list[float] = [0.0, 0.25, 0.5, 0.75, 1.0]
SWEEP_K_BF: list[int] = [1, 2, 4, 8, 16, 24, 32]
SWEEP_K_SQUARE: list[int] = [1, 2, 3, 4, 5, 6, 7]

# Multiple seeds per (grid, density) to average out a single bad draw.
SEEDS: list[int] = [0, 1, 2]

# M2 accuracy threshold.
MEAN_REL_ERR_GATE: float = 0.02


def run() -> dict:
    rows: list[dict] = []
    for grid_shape in SWEEP_GRID_SHAPES:
        cfg = GridCityConfig(grid_shape=grid_shape)
        for density in SWEEP_DENSITIES:
            for seed in SEEDS:
                world = make_world(seed, cfg)
                mask = make_mask(world.n_candidate_edges, density, seed=seed + 1000)

                A = _build_direct_distance_matrix(world, mask)
                D_exact = compute_travel_times(world, mask)

                # Bellman-Ford-style sweep.
                for K in SWEEP_K_BF:
                    t0 = time.perf_counter()
                    D_approx = apsp_bellman_ford(A, K)
                    wall = time.perf_counter() - t0
                    stats = relative_error_stats(D_approx, D_exact)
                    rows.append({
                        "algorithm": "bellman_ford",
                        "grid_rows": grid_shape[0],
                        "grid_cols": grid_shape[1],
                        "n_zones": world.n_zones,
                        "n_candidate_edges": world.n_candidate_edges,
                        "density": density,
                        "n_active_edges": int(mask.sum()),
                        "seed": seed,
                        "K": K,
                        "wall_s": wall,
                        **stats.as_dict(),
                    })

                # Matrix-squaring sweep.
                for K in SWEEP_K_SQUARE:
                    t0 = time.perf_counter()
                    D_approx = apsp_matrix_squaring(A, K)
                    wall = time.perf_counter() - t0
                    stats = relative_error_stats(D_approx, D_exact)
                    rows.append({
                        "algorithm": "matrix_squaring",
                        "grid_rows": grid_shape[0],
                        "grid_cols": grid_shape[1],
                        "n_zones": world.n_zones,
                        "n_candidate_edges": world.n_candidate_edges,
                        "density": density,
                        "n_active_edges": int(mask.sum()),
                        "seed": seed,
                        "K": K,
                        "wall_s": wall,
                        **stats.as_dict(),
                    })
    return {"rows": rows}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def summarize(rows: list[dict]) -> dict:
    """Aggregate over seeds: for each (algorithm, grid, density, K), report
    seed-averaged mean / p95 / max relative error.
    """
    keyed: dict[tuple, list[dict]] = {}
    for r in rows:
        key = (r["algorithm"], (r["grid_rows"], r["grid_cols"]), r["density"], r["K"])
        keyed.setdefault(key, []).append(r)
    summary = []
    for key, group in keyed.items():
        summary.append({
            "algorithm": key[0],
            "grid_shape": key[1],
            "density": key[2],
            "K": key[3],
            "mean_rel_err": float(np.mean([g["mean_rel"] for g in group])),
            "p95_rel_err": float(np.mean([g["p95_rel"] for g in group])),
            "max_rel_err": float(np.mean([g["max_rel"] for g in group])),
            "mean_wall_s": float(np.mean([g["wall_s"] for g in group])),
            "n_seeds": len(group),
        })
    return {"summary": summary}


def print_table(summary: list[dict]) -> None:
    """Print compact per-algorithm tables: rows = K, cols = density."""
    by_algo: dict[str, list[dict]] = {}
    for s in summary:
        by_algo.setdefault(s["algorithm"], []).append(s)

    densities = sorted({s["density"] for s in summary})
    grid_shapes = sorted({s["grid_shape"] for s in summary})

    for algo, entries in by_algo.items():
        print(f"\n{'=' * 80}")
        print(f"  {algo}  —  mean relative error (averaged over seeds)")
        print("=" * 80)
        for grid in grid_shapes:
            print(f"\n  grid {grid[0]}x{grid[1]}  (N = {grid[0] * grid[1]})")
            header = f"  K = " + " ".join(f"{d:>8.2f}" for d in densities)
            print(f"  {'':>6}  " + "  ".join(f"d={d:.2f}" for d in densities))
            ks = sorted({s["K"] for s in entries if s["grid_shape"] == grid})
            for K in ks:
                cells = []
                for d in densities:
                    match = next(
                        (s for s in entries
                         if s["grid_shape"] == grid and s["K"] == K
                            and s["density"] == d),
                        None,
                    )
                    if match is None:
                        cells.append("    -   ")
                    else:
                        # Render as % with a marker if it passes the gate.
                        pct = match["mean_rel_err"] * 100
                        marker = "*" if match["mean_rel_err"] < MEAN_REL_ERR_GATE else " "
                        cells.append(f"{pct:>6.2f}%{marker}")
                print(f"  K={K:<4}  " + "  ".join(cells))
        print(f"\n  (* marks cells passing the M2 mean-error gate of <{MEAN_REL_ERR_GATE * 100:.0f}%)")


def m2_verdict(summary: list[dict]) -> dict:
    """Decide pass/fail at the M2-spec K choices.

    BF gate: K = graph_hop_diameter (roughly 2*sqrt(N)).
    Squaring gate: K = ceil(log2(graph_hop_diameter)).

    Each (grid, density) combination must hit mean_rel_err < 2% at that K.
    """
    results = []
    for grid in sorted({s["grid_shape"] for s in summary}):
        N = grid[0] * grid[1]
        bf_K = int(min(max(int(2 * np.sqrt(N)), 1), 32))
        sq_K = int(max(1, np.ceil(np.log2(bf_K)).item()))

        for algo, target_K in (("bellman_ford", bf_K), ("matrix_squaring", sq_K)):
            # Snap to the nearest K we actually swept.
            available_K = sorted({s["K"] for s in summary if s["algorithm"] == algo})
            if not available_K:
                continue
            chosen_K = min(available_K, key=lambda k: (abs(k - target_K), k))
            for d in sorted({s["density"] for s in summary}):
                match = next(
                    (s for s in summary
                     if s["algorithm"] == algo and s["grid_shape"] == grid
                        and s["K"] == chosen_K and s["density"] == d),
                    None,
                )
                if match is None:
                    continue
                results.append({
                    "algorithm": algo,
                    "grid_shape": grid,
                    "density": d,
                    "target_K": target_K,
                    "evaluated_K": chosen_K,
                    "mean_rel_err": match["mean_rel_err"],
                    "passes_gate": match["mean_rel_err"] < MEAN_REL_ERR_GATE,
                })

    by_algo: dict[str, dict] = {}
    for algo in sorted({r["algorithm"] for r in results}):
        cells = [r for r in results if r["algorithm"] == algo]
        by_algo[algo] = {
            "n_cells": len(cells),
            "n_passing": sum(1 for c in cells if c["passes_gate"]),
            "worst_cell": max(cells, key=lambda c: c["mean_rel_err"]),
        }
    return {"per_cell": results, "by_algorithm": by_algo}


def main() -> None:
    print("=" * 80)
    print("M2 accuracy study: K-round parallel min-plus relaxation vs exact APSP")
    print("=" * 80)
    print(f"  Grid shapes:  {SWEEP_GRID_SHAPES}")
    print(f"  Densities:    {SWEEP_DENSITIES}")
    print(f"  Seeds:        {SEEDS}")
    print(f"  K (BF):       {SWEEP_K_BF}")
    print(f"  K (squaring): {SWEEP_K_SQUARE}")
    print(f"  Mean-err gate: < {MEAN_REL_ERR_GATE * 100:.1f}%")

    t0 = time.perf_counter()
    raw = run()
    summary = summarize(raw["rows"])["summary"]
    wall = time.perf_counter() - t0
    print(f"\n  Total wall time: {wall:.1f} s   ({len(raw['rows'])} (algo,grid,density,seed,K) cells)")

    print_table(summary)

    verdict = m2_verdict(summary)
    print(f"\n{'=' * 80}")
    print("  M2 accuracy gate verdict")
    print("=" * 80)
    for algo, info in verdict["by_algorithm"].items():
        worst = info["worst_cell"]
        print(f"\n  {algo}:  {info['n_passing']}/{info['n_cells']} cells pass at target K")
        print(f"    worst: grid {worst['grid_shape']}  density {worst['density']:.2f}  "
              f"K={worst['evaluated_K']}  mean_err={worst['mean_rel_err'] * 100:.3f}%")

    # Write outputs.
    out_dir = REPO_ROOT / "results" / "m2_apsp_accuracy"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    json_path = out_dir / f"m2_accuracy_{ts}.json"
    json_path.write_text(json.dumps({
        "config": {
            "grid_shapes": SWEEP_GRID_SHAPES,
            "densities": SWEEP_DENSITIES,
            "seeds": SEEDS,
            "K_bf": SWEEP_K_BF,
            "K_squaring": SWEEP_K_SQUARE,
            "mean_rel_err_gate": MEAN_REL_ERR_GATE,
        },
        "raw_rows": raw["rows"],
        "summary": summary,
        "verdict": verdict,
        "wall_s": wall,
    }, indent=2, default=str))

    csv_path = out_dir / f"m2_accuracy_{ts}.csv"
    fieldnames = list(raw["rows"][0].keys())
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(raw["rows"])

    print(f"\n  Wrote: {json_path.relative_to(REPO_ROOT)}")
    print(f"  Wrote: {csv_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
