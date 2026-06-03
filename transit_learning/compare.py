"""Compare TopoGraph's matrix-squaring APSP against transit_learning's FW.

Three experiments, all on the same edge-cost matrices:

  equiv       Distance equivalence: does matrix-squaring (at a conservative K)
              produce the same shortest-path distances as transit_learning's
              exact Floyd-Warshall, on the real Mandl/Mumford graphs?

  ksweep      Convergence-K: for each city, the smallest K at which
              matrix-squaring is bit-exact vs FW. Validates whether TopoGraph's
              K=5 is enough for these (non-grid) topologies.

  throughput  Wall-clock APSP-calls/sec: transit_learning FW vs matrix-squaring,
              over batches of graphs, on CPU and (if available) GPU. This is the
              number that answers "does the TopoGraph kernel speed up
              transit_learning's shortest-path step?"

Run:
    python compare.py all
    python compare.py equiv --cities Mandl Mumford0 Mumford1
    python compare.py throughput --batch 256 --device cuda

The baseline is transit_learning's own `torch_utils.floyd_warshall`, imported
from the repo (point --tl-root at it if it is not the default location).
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch

from apsp_matrix_squaring import apsp_matrix_squaring, k_for_n_nodes
import data_loader as dl

DEFAULT_TL_ROOT = Path("/Users/dannymo/dev/transit_learning")
# tolerance for "same distance" -- fp32 round-off on these magnitudes
ATOL = 1e-3
RTOL = 1e-4


# --------------------------------------------------------------------------- #
# transit_learning baseline import
# --------------------------------------------------------------------------- #
def load_tl_floyd_warshall(tl_root: Path):
    """Import `floyd_warshall` from transit_learning's torch_utils.py.

    torch_utils only depends on torch, so we load it as a standalone module
    rather than importing the whole package (which would pull in heavy deps).
    """
    tu_path = Path(tl_root) / "torch_utils.py"
    if not tu_path.exists():
        raise FileNotFoundError(
            f"{tu_path} not found. Pass --tl-root <path to transit_learning repo>."
        )
    spec = importlib.util.spec_from_file_location("tl_torch_utils", tu_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["tl_torch_utils"] = module
    spec.loader.exec_module(module)
    return module.floyd_warshall


def fw_distances(floyd_warshall, edge_costs: torch.Tensor) -> torch.Tensor:
    """Run transit_learning FW and return just the (B, N, N) distance tensor.

    `floyd_warshall(..., return_raw_tensors=True)` returns (nexts, dists), both
    already shaped (B, N, N). We discard `nexts` -- matrix-squaring has no
    predecessor analogue, so distances are the only common ground.
    """
    _nexts, dists = floyd_warshall(edge_costs, return_raw_tensors=True)
    return dists


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #
def _diff_stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Compare two distance tensors, treating +inf==+inf as a match."""
    both_inf = torch.isinf(a) & torch.isinf(b)
    finite = ~(torch.isinf(a) | torch.isinf(b))
    inf_mismatch = int((torch.isinf(a) != torch.isinf(b)).sum())
    if finite.any():
        d = (a[finite] - b[finite]).abs()
        max_abs = float(d.max())
        denom = b[finite].abs().clamp(min=1e-9)
        max_rel = float((d / denom).max())
    else:
        max_abs = max_rel = 0.0
    close = bool(inf_mismatch == 0 and max_abs <= ATOL + RTOL * 1.0)
    return {
        "max_abs": max_abs,
        "max_rel": max_rel,
        "inf_mismatch": inf_mismatch,
        "n_finite": int(finite.sum()),
        "n_both_inf": int(both_inf.sum()),
        "close": close,
    }


def run_equiv(floyd_warshall, cities, instances_dir):
    print("\n=== Distance equivalence: matrix-squaring vs transit_learning FW ===")
    print(f"{'city':<10}{'N':>5}{'K':>4}{'max_abs':>12}{'max_rel':>12}"
          f"{'inf_mism':>10}  verdict")
    all_ok = True
    for city in cities:
        ec = dl.load_city_edge_costs(city, instances_dir)
        n = ec.shape[0]
        k = k_for_n_nodes(n)  # conservative: 2^K >= N-1
        batch = ec.unsqueeze(0)
        fw = fw_distances(floyd_warshall, batch)
        ms = apsp_matrix_squaring(batch, k_iterations=k)
        s = _diff_stats(ms, fw)
        verdict = "MATCH" if s["close"] else "DIFFER"
        all_ok &= s["close"]
        print(f"{city:<10}{n:>5}{k:>4}{s['max_abs']:>12.2e}{s['max_rel']:>12.2e}"
              f"{s['inf_mismatch']:>10}  {verdict}")
    print(f"\nOverall: {'ALL MATCH' if all_ok else 'MISMATCHES PRESENT'}")
    return all_ok


def run_ksweep(floyd_warshall, cities, instances_dir):
    print("\n=== Convergence-K: smallest K with bit-exact distances vs FW ===")
    print("(TopoGraph uses K=5 for grid cities; this checks it on real graphs)")
    print(f"{'city':<10}{'N':>5}{'conservative_K':>16}{'min_exact_K':>14}"
          f"{'K=5 exact?':>12}")
    for city in cities:
        ec = dl.load_city_edge_costs(city, instances_dir)
        n = ec.shape[0]
        batch = ec.unsqueeze(0)
        fw = fw_distances(floyd_warshall, batch)
        cons_k = k_for_n_nodes(n)
        min_exact = None
        for k in range(1, cons_k + 3):
            ms = apsp_matrix_squaring(batch, k_iterations=k)
            if _diff_stats(ms, fw)["close"]:
                min_exact = k
                break
        ms5 = apsp_matrix_squaring(batch, k_iterations=5)
        k5_ok = "yes" if _diff_stats(ms5, fw)["close"] else "NO"
        me = str(min_exact) if min_exact is not None else f">{cons_k + 2}"
        print(f"{city:<10}{n:>5}{cons_k:>16}{me:>14}{k5_ok:>12}")


def _time_fn(fn, n_iters: int, device: torch.device) -> float:
    """Median seconds per call over n_iters, after a warmup, with cuda sync."""
    # warmup
    for _ in range(3):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


def run_throughput(floyd_warshall, instances_dir, batch_sizes, device_str,
                   city, synth_n, n_iters):
    device = torch.device(device_str)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        device = torch.device("cpu")

    print(f"\n=== Throughput: FW vs matrix-squaring on {device.type.upper()} ===")
    print(f"(graph source: {city or f'synthetic N={synth_n}'}; "
          f"median of {n_iters} calls)")
    print(f"{'batch':>7}{'N':>5}{'K':>4}{'FW (ms)':>11}{'MS (ms)':>11}"
          f"{'speedup':>9}{'FW aps':>10}{'MS aps':>10}{'peak MB':>10}")

    for b in batch_sizes:
        if city:
            ec = dl.load_city_edge_costs(city, instances_dir)
            batch = dl.replicate_batch(ec, b).to(device)
        else:
            batch = dl.synthetic_batch(b, synth_n, device=device)
        n = batch.shape[-1]
        k = k_for_n_nodes(n)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        # Each cell is wrapped so one out-of-memory (torch eager materializes
        # the (B, N, N, N) min-plus intermediate -- O(B*N^3) -- and will OOM at
        # large batch*N where JAX/XLA fusion would not) does not abort the whole
        # overnight sweep. An OOM cell is itself a reportable data point: the
        # eager memory ceiling this kernel hits without fusion.
        try:
            fw_t = _time_fn(lambda: fw_distances(floyd_warshall, batch),
                            n_iters, device)
            ms_t = _time_fn(lambda: apsp_matrix_squaring(batch, k_iterations=k),
                            n_iters, device)
        except RuntimeError as e:
            kind = "OOM" if "out of memory" in str(e).lower() else "ERR"
            cell_peak = (torch.cuda.max_memory_allocated() / 1e6
                         if device.type == "cuda" else 0.0)
            print(f"{b:>7}{n:>5}{k:>4}{'  ' + kind:>33}"
                  f"   (peak before fail: {cell_peak:.0f} MB)")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            del batch
            continue

        speedup = fw_t / ms_t if ms_t > 0 else float("inf")
        cell_peak = (torch.cuda.max_memory_allocated() / 1e6
                     if device.type == "cuda" else 0.0)
        print(f"{b:>7}{n:>5}{k:>4}{fw_t * 1e3:>11.2f}{ms_t * 1e3:>11.2f}"
              f"{speedup:>8.1f}x{b / fw_t:>10.0f}{b / ms_t:>10.0f}"
              f"{cell_peak:>10.0f}")
        del batch
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print("\naps = APSP solves/sec. MS = matrix-squaring (TopoGraph kernel).")
    print("Note: matrix-squaring returns distances only; FW also returns the")
    print("predecessor matrix for path reconstruction (not timed-out here).")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["equiv", "ksweep", "throughput", "all"])
    p.add_argument("--tl-root", type=Path, default=DEFAULT_TL_ROOT,
                   help="Path to the transit_learning repo (for torch_utils.py).")
    p.add_argument("--instances-dir", type=Path, default=dl.DEFAULT_INSTANCES_DIR,
                   help="Mumford dataset 'Instances' dir (Mandl/Mumford*.txt).")
    p.add_argument("--cities", nargs="*", default=None,
                   help="Cities for equiv/ksweep (default: all present).")
    p.add_argument("--batch", nargs="*", type=int,
                   default=[1, 16, 64, 256, 1024],
                   help="Batch sizes for throughput.")
    p.add_argument("--device", default="cpu", help="cpu or cuda.")
    p.add_argument("--throughput-city", default=None,
                   help="Use this real city's graph for throughput (replicated). "
                        "Omit to use synthetic graphs of --synth-n nodes.")
    p.add_argument("--synth-n", type=int, default=100,
                   help="Node count for synthetic throughput graphs.")
    p.add_argument("--iters", type=int, default=20, help="Timed iters per cell.")
    args = p.parse_args()

    floyd_warshall = load_tl_floyd_warshall(args.tl_root)
    cities = args.cities or dl.available_cities(args.instances_dir)
    if not cities:
        raise SystemExit(
            f"No city files found under {args.instances_dir}. "
            f"See transit_learning/README.md for dataset setup.")

    if args.mode in ("equiv", "all"):
        run_equiv(floyd_warshall, cities, args.instances_dir)
    if args.mode in ("ksweep", "all"):
        run_ksweep(floyd_warshall, cities, args.instances_dir)
    if args.mode in ("throughput", "all"):
        run_throughput(floyd_warshall, args.instances_dir, args.batch,
                       args.device, args.throughput_city, args.synth_n, args.iters)


if __name__ == "__main__":
    main()
