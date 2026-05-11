"""CLI entry point: ``python -m topograph.bench`` (or ``topograph-bench``).

A thin wrapper around `topograph.bench.runner.run_benchmark`. All real
logic lives in the runner so the CLI can stay declarative.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .runner import (
    BenchConfig,
    default_output_filename,
    format_summary,
    run_benchmark,
)

# By default, write into the repo's ``results/bench/`` directory. The runner
# is repo-aware in the sense that this default points to a known location;
# users can override with ``--output``.
_DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[3] / "results" / "bench"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="topograph-bench",
        description=(
            "Measure CPU-baseline simulator throughput under a fixed policy. "
            "Reports rollouts/sec and per-episode wall-time statistics; "
            "writes a JSON record to results/bench/ by default."
        ),
    )
    parser.add_argument(
        "--grid",
        type=int,
        default=10,
        help="Grid side length (square, NxN). Default: 10.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=15,
        help="Episode length in steps. Default: 15.",
    )
    parser.add_argument(
        "--k-nearest",
        type=int,
        default=8,
        dest="k_nearest",
        help="k for the k-nearest candidate-edge construction. Default: 8.",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=50.0,
        help="Initial edge-construction budget (in grid units). Default: 50.",
    )
    parser.add_argument(
        "--policy",
        choices=["random", "greedy", "noop"],
        default="random",
        help="Policy to run. Default: random.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Number of measured episodes. Default: 1000.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup episodes (timings discarded). Default: 10.",
    )
    parser.add_argument(
        "--world-seed",
        type=int,
        default=0,
        help="Seed for world generation. Default: 0.",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=0,
        help="Seed for per-episode rng (offset by episode index). Default: 0.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSON path. Default: <repo>/results/bench/<auto-named>.json. "
            "Use --no-save to skip writing."
        ),
    )
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Include raw per-episode timings and returns in the output JSON.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write a JSON file; print the summary to stdout only.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = BenchConfig(
        grid_shape=(args.grid, args.grid),
        horizon=args.horizon,
        candidate_k_nearest=args.k_nearest,
        initial_budget=args.budget,
        policy=args.policy,
        n_episodes=args.episodes,
        warmup_episodes=args.warmup,
        world_seed=args.world_seed,
        rng_seed=args.rng_seed,
    )

    result = run_benchmark(cfg, include_raw=args.save_raw)
    print(format_summary(result))

    if not args.no_save:
        out_path = args.output
        if out_path is None:
            out_path = _DEFAULT_RESULTS_DIR / default_output_filename(
                cfg, result["timestamp"]
            )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
