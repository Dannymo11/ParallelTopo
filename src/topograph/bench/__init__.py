"""Benchmark harness — rollouts/sec under fixed-policy load.

Public surface:

* `BenchConfig` — typed configuration object.
* `run_benchmark(cfg)` — run and return a JSON-ready result dict.
* `format_summary(result)` — human-readable one-block render of a result.
* `make_policy(name)` — factory used by both the CLI and the runner.
* `default_output_filename(cfg, timestamp)` — stable JSON filename for
  ``results/bench/`` writes.
"""

from .runner import (
    SCHEMA_VERSION,
    BenchConfig,
    default_output_filename,
    format_summary,
    make_policy,
    run_benchmark,
)

__all__ = [
    "SCHEMA_VERSION",
    "BenchConfig",
    "default_output_filename",
    "format_summary",
    "make_policy",
    "run_benchmark",
]
