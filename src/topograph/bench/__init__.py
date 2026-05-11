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

# Note: profiling symbols (ProfileConfig, run_profile, step_profiled, ...)
# are NOT re-exported here. Importing them via `topograph.bench` would
# trigger an eager import of `topograph.bench.profiling`, which then
# breaks `python -m topograph.bench.profiling` with a RuntimeWarning
# about the module being in sys.modules before its own execution. Import
# them from `topograph.bench.profiling` directly instead.

__all__ = [
    "SCHEMA_VERSION",
    "BenchConfig",
    "default_output_filename",
    "format_summary",
    "make_policy",
    "run_benchmark",
]
