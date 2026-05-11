"""Benchmark harness — rollouts/sec under fixed-policy load.

Public surface:

* `BenchConfig` — typed configuration object.
* `run_benchmark(cfg)` — run and return a JSON-ready result dict.
* `format_summary(result)` — human-readable one-block render of a result.
* `make_policy(name)` — factory used by both the CLI and the runner.
* `default_output_filename(cfg, timestamp)` — stable JSON filename for
  ``results/bench/`` writes.
"""

from .profiling import (
    COMPONENTS,
    PROFILE_SCHEMA_VERSION,
    ProfileConfig,
    default_output_stem,
    format_breakdown,
    run_profile,
    step_profiled,
    write_csv,
    write_figure,
)
from .runner import (
    SCHEMA_VERSION,
    BenchConfig,
    default_output_filename,
    format_summary,
    make_policy,
    run_benchmark,
)

__all__ = [
    "COMPONENTS",
    "PROFILE_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "BenchConfig",
    "ProfileConfig",
    "default_output_filename",
    "default_output_stem",
    "format_breakdown",
    "format_summary",
    "make_policy",
    "run_benchmark",
    "run_profile",
    "step_profiled",
    "write_csv",
    "write_figure",
]
