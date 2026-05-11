"""Benchmark harness — rollouts/sec under fixed-policy load.

Pure programmatic API. The CLI in ``__main__.py`` is a thin wrapper. Output
is a JSON-ready dict so figures and reports can pull straight from
``results/bench/*.json`` without intermediate adapters.

Key design decisions:

* **Throughput is computed from the overall wall clock, not the sum of
  per-episode timings.** Bookkeeping between episodes (dataclass.replace,
  rng spawning, list appends) is part of the realistic cost; hiding it
  inflates the headline number.
* **Warmup episodes are run before the timed loop** to absorb first-call
  caching effects (NumPy ufunc dispatch, allocator warm-up, etc.). They
  are *not* included in the reported summary.
* **Per-episode rng seeds are deterministic from `rng_seed + i`** so a
  rerun with the same config and seeds produces identical trajectories
  and identical timings (modulo wall-clock noise).
* **Per-episode raw timings and returns are kept in memory and surfaced
  through the summary stats**, but only written into the output JSON
  when `include_raw=True`. This keeps the default JSON small and
  diff-friendly across runs while still allowing per-episode analysis
  later.
"""

from __future__ import annotations

import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

from topograph.policies import OneStepGreedyPolicy, RandomLegalPolicy, ScriptedPolicy
from topograph.sim_cpu import (
    GridCityConfig,
    Policy,
    State,
    make_world,
    reset,
    run_episode,
)

SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BenchConfig:
    """Inputs to a single benchmark run.

    Defaults match the M1 representative workload: 10x10 grid, horizon 15,
    k=8 nearest neighbors, 1000 measured episodes after 10 warmup, random
    policy. Override grid/horizon/policy to sweep along any single axis.
    """

    grid_shape: tuple[int, int] = (10, 10)
    horizon: int = 15
    candidate_k_nearest: int = 8
    initial_budget: float = 50.0
    policy: str = "random"
    n_episodes: int = 1000
    warmup_episodes: int = 10
    world_seed: int = 0
    rng_seed: int = 0


# ---------------------------------------------------------------------------
# Policy factory
# ---------------------------------------------------------------------------


class _NoOpPolicy:
    """No-op policy. Used as a "bench-the-driver" baseline that excludes
    policy cost from the throughput number."""

    def __call__(self, state: State, rng: np.random.Generator) -> int:
        del rng
        return state.world.no_op_action


_POLICY_REGISTRY: dict[str, type[Policy] | type[_NoOpPolicy]] = {
    "random": RandomLegalPolicy,
    "greedy": OneStepGreedyPolicy,
    "noop": _NoOpPolicy,
    "scripted": ScriptedPolicy,  # rarely used directly via CLI; present for API
}


def make_policy(name: str) -> Policy:
    """Return a fresh instance of the named policy."""
    cls = _POLICY_REGISTRY.get(name)
    if cls is None:
        valid = sorted(_POLICY_REGISTRY)
        raise ValueError(f"unknown policy {name!r}; valid choices: {valid}")
    if cls is ScriptedPolicy:
        # ScriptedPolicy needs an explicit action sequence; the CLI doesn't
        # carry one, so we expose it through the registry but ask callers
        # to build it directly.
        raise ValueError(
            "ScriptedPolicy must be constructed directly with an action sequence"
        )
    return cls()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Core run
# ---------------------------------------------------------------------------


def run_benchmark(
    cfg: BenchConfig,
    *,
    include_raw: bool = False,
) -> dict[str, Any]:
    """Run the benchmark and return a JSON-ready result dict.

    The returned dict has a stable schema (top-level keys + ``schema_version``)
    so downstream figure scripts can rely on it.
    """
    world_cfg = GridCityConfig(
        grid_shape=cfg.grid_shape,
        horizon=cfg.horizon,
        candidate_k_nearest=cfg.candidate_k_nearest,
        initial_budget=cfg.initial_budget,
    )
    world = make_world(seed=cfg.world_seed, cfg=world_cfg)
    policy = make_policy(cfg.policy)

    # Warmup: discard timings.
    for i in range(cfg.warmup_episodes):
        run_episode(reset(world), policy, rng=cfg.rng_seed + 10_000_000 + i)

    n = cfg.n_episodes
    if n <= 0:
        raise ValueError(f"n_episodes must be positive, got {n}")
    times = np.empty(n, dtype=np.float64)
    returns = np.empty(n, dtype=np.float64)

    overall_t0 = time.perf_counter()
    for i in range(n):
        t0 = time.perf_counter()
        traj = run_episode(reset(world), policy, rng=cfg.rng_seed + i)
        t1 = time.perf_counter()
        times[i] = t1 - t0
        returns[i] = traj.episode_return
    overall_t1 = time.perf_counter()
    total_wall_s = overall_t1 - overall_t0

    n_steps = n * cfg.horizon
    summary = {
        "total_wall_s": total_wall_s,
        "rollouts_per_sec": n / total_wall_s if total_wall_s > 0 else float("inf"),
        "env_steps_per_sec": n_steps / total_wall_s if total_wall_s > 0 else float("inf"),
        "mean_episode_s": float(times.mean()),
        "std_episode_s": float(times.std(ddof=1)) if n > 1 else 0.0,
        "median_episode_s": float(np.median(times)),
        "min_episode_s": float(times.min()),
        "max_episode_s": float(times.max()),
        "p95_episode_s": float(np.quantile(times, 0.95)),
        "p99_episode_s": float(np.quantile(times, 0.99)),
    }

    returns_summary = {
        "mean": float(returns.mean()),
        "std": float(returns.std(ddof=1)) if n > 1 else 0.0,
        "min": float(returns.min()),
        "max": float(returns.max()),
    }

    out: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "topograph_bench",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": asdict(cfg),
        "world_info": {
            "n_zones": world.n_zones,
            "n_candidate_edges": world.n_candidate_edges,
            "action_dim": world.action_dim,
        },
        "system_info": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "summary": summary,
        "returns_summary": returns_summary,
    }
    if include_raw:
        out["raw"] = {
            "episode_seconds": times.tolist(),
            "episode_returns": returns.tolist(),
        }
    return out


# ---------------------------------------------------------------------------
# Pretty-printing for the CLI
# ---------------------------------------------------------------------------


def format_summary(result: dict[str, Any]) -> str:
    """Render a single benchmark result as a compact human-readable block."""
    cfg = result["config"]
    summary = result["summary"]
    info = result["world_info"]
    rs = result["returns_summary"]

    grid_r, grid_c = cfg["grid_shape"]
    lines = [
        f"topograph-bench  ({result['timestamp']})",
        f"  config:    grid={grid_r}x{grid_c}  horizon={cfg['horizon']}  "
        f"policy={cfg['policy']}  episodes={cfg['n_episodes']} (+{cfg['warmup_episodes']} warmup)",
        f"  world:     n_zones={info['n_zones']}  "
        f"n_candidate_edges={info['n_candidate_edges']}  action_dim={info['action_dim']}",
        f"  throughput: {summary['rollouts_per_sec']:>12,.1f} rollouts/sec   "
        f"{summary['env_steps_per_sec']:>14,.1f} env_steps/sec",
        f"  episode wall-time (ms): "
        f"mean={summary['mean_episode_s'] * 1e3:.3f}  "
        f"std={summary['std_episode_s'] * 1e3:.3f}  "
        f"p50={summary['median_episode_s'] * 1e3:.3f}  "
        f"p95={summary['p95_episode_s'] * 1e3:.3f}  "
        f"p99={summary['p99_episode_s'] * 1e3:.3f}",
        f"  episode return:  mean={rs['mean']:.4g}  std={rs['std']:.4g}  "
        f"min={rs['min']:.4g}  max={rs['max']:.4g}",
        f"  total wall: {summary['total_wall_s']:.3f}s",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Default output path
# ---------------------------------------------------------------------------


def default_output_filename(cfg: BenchConfig, timestamp: str) -> str:
    """Filename for the default ``results/bench/<file>.json`` location.

    Format: ``cpu_baseline_<policy>_<grid>x<grid>_h<horizon>_<YYYYMMDDTHHMMSS>.json``
    Stable across runs at the second granularity, sortable, and self-
    describing without needing to open the file.
    """
    # Strip the timezone offset and colons for a filename-safe string.
    ts_safe = timestamp.replace(":", "").replace("-", "").replace("+0000", "Z")
    g_r, g_c = cfg.grid_shape
    return (
        f"cpu_baseline_{cfg.policy}_{g_r}x{g_c}_h{cfg.horizon}_{ts_safe}.json"
    )
