"""Tests for the benchmark harness.

These tests verify the harness produces a well-formed JSON-ready dict
with the expected schema, that throughput numbers are positive, and that
re-running with the same config produces identical *returns* (timings of
course vary).

Smoke-level performance: tests use a tiny config (4x4 grid, horizon=3,
20 episodes) so the suite stays fast.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from topograph.bench import (
    SCHEMA_VERSION,
    BenchConfig,
    default_output_filename,
    format_summary,
    make_policy,
    run_benchmark,
)


@pytest.fixture
def small_cfg() -> BenchConfig:
    return BenchConfig(
        grid_shape=(4, 4),
        horizon=3,
        candidate_k_nearest=4,
        n_episodes=20,
        warmup_episodes=2,
    )


# ---------------------------------------------------------------------------
# make_policy
# ---------------------------------------------------------------------------


def test_make_policy_returns_callable_for_known_names() -> None:
    for name in ("random", "greedy", "noop"):
        policy = make_policy(name)
        assert callable(policy)


def test_make_policy_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown policy"):
        make_policy("definitely-not-a-policy")


def test_make_policy_rejects_scripted() -> None:
    """Scripted requires an explicit action sequence; CLI/factory can't supply one."""
    with pytest.raises(ValueError, match="action sequence"):
        make_policy("scripted")


# ---------------------------------------------------------------------------
# run_benchmark: schema and content
# ---------------------------------------------------------------------------


def test_result_has_expected_top_level_keys(small_cfg: BenchConfig) -> None:
    result = run_benchmark(small_cfg)
    expected = {
        "schema_version",
        "kind",
        "timestamp",
        "config",
        "world_info",
        "system_info",
        "summary",
        "returns_summary",
    }
    assert expected.issubset(result.keys())
    assert result["schema_version"] == SCHEMA_VERSION
    assert result["kind"] == "topograph_bench"


def test_result_summary_has_expected_metrics(small_cfg: BenchConfig) -> None:
    summary = run_benchmark(small_cfg)["summary"]
    expected = {
        "total_wall_s",
        "rollouts_per_sec",
        "env_steps_per_sec",
        "mean_episode_s",
        "std_episode_s",
        "median_episode_s",
        "min_episode_s",
        "max_episode_s",
        "p95_episode_s",
        "p99_episode_s",
    }
    assert expected.issubset(summary.keys())
    # All wall-time fields are non-negative; throughput is positive.
    assert summary["total_wall_s"] > 0
    assert summary["rollouts_per_sec"] > 0
    assert summary["env_steps_per_sec"] >= summary["rollouts_per_sec"]
    assert summary["mean_episode_s"] > 0
    assert summary["max_episode_s"] >= summary["min_episode_s"]


def test_result_world_info_matches_grid(small_cfg: BenchConfig) -> None:
    info = run_benchmark(small_cfg)["world_info"]
    assert info["n_zones"] == 16  # 4x4
    assert info["n_candidate_edges"] > 0
    assert info["action_dim"] == info["n_candidate_edges"] + 1


def test_result_returns_summary_consistent_with_cfg(small_cfg: BenchConfig) -> None:
    result = run_benchmark(small_cfg)
    assert "mean" in result["returns_summary"]
    # min <= mean is true for any distribution; this is an ordering
    # sanity check, not a dynamics check.
    assert result["returns_summary"]["min"] <= result["returns_summary"]["mean"]


def test_result_is_json_serializable(small_cfg: BenchConfig) -> None:
    result = run_benchmark(small_cfg)
    text = json.dumps(result)
    parsed = json.loads(text)
    assert parsed["schema_version"] == SCHEMA_VERSION


def test_include_raw_adds_raw_block(small_cfg: BenchConfig) -> None:
    plain = run_benchmark(small_cfg, include_raw=False)
    raw = run_benchmark(small_cfg, include_raw=True)
    assert "raw" not in plain
    assert "raw" in raw
    assert len(raw["raw"]["episode_seconds"]) == small_cfg.n_episodes
    assert len(raw["raw"]["episode_returns"]) == small_cfg.n_episodes


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_returns_are_reproducible_under_seed(small_cfg: BenchConfig) -> None:
    a = run_benchmark(small_cfg, include_raw=True)
    b = run_benchmark(small_cfg, include_raw=True)
    np.testing.assert_array_equal(
        a["raw"]["episode_returns"], b["raw"]["episode_returns"]
    )


def test_invalid_n_episodes_raises() -> None:
    cfg = BenchConfig(
        grid_shape=(3, 3), horizon=2, n_episodes=0, warmup_episodes=0
    )
    with pytest.raises(ValueError, match="n_episodes"):
        run_benchmark(cfg)


# ---------------------------------------------------------------------------
# format_summary: smoke
# ---------------------------------------------------------------------------


def test_format_summary_includes_key_fields(small_cfg: BenchConfig) -> None:
    text = format_summary(run_benchmark(small_cfg))
    assert "topograph-bench" in text
    assert "rollouts/sec" in text
    assert "env_steps/sec" in text
    assert "episode wall-time" in text


# ---------------------------------------------------------------------------
# default_output_filename
# ---------------------------------------------------------------------------


def test_default_output_filename_self_describing(small_cfg: BenchConfig) -> None:
    fname = default_output_filename(small_cfg, "2026-05-07T12:34:56+00:00")
    assert fname.startswith("cpu_baseline_")
    assert small_cfg.policy in fname
    assert "4x4" in fname
    assert "h3" in fname
    assert fname.endswith(".json")
    # Filename must be safe for any common filesystem.
    assert ":" not in fname
    assert " " not in fname


def test_results_dir_is_writable(tmp_path: Path, small_cfg: BenchConfig) -> None:
    """Smoke test for the round-trip the CLI does: run, serialize, write."""
    result = run_benchmark(small_cfg)
    out = tmp_path / default_output_filename(small_cfg, result["timestamp"])
    out.write_text(json.dumps(result, indent=2))
    assert out.exists()
    reloaded = json.loads(out.read_text())
    assert reloaded["schema_version"] == SCHEMA_VERSION
