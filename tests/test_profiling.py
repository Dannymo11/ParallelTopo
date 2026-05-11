"""Tests for the per-component profiling harness.

Verify the breakdown produces a well-formed JSON-ready dict with the
expected schema, that fractions sum to 1.0, that the CSV round-trips,
and that the optional figure path degrades gracefully when matplotlib
isn't installed.

Smoke-level workload: tiny config (3x3 grid, horizon=2, 5 episodes)
so the suite stays fast.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from topograph.bench.profiling import (
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
from topograph.sim_cpu import GridCityConfig, make_world, reset


@pytest.fixture
def small_cfg() -> ProfileConfig:
    return ProfileConfig(
        grid_shape=(3, 3),
        horizon=2,
        candidate_k_nearest=4,
        n_episodes=5,
        warmup_episodes=1,
    )


# ---------------------------------------------------------------------------
# step_profiled
# ---------------------------------------------------------------------------


def test_step_profiled_records_all_components() -> None:
    world = make_world(seed=0, cfg=GridCityConfig(grid_shape=(3, 3), horizon=2))
    state = reset(world)
    _state, _r, _done, info = step_profiled(state, world.no_op_action)
    timings = info["timings_s"]
    for name in COMPONENTS:
        assert name in timings, f"missing component: {name}"
        assert timings[name] >= 0


def test_step_profiled_preserves_step_semantics() -> None:
    """Profiled step should produce identical (state, reward, done) to the
    production step (just with timing instrumentation added)."""
    from topograph.sim_cpu import step as production_step

    world = make_world(seed=0, cfg=GridCityConfig(grid_shape=(3, 3), horizon=2))

    # Take a single step from a fresh reset under both.
    s_prod = reset(world)
    s_prof = reset(world)
    next_prod, r_prod, done_prod, _ = production_step(s_prod, 0)
    next_prof, r_prof, done_prof, _ = step_profiled(s_prof, 0)
    assert r_prod == pytest.approx(r_prof)
    assert done_prod == done_prof
    assert next_prod.step == next_prof.step
    assert (next_prod.edge_mask == next_prof.edge_mask).all()


# ---------------------------------------------------------------------------
# run_profile: schema
# ---------------------------------------------------------------------------


def test_run_profile_top_level_keys(small_cfg: ProfileConfig) -> None:
    result = run_profile(small_cfg)
    expected = {
        "schema_version",
        "kind",
        "timestamp",
        "config",
        "world_info",
        "system_info",
        "total_steps",
        "wall_clock_total_s",
        "component_total_s",
        "policy_total_s",
        "components",
    }
    assert expected.issubset(result.keys())
    assert result["schema_version"] == PROFILE_SCHEMA_VERSION
    assert result["kind"] == "topograph_profile"


def test_run_profile_total_steps_matches_config(small_cfg: ProfileConfig) -> None:
    result = run_profile(small_cfg)
    assert result["total_steps"] == small_cfg.n_episodes * small_cfg.horizon


def test_run_profile_components_all_present(small_cfg: ProfileConfig) -> None:
    components = run_profile(small_cfg)["components"]
    for name in COMPONENTS:
        assert name in components
        c = components[name]
        for k in ("total_s", "mean_us_per_step", "fraction"):
            assert k in c
            assert c[k] >= 0


def test_run_profile_fractions_sum_to_one(small_cfg: ProfileConfig) -> None:
    components = run_profile(small_cfg)["components"]
    total = sum(c["fraction"] for c in components.values())
    assert total == pytest.approx(1.0, abs=1e-9)


def test_run_profile_apsp_is_dominant(small_cfg: ProfileConfig) -> None:
    """At any meaningful grid size, Floyd-Warshall should be the single
    largest component. Defended at the smallest meaningful size (3x3) — if
    APSP isn't already the heaviest here, the whole project's framing
    needs to change. This is the checkpoint's "is the hot spot what we
    think it is" gate.
    """
    components = run_profile(small_cfg)["components"]
    fractions = {name: c["fraction"] for name, c in components.items()}
    # APSP should beat the next-heaviest component by a non-trivial margin.
    top_two = sorted(fractions.values(), reverse=True)[:2]
    assert fractions["compute_travel_times"] == top_two[0]
    # And meaningfully dominant — at least 1.5x the runner-up. Loose to
    # avoid flakiness on tiny grids where the gap can be small.
    assert top_two[0] >= 1.5 * top_two[1]


def test_run_profile_json_serializable(small_cfg: ProfileConfig) -> None:
    json.dumps(run_profile(small_cfg))


# ---------------------------------------------------------------------------
# write_csv
# ---------------------------------------------------------------------------


def test_write_csv_roundtrip(small_cfg: ProfileConfig, tmp_path: Path) -> None:
    result = run_profile(small_cfg)
    out = tmp_path / "breakdown.csv"
    write_csv(result, out)
    assert out.exists()
    with out.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert {r["component"] for r in rows} == set(COMPONENTS)
    # Hot component first.
    assert rows[0]["component"] == "compute_travel_times"
    # All numeric columns parse.
    for r in rows:
        assert float(r["total_s"]) >= 0
        assert float(r["mean_us_per_step"]) >= 0
        assert 0 <= float(r["fraction"]) <= 1


# ---------------------------------------------------------------------------
# write_figure: optional, must degrade gracefully
# ---------------------------------------------------------------------------


def test_write_figure_returns_bool(small_cfg: ProfileConfig, tmp_path: Path) -> None:
    """Returns True if matplotlib available, False otherwise.
    Either way, must not raise."""
    result = run_profile(small_cfg)
    ok = write_figure(result, tmp_path / "breakdown.png")
    assert isinstance(ok, bool)
    if ok:
        assert (tmp_path / "breakdown.png").exists()


# ---------------------------------------------------------------------------
# format_breakdown and default_output_stem
# ---------------------------------------------------------------------------


def test_format_breakdown_includes_all_components(small_cfg: ProfileConfig) -> None:
    text = format_breakdown(run_profile(small_cfg))
    for name in COMPONENTS:
        assert name in text
    assert "topograph-profile" in text
    assert "step samples" in text


def test_default_output_stem_filesystem_safe(small_cfg: ProfileConfig) -> None:
    stem = default_output_stem(small_cfg, "2026-05-08T12:34:56+00:00")
    assert stem.startswith("cpu_breakdown_")
    assert "3x3" in stem
    assert "h2" in stem
    assert ":" not in stem
    assert " " not in stem
