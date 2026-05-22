"""Per-component unit tests for the simulator dynamics.

Each function in `topograph.sim_cpu.dynamics` is tested in isolation.
These tests are mostly *characterization* — they pin down the formula
each component implements so that an inadvertent sign flip, off-by-one,
or units mismatch surfaces here, before contaminating trajectory-level
behavior or the benchmark numbers.

For the APSP travel-time test we cross-check against
`scipy.sparse.csgraph.shortest_path` (Floyd-Warshall in C) on the same
graph, so a regression in our hand-vectorized FW shows up immediately.
"""

from __future__ import annotations

import numpy as np
import pytest
from topograph.sim_cpu import (
    GridCityConfig,
    WorldConfig,
    apply_action,
    compute_accessibility,
    compute_demand,
    compute_travel_times,
    compute_travel_times_numpy,
    compute_welfare,
    find_edge_index,
    make_world,
    reset,
    update_activity,
)


@pytest.fixture
def world() -> WorldConfig:
    return make_world(seed=0, cfg=GridCityConfig(grid_shape=(5, 5), horizon=5))


@pytest.fixture
def small_world() -> WorldConfig:
    """Tiny 3x3 grid, easier to reason about by hand."""
    return make_world(
        seed=0,
        cfg=GridCityConfig(grid_shape=(3, 3), horizon=3, candidate_k_nearest=4),
    )


# ---------------------------------------------------------------------------
# apply_action
# ---------------------------------------------------------------------------


def test_apply_action_no_op_unchanged(world: WorldConfig) -> None:
    s = reset(world)
    new_mask, new_budget, did = apply_action(s, world.no_op_action)
    assert new_mask is s.edge_mask  # not copied
    assert new_budget == s.budget_remaining
    assert did is False


def test_apply_action_legal_edge_activates_and_charges(world: WorldConfig) -> None:
    s = reset(world)
    new_mask, new_budget, did = apply_action(s, 0)
    assert did is True
    assert new_mask[0]
    assert new_mask is not s.edge_mask  # fresh array
    assert not s.edge_mask[0]  # original untouched
    assert new_budget == pytest.approx(
        s.budget_remaining - float(world.cost_per_edge[0])
    )


def test_apply_action_already_active_silent_no_op(world: WorldConfig) -> None:
    s = reset(world)
    mask = s.edge_mask.copy()
    mask[0] = True
    s = type(s)(
        world=s.world,
        edge_mask=mask,
        activity=s.activity,
        budget_remaining=s.budget_remaining,
        step=s.step,
        done=s.done,
    )
    _, new_budget, did = apply_action(s, 0)
    assert did is False
    assert new_budget == s.budget_remaining


def test_apply_action_unaffordable_silent_no_op(world: WorldConfig) -> None:
    s = reset(world)
    s = type(s)(
        world=s.world,
        edge_mask=s.edge_mask,
        activity=s.activity,
        budget_remaining=0.0,  # broke
        step=s.step,
        done=s.done,
    )
    new_mask, new_budget, did = apply_action(s, 0)
    assert did is False
    assert new_budget == 0.0
    assert not new_mask[0]


# ---------------------------------------------------------------------------
# compute_travel_times
# ---------------------------------------------------------------------------


def test_travel_times_scipy_matches_numpy_reference(world: WorldConfig) -> None:
    """The scipy (default) and hand-NumPy backends must produce identical
    output. This is the cross-implementation check: a regression in either
    backend's matrix-construction or APSP call surfaces here before
    contaminating downstream behavior."""
    for mask_kind, mask in (
        ("empty", np.zeros(world.n_candidate_edges, dtype=np.bool_)),
        ("first_three_active", _mask_first_n(world.n_candidate_edges, 3)),
    ):
        t_scipy = compute_travel_times(world, mask)
        t_numpy = compute_travel_times_numpy(world, mask)
        np.testing.assert_allclose(
            t_scipy, t_numpy, rtol=1e-9, atol=1e-9,
            err_msg=f"scipy vs numpy mismatch on {mask_kind}",
        )


def _mask_first_n(n_candidate: int, n: int) -> np.ndarray:
    mask = np.zeros(n_candidate, dtype=np.bool_)
    mask[: min(n, n_candidate)] = True
    return mask


def test_travel_times_active_edge_does_not_increase_any_path(world: WorldConfig) -> None:
    """Adding shortcut edges can only shorten or preserve any pair's path."""
    empty = compute_travel_times(world, np.zeros(world.n_candidate_edges, dtype=np.bool_))
    mask = np.zeros(world.n_candidate_edges, dtype=np.bool_)
    mask[0] = True
    with_one = compute_travel_times(world, mask)
    assert (with_one <= empty + 1e-9).all()


def test_travel_times_zero_diagonal_and_symmetric(world: WorldConfig) -> None:
    mask = np.zeros(world.n_candidate_edges, dtype=np.bool_)
    mask[: min(3, world.n_candidate_edges)] = True
    t = compute_travel_times(world, mask)
    np.testing.assert_allclose(np.diag(t), 0.0, atol=1e-12)
    np.testing.assert_allclose(t, t.T, rtol=1e-9, atol=1e-9)


def test_travel_times_active_edge_pair_uses_transit_weight(small_world: WorldConfig) -> None:
    """Activating a single edge between adjacent zones should drop their
    direct-pair travel time to the transit weight (faster than walking)."""
    a, b = 0, 1
    e_idx = find_edge_index(small_world, a, b)
    transit_weight = float(small_world.candidate_edge_weights[e_idx])
    walk_weight = float(small_world.walk_times[a, b])
    assert transit_weight < walk_weight  # transit faster than walking, by construction
    mask = np.zeros(small_world.n_candidate_edges, dtype=np.bool_)
    mask[e_idx] = True
    t = compute_travel_times(small_world, mask)
    assert t[a, b] == pytest.approx(transit_weight, rel=1e-9)


# ---------------------------------------------------------------------------
# compute_accessibility
# ---------------------------------------------------------------------------


def test_accessibility_uniform_activity_and_geometry(world: WorldConfig) -> None:
    """For uniform A and a symmetric travel-time matrix, accessibility is
    determined entirely by zone position (not population). Center zones
    score higher than corner zones."""
    s = reset(world)
    t = compute_travel_times(world, s.edge_mask)
    acc = compute_accessibility(world, s.activity, t)
    # 5x5: center is index 12, corner is index 0.
    assert acc[12] > acc[0]


def test_accessibility_extreme_decay_collapses_to_self(world: WorldConfig) -> None:
    """In the limit of very fast decay, off-diagonal kernel terms vanish
    and acc_i collapses to A_i. This is the cleanest characterization of
    what `accessibility_decay` does — larger decay = shorter reach."""
    s = reset(world)
    t = compute_travel_times(world, s.edge_mask)
    # Smallest off-diagonal walking time on the 5x5 grid is one cell at
    # walk_speed: 500 / 1.4 ≈ 357 s. exp(-10 * 357) << machine epsilon.
    extreme = WorldConfig(
        zone_positions=world.zone_positions,
        walk_distances=world.walk_distances,
        walk_times=world.walk_times,
        candidate_edges=world.candidate_edges,
        candidate_edge_weights=world.candidate_edge_weights,
        cost_per_edge=world.cost_per_edge,
        initial_activity=world.initial_activity,
        initial_budget=world.initial_budget,
        horizon=world.horizon,
        gravity_alpha=world.gravity_alpha,
        gravity_beta=world.gravity_beta,
        accessibility_decay=10.0,  # per second — kills any off-diagonal contribution
        growth_rate=world.growth_rate,
    )
    acc_extreme = compute_accessibility(extreme, s.activity, t)
    np.testing.assert_allclose(acc_extreme, s.activity, rtol=1e-6)


# ---------------------------------------------------------------------------
# update_activity
# ---------------------------------------------------------------------------


def test_update_activity_preserves_total(world: WorldConfig) -> None:
    s = reset(world)
    t = compute_travel_times(world, s.edge_mask)
    acc = compute_accessibility(world, s.activity, t)
    new = update_activity(s.activity, acc, growth_rate=0.5)
    assert new.sum() == pytest.approx(s.activity.sum(), rel=1e-12)


def test_update_activity_grows_high_acc_shrinks_low_acc(world: WorldConfig) -> None:
    s = reset(world)
    t = compute_travel_times(world, s.edge_mask)
    acc = compute_accessibility(world, s.activity, t)
    new = update_activity(s.activity, acc, growth_rate=0.5)
    # Center has higher acc than corner, so center activity grows.
    assert new[12] > s.activity[12]
    assert new[0] < s.activity[0]


def test_update_activity_zero_growth_rate_is_identity(world: WorldConfig) -> None:
    s = reset(world)
    t = compute_travel_times(world, s.edge_mask)
    acc = compute_accessibility(world, s.activity, t)
    new = update_activity(s.activity, acc, growth_rate=0.0)
    np.testing.assert_allclose(new, s.activity, rtol=1e-12)


# ---------------------------------------------------------------------------
# compute_welfare
# ---------------------------------------------------------------------------


def test_welfare_uniform_activity_equals_mean_acc(world: WorldConfig) -> None:
    s = reset(world)  # uniform activity by construction
    t = compute_travel_times(world, s.edge_mask)
    acc = compute_accessibility(world, s.activity, t)
    welfare = compute_welfare(s.activity, acc)
    assert welfare == pytest.approx(float(acc.mean()), rel=1e-12)


def test_welfare_zero_activity_returns_zero(world: WorldConfig) -> None:
    zeros = np.zeros(world.n_zones, dtype=np.float64)
    acc = np.ones(world.n_zones, dtype=np.float64)  # arbitrary
    assert compute_welfare(zeros, acc) == 0.0


# ---------------------------------------------------------------------------
# compute_demand
# ---------------------------------------------------------------------------


def test_demand_diagonal_is_zero(world: WorldConfig) -> None:
    s = reset(world)
    t = compute_travel_times(world, s.edge_mask)
    d = compute_demand(world, s.activity, t)
    assert (np.diag(d) == 0.0).all()


def test_demand_scales_quadratically_with_activity(world: WorldConfig) -> None:
    s = reset(world)
    t = compute_travel_times(world, s.edge_mask)
    a = s.activity
    d_one = compute_demand(world, a, t)
    d_two = compute_demand(world, 2.0 * a, t)
    np.testing.assert_allclose(d_two, 4.0 * d_one, rtol=1e-9)
