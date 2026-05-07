"""Contract tests for the simulator API surface (no dynamics yet).

These tests pin down shapes, types, and step-transition behavior. They pass
on the stub `step` (which advances the counter but does nothing else) and
must continue to pass when real dynamics land in task #8 — they are *not*
correctness tests for the dynamics, only for the API.

The point: a regression in the API surface (wrong shape, broken
determinism, off-by-one in horizon termination) gets caught here before it
contaminates the benchmark or the sanity tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from topograph.sim_cpu import (
    GridCityConfig,
    State,
    Trajectory,
    WorldConfig,
    make_world,
    reset,
    run_episode,
    step,
    valid_action_mask,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_cfg() -> GridCityConfig:
    """A small grid for fast tests."""
    return GridCityConfig(grid_shape=(4, 4), horizon=5, candidate_k_nearest=4)


@pytest.fixture
def world(small_cfg: GridCityConfig) -> WorldConfig:
    return make_world(seed=0, cfg=small_cfg)


# ---------------------------------------------------------------------------
# make_world: shape and structure
# ---------------------------------------------------------------------------


def test_make_world_shapes(world: WorldConfig) -> None:
    n = world.n_zones
    e = world.n_candidate_edges
    assert n == 16  # 4x4
    assert e > 0
    assert world.zone_positions.shape == (n, 2)
    assert world.walk_distances.shape == (n, n)
    assert world.walk_times.shape == (n, n)
    assert world.candidate_edges.shape == (e, 2)
    assert world.candidate_edge_weights.shape == (e,)
    assert world.cost_per_edge.shape == (e,)
    assert world.initial_activity.shape == (n,)


def test_make_world_walk_distances_are_a_metric(world: WorldConfig) -> None:
    d = world.walk_distances
    assert np.all(d >= 0)
    assert np.allclose(np.diag(d), 0.0)
    assert np.allclose(d, d.T)  # symmetry


def test_make_world_candidate_edges_are_undirected_unique(world: WorldConfig) -> None:
    edges = world.candidate_edges
    # i < j convention
    assert np.all(edges[:, 0] < edges[:, 1])
    # No duplicates
    as_tuples = {tuple(row) for row in edges.tolist()}
    assert len(as_tuples) == edges.shape[0]


def test_make_world_is_deterministic(small_cfg: GridCityConfig) -> None:
    a = make_world(seed=0, cfg=small_cfg)
    b = make_world(seed=0, cfg=small_cfg)
    np.testing.assert_array_equal(a.zone_positions, b.zone_positions)
    np.testing.assert_array_equal(a.candidate_edges, b.candidate_edges)
    np.testing.assert_array_equal(a.cost_per_edge, b.cost_per_edge)


# ---------------------------------------------------------------------------
# reset: initial state surface
# ---------------------------------------------------------------------------


def test_reset_initial_state(world: WorldConfig) -> None:
    s = reset(world)
    assert s.step == 0
    assert s.done is False
    assert s.budget_remaining == world.initial_budget
    assert s.edge_mask.dtype == np.bool_
    assert s.edge_mask.shape == (world.n_candidate_edges,)
    assert not s.edge_mask.any()
    np.testing.assert_array_equal(s.activity, world.initial_activity)
    # Defensive: the State should not alias the world's array.
    assert s.activity.base is not world.initial_activity


# ---------------------------------------------------------------------------
# step: counter, termination, validation
# ---------------------------------------------------------------------------


def test_step_advances_counter(world: WorldConfig) -> None:
    s = reset(world)
    s2, r, done, info = step(s, world.no_op_action)
    assert s2.step == 1
    assert done is False
    assert r == 0.0
    assert info.get("stub") is True


def test_step_terminates_at_horizon(world: WorldConfig) -> None:
    s = reset(world)
    for t in range(world.horizon - 1):
        s, _, done, _ = step(s, world.no_op_action)
        assert done is False, f"unexpected early termination at t={t}"
    s, _, done, _ = step(s, world.no_op_action)
    assert done is True
    assert s.step == world.horizon


def test_step_rejects_out_of_range_action(world: WorldConfig) -> None:
    s = reset(world)
    with pytest.raises(ValueError):
        step(s, world.action_dim)  # one past the no-op
    with pytest.raises(ValueError):
        step(s, -1)


def test_step_rejects_post_terminal(world: WorldConfig) -> None:
    s = reset(world)
    for _ in range(world.horizon):
        s, _, _, _ = step(s, world.no_op_action)
    assert s.done
    with pytest.raises(RuntimeError):
        step(s, world.no_op_action)


def test_step_returns_fresh_instance(world: WorldConfig) -> None:
    """Frozen-dataclass contract: each step returns a new State, not a mutated one."""
    s = reset(world)
    s2, *_ = step(s, world.no_op_action)
    assert s2 is not s
    assert s.step == 0  # original untouched
    assert s2.step == 1


# ---------------------------------------------------------------------------
# run_episode: trajectory shape and determinism
# ---------------------------------------------------------------------------


def _no_op_policy(state: State, rng: np.random.Generator) -> int:
    del rng
    return state.world.no_op_action


def _random_policy(state: State, rng: np.random.Generator) -> int:
    return int(rng.integers(0, state.world.action_dim))


def test_run_episode_shapes(world: WorldConfig) -> None:
    traj = run_episode(reset(world), _no_op_policy, rng=0)
    assert isinstance(traj, Trajectory)
    assert traj.actions.shape == (world.horizon,)
    assert traj.rewards.shape == (world.horizon,)
    assert traj.final_state.done is True
    assert traj.final_state.step == world.horizon


def test_run_episode_is_deterministic_under_seed(world: WorldConfig) -> None:
    a = run_episode(reset(world), _random_policy, rng=42)
    b = run_episode(reset(world), _random_policy, rng=42)
    np.testing.assert_array_equal(a.actions, b.actions)
    np.testing.assert_array_equal(a.rewards, b.rewards)


def test_run_episode_random_policy_varies_with_seed(world: WorldConfig) -> None:
    a = run_episode(reset(world), _random_policy, rng=1)
    b = run_episode(reset(world), _random_policy, rng=2)
    # At horizon=5 with a non-trivial action_dim there should almost
    # certainly be at least one differing action.
    assert not np.array_equal(a.actions, b.actions)


# ---------------------------------------------------------------------------
# valid_action_mask
# ---------------------------------------------------------------------------


def test_valid_action_mask_initially_all_legal(world: WorldConfig) -> None:
    s = reset(world)
    mask = valid_action_mask(s)
    assert mask.shape == (world.action_dim,)
    # Every candidate edge fits in the initial budget for the small fixture.
    assert mask.all()


def test_valid_action_mask_no_op_always_legal(world: WorldConfig) -> None:
    s = reset(world)
    mask = valid_action_mask(s)
    assert mask[world.no_op_action]
