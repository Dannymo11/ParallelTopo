"""Unit tests for the three baseline policies.

These tests check policy *behavior* (legality, reproducibility, replay) and
do not depend on the simulator's reward dynamics. They pass on the stub
`step` from step 2 and must continue to pass after task #8 lands real
dynamics.
"""

from __future__ import annotations

import numpy as np
import pytest

from topograph.policies import OneStepGreedyPolicy, RandomLegalPolicy, ScriptedPolicy
from topograph.sim_cpu import (
    GridCityConfig,
    WorldConfig,
    find_edge_index,
    make_world,
    reset,
    run_episode,
    valid_action_mask,
)


@pytest.fixture
def world() -> WorldConfig:
    return make_world(seed=0, cfg=GridCityConfig(grid_shape=(4, 4), horizon=5))


# ---------------------------------------------------------------------------
# RandomLegalPolicy
# ---------------------------------------------------------------------------


def test_random_policy_returns_legal_actions(world: WorldConfig) -> None:
    policy = RandomLegalPolicy()
    rng = np.random.default_rng(0)
    s = reset(world)
    for _ in range(world.horizon):
        a = policy(s, rng)
        assert valid_action_mask(s)[a], f"action {a} is not legal"
        s, *_ = __step(s, a)


def test_random_policy_is_reproducible_under_seed(world: WorldConfig) -> None:
    policy_a = RandomLegalPolicy()
    policy_b = RandomLegalPolicy()
    a = run_episode(reset(world), policy_a, rng=42)
    b = run_episode(reset(world), policy_b, rng=42)
    np.testing.assert_array_equal(a.actions, b.actions)


# ---------------------------------------------------------------------------
# ScriptedPolicy
# ---------------------------------------------------------------------------


def test_scripted_policy_replays_in_order(world: WorldConfig) -> None:
    plan = [0, 1, world.no_op_action, 2, world.no_op_action]
    policy = ScriptedPolicy(plan)
    traj = run_episode(reset(world), policy, rng=0)
    np.testing.assert_array_equal(traj.actions, np.asarray(plan, dtype=np.int64))


def test_scripted_policy_fallback_to_no_op_when_exhausted(world: WorldConfig) -> None:
    short_plan = [0, 1]  # shorter than horizon=5
    policy = ScriptedPolicy(short_plan)
    traj = run_episode(reset(world), policy, rng=0)
    expected = np.array(
        short_plan + [world.no_op_action] * (world.horizon - len(short_plan)),
        dtype=np.int64,
    )
    np.testing.assert_array_equal(traj.actions, expected)


def test_scripted_policy_uses_explicit_fallback(world: WorldConfig) -> None:
    policy = ScriptedPolicy([0], fallback=1)
    traj = run_episode(reset(world), policy, rng=0)
    expected = np.array([0] + [1] * (world.horizon - 1), dtype=np.int64)
    np.testing.assert_array_equal(traj.actions, expected)


def test_scripted_policy_is_stateless(world: WorldConfig) -> None:
    """Re-using the same policy instance across runs must replay identically."""
    plan = [0, 1, 2]
    policy = ScriptedPolicy(plan)
    a = run_episode(reset(world), policy, rng=0).actions
    b = run_episode(reset(world), policy, rng=0).actions
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# OneStepGreedyPolicy (stub-era behavior: ties resolve to no-op)
# ---------------------------------------------------------------------------


def test_greedy_policy_returns_legal_actions(world: WorldConfig) -> None:
    policy = OneStepGreedyPolicy()
    rng = np.random.default_rng(0)
    s = reset(world)
    for _ in range(world.horizon):
        a = policy(s, rng)
        assert valid_action_mask(s)[a]
        s, *_ = __step(s, a)


def test_greedy_policy_is_deterministic(world: WorldConfig) -> None:
    policy = OneStepGreedyPolicy()
    a = run_episode(reset(world), policy, rng=0).actions
    b = run_episode(reset(world), policy, rng=99).actions  # rng must be ignored
    np.testing.assert_array_equal(a, b)


def test_greedy_picks_lowest_index_tie_break(world: WorldConfig) -> None:
    """Under the step-2 stub, every action returns reward 0.

    Greedy uses strict ``>`` comparison, so among tied legal actions it
    picks the *first* one iterated. Combined with the stub's frozen
    edge_mask / budget (no state changes step-to-step), the same lowest
    legal action is selected every step.

    Once task #8 lands real dynamics this test will need updating — by
    then ties will be much rarer and the lowest-index path won't be the
    interesting one. The test stays as a stub-era regression for now.
    """
    policy = OneStepGreedyPolicy()
    traj = run_episode(reset(world), policy, rng=0)
    # Under stub dynamics every step has the same legal set, so greedy
    # picks the same lowest legal index each time.
    assert (traj.actions == traj.actions[0]).all()


# ---------------------------------------------------------------------------
# find_edge_index helper
# ---------------------------------------------------------------------------


def test_find_edge_index_undirected_lookup(world: WorldConfig) -> None:
    e0 = world.candidate_edges[0]
    a, b = int(e0[0]), int(e0[1])
    assert find_edge_index(world, a, b) == 0
    assert find_edge_index(world, b, a) == 0


def test_find_edge_index_raises_for_missing(world: WorldConfig) -> None:
    # In a 4x4 grid with k=8 nearest, opposite-corner pairs are typically
    # outside the candidate set.
    n = world.n_zones
    with pytest.raises(KeyError):
        find_edge_index(world, 0, n - 1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def __step(state, action):
    """Local alias for `step` to keep the test imports tidy."""
    from topograph.sim_cpu import step

    return step(state, int(action))
