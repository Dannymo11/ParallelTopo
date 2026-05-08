"""Sanity tests for the CPU simulator dynamics.

These tests are **deliberately RED on the step-2 no-op stub.** They become
GREEN when task #8 lands real dynamics. Concretely they test:

1. *Walking-only floor.* A no-op rollout still produces non-zero return,
   because walking accessibility itself contributes welfare. A simulator
   that always returns zero reward — exactly what the stub does — fails
   this test, which is the desired signal.

2. *Edges help.* A hand-built corridor of transit edges along the central
   row strictly improves return over a no-op rollout on the same world.
   Tests the action → reward feedback is sign-correct.

3. *Greedy beats random.* A one-step-greedy policy strictly outperforms
   uniform-random over multiple seeds. Tests that the reward landscape has
   enough signal that a trivial heuristic can find better edges than
   noise.

These are *trajectory-level* sanity tests, not unit tests of any specific
component (those will live in `test_dynamics.py` once dynamics land). The
philosophy: even if every individual computation in the simulator is wrong
in some compensating way, getting all three of these to pass implies the
simulator is at least directionally correct.

Why the tests are checked in BEFORE the implementation: this is the
checkpoint's "evaluation code that runs and can definitively say 'empty
images / white noise / zero reward' is not successful" requirement,
literally.
"""

from __future__ import annotations

import numpy as np
import pytest

from topograph.policies import (
    OneStepGreedyPolicy,
    RandomLegalPolicy,
    ScriptedPolicy,
)
from topograph.sim_cpu import (
    GridCityConfig,
    WorldConfig,
    find_edge_index,
    make_world,
    reset,
    run_episode,
)

# Sanity tests are expected to fail until task #8 lands. Mark the whole
# module so `pytest` reports them as XFAIL rather than FAILED — the suite
# still runs cleanly while the implementation is in flight.
pytestmark = pytest.mark.xfail(
    reason="Sanity tests are RED on the step-2 stub; task #8 turns them GREEN.",
    strict=True,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def world_5x5() -> WorldConfig:
    """A small enough world that a corridor along the middle row fits in
    the horizon (5 steps, 4 corridor edges + 1 no-op)."""
    return make_world(
        seed=0,
        cfg=GridCityConfig(
            grid_shape=(5, 5), horizon=5, candidate_k_nearest=4, initial_budget=50.0
        ),
    )


@pytest.fixture
def world_default() -> WorldConfig:
    """The 'representative' world: 10x10, horizon 15. Used by sanity 1 & 3."""
    return make_world(seed=0, cfg=GridCityConfig())


def _no_op_policy(state, rng):
    del rng
    return state.world.no_op_action


# ---------------------------------------------------------------------------
# Sanity 1 — walking-only floor
# ---------------------------------------------------------------------------


def test_empty_network_rollout_has_nonzero_return(world_default: WorldConfig) -> None:
    traj = run_episode(reset(world_default), _no_op_policy, rng=0)
    assert traj.final_state.done
    # No-op rollout never activates an edge.
    assert not traj.final_state.edge_mask.any()
    # Walking accessibility itself is non-zero, so the welfare-derived
    # reward is non-zero.
    assert traj.episode_return > 0.0
    # Welfare is non-negative each step (lower bound: zero accessibility).
    assert (traj.rewards >= 0.0).all()


# ---------------------------------------------------------------------------
# Sanity 2 — central corridor strictly beats empty
# ---------------------------------------------------------------------------


def _central_corridor_actions(world: WorldConfig) -> list[int]:
    """Action sequence that activates edges along the central row.

    For a 5x5 grid, the central row is r=2, with zones {10, 11, 12, 13, 14}
    and corridor edges (10,11), (11,12), (12,13), (13,14). We pad with
    no-ops so the sequence length matches the horizon.
    """
    rows, cols = world.zone_positions.shape[0] // 5, 5  # 5x5 fixture only
    del rows  # unused; here for clarity
    middle_row = 2
    zones = [middle_row * cols + c for c in range(cols)]
    edges = [(zones[i], zones[i + 1]) for i in range(cols - 1)]
    return [find_edge_index(world, a, b) for (a, b) in edges]


def test_central_corridor_beats_empty(world_5x5: WorldConfig) -> None:
    empty = run_episode(reset(world_5x5), _no_op_policy, rng=0)

    plan = _central_corridor_actions(world_5x5)
    corridor = run_episode(reset(world_5x5), ScriptedPolicy(plan), rng=0)

    # All planned edges should actually be active in the final state.
    for edge_idx in plan:
        assert corridor.final_state.edge_mask[edge_idx], (
            f"corridor edge {edge_idx} was not activated"
        )

    # Adding a non-trivial corridor should strictly improve return.
    assert corridor.episode_return > empty.episode_return


# ---------------------------------------------------------------------------
# Sanity 3 — greedy beats random
# ---------------------------------------------------------------------------


def test_greedy_beats_random_on_average(world_default: WorldConfig) -> None:
    n_seeds = 5
    greedy_returns = np.empty(n_seeds, dtype=np.float64)
    random_returns = np.empty(n_seeds, dtype=np.float64)

    for s in range(n_seeds):
        greedy_returns[s] = run_episode(
            reset(world_default), OneStepGreedyPolicy(), rng=s
        ).episode_return
        random_returns[s] = run_episode(
            reset(world_default), RandomLegalPolicy(), rng=s
        ).episode_return

    # Strict mean improvement, not "no worse" — the greedy lookahead should
    # find clearly better edges than uniform random on a non-trivial reward.
    assert greedy_returns.mean() > random_returns.mean()
