"""JAX assembled-step correctness: matches the CPU simulator at fp32.

The milestone's headline M3 test: a vectorized rollout at B=1 reproduces
the CPU baseline's reward trajectory and final state up to fp32 tolerance.
Plus the supporting equivalence checks for the pieces the assembled step
adds on top of the already-tested dynamics components:

* `apply_action` — including the already-active / unaffordable / no-op
  silent-degradation paths.
* on-device `build_direct_distance` vs the CPU matrix builder.
* a batched per-env step vs per-env CPU `step` (cross-env leakage guard).
* the `freeze_mask` ablation actually freezes topology.

Travel times on the CPU come from scipy Floyd-Warshall; on the GPU from
K=5 min-plus matrix-squaring. The M2 sweep established these agree within
fp32 round-off, so a slightly looser tolerance than bit-identity is used
and accumulation across the 15-step horizon is expected to stay within it.

Skipped if JAX isn't installed (a `[gpu]` extra).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from topograph.sim_cpu import (
    GridCityConfig,
    make_world,
    reset,
    run_episode,
    step as cpu_step,
    valid_action_mask,
)
from topograph.sim_cpu.dynamics import _build_direct_distance_matrix
from topograph.sim_gpu import (
    DEFAULT_K_ITERATIONS,
    apply_action_batched,
    build_direct_distance_single,
    reset_batched,
    rollout_batched,
    step_batched,
    world_arrays_from_config,
)

RTOL = 2e-3
ATOL = 2e-3


def _f32(x):
    return jnp.asarray(np.asarray(x, dtype=np.float32))


def _edge_preferring_policy(state, rng):
    """Pick a random affordable inactive edge; no-op only if none exist.

    Drives real edge activations every step so the mask scatter + budget
    path is exercised. With the default budget (50) and ~unit edge costs, a
    15-step episode never exhausts the budget, so activation decisions are
    unaffected by fp32-vs-fp64 budget round-off and the masks stay identical
    between CPU and GPU.
    """
    mask = valid_action_mask(state)
    n = state.world.no_op_action
    edge_actions = np.flatnonzero(mask[:n])
    if edge_actions.size:
        return int(rng.choice(edge_actions))
    return int(n)


# ---------------------------------------------------------------------------
# Headline: B=1 full-rollout equivalence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("grid_shape", [(5, 5), (10, 10), (15, 15)])
def test_b1_rollout_matches_cpu(grid_shape):
    world = make_world(0, GridCityConfig(grid_shape=grid_shape))

    # CPU reference rollout, recording the action schedule + reward trace.
    traj = run_episode(reset(world), _edge_preferring_policy, rng=0)

    # Replay the SAME actions through the GPU rollout at B=1.
    wa = world_arrays_from_config(world)
    gstate = reset_batched(wa, 1, _f32(world.initial_activity), world.initial_budget)
    actions_tb = jnp.asarray(traj.actions.reshape(world.horizon, 1).astype(np.int32))
    final_state, rewards_tb = rollout_batched(
        gstate, actions_tb, wa, horizon=world.horizon
    )

    gpu_rewards = np.asarray(rewards_tb)[:, 0]
    np.testing.assert_allclose(
        gpu_rewards, traj.rewards, rtol=RTOL, atol=ATOL,
        err_msg=f"GPU reward trajectory diverges from CPU at grid={grid_shape}",
    )

    # Final state agreement.
    np.testing.assert_allclose(
        np.asarray(final_state.activity)[0], traj.final_state.activity,
        rtol=RTOL, atol=ATOL, err_msg="final activity mismatch",
    )
    np.testing.assert_array_equal(
        np.asarray(final_state.edge_mask)[0], traj.final_state.edge_mask,
    )
    np.testing.assert_allclose(
        float(np.asarray(final_state.budget)[0]), traj.final_state.budget_remaining,
        rtol=1e-4, atol=1e-2, err_msg="final budget mismatch",
    )
    # Sanity: the rollout actually built a network (mask path exercised).
    assert int(np.asarray(final_state.edge_mask)[0].sum()) > 0


# ---------------------------------------------------------------------------
# Batched per-env step vs per-env CPU step (cross-env leakage guard)
# ---------------------------------------------------------------------------


def test_step_batched_matches_per_env_cpu():
    world = make_world(0, GridCityConfig(grid_shape=(10, 10)))
    wa = world_arrays_from_config(world)
    n_envs = 6
    rng = np.random.default_rng(0)

    cpu_states, actions = [], []
    for i in range(n_envs):
        # Distinct per-env state: random pre-activated mask, scaled activity,
        # varied budget — so a whole-batch reduction anywhere would diverge.
        mask = rng.random(world.n_candidate_edges) < (0.1 * (i + 1))
        activity = (rng.uniform(0.2, 3.0, world.n_zones) * (i + 1)).astype(np.float64)
        budget = float(world.initial_budget - i)
        s = dataclasses.replace(
            reset(world),
            edge_mask=mask.astype(np.bool_),
            activity=activity,
            budget_remaining=budget,
        )
        cpu_states.append(s)
        # An affordable inactive edge if one exists, else no-op.
        vm = valid_action_mask(s)
        edges = np.flatnonzero(vm[: world.no_op_action])
        actions.append(int(edges[i % edges.size]) if edges.size else int(world.no_op_action))

    # Per-env CPU reference.
    cpu_next = [cpu_step(s, a) for s, a in zip(cpu_states, actions)]
    cpu_rewards = np.array([r for _, r, _, _ in cpu_next])
    cpu_activity = np.stack([ns.activity for ns, _, _, _ in cpu_next])
    cpu_mask = np.stack([ns.edge_mask for ns, _, _, _ in cpu_next])

    # Batched GPU step.
    from topograph.sim_gpu import GPUState
    gstate = GPUState(
        edge_mask=jnp.asarray(np.stack([s.edge_mask for s in cpu_states])),
        activity=_f32(np.stack([s.activity for s in cpu_states])),
        budget=_f32(np.array([s.budget_remaining for s in cpu_states])),
        step=jnp.asarray(np.array([s.step for s in cpu_states], dtype=np.int32)),
        done=jnp.asarray(np.array([s.done for s in cpu_states])),
    )
    next_state, rewards = step_batched(
        gstate, jnp.asarray(np.array(actions, dtype=np.int32)), wa, horizon=world.horizon
    )

    np.testing.assert_allclose(np.asarray(rewards), cpu_rewards, rtol=RTOL, atol=ATOL,
                               err_msg="batched step rewards diverge per env")
    np.testing.assert_allclose(np.asarray(next_state.activity), cpu_activity,
                               rtol=RTOL, atol=ATOL, err_msg="batched step activity diverges")
    np.testing.assert_array_equal(np.asarray(next_state.edge_mask), cpu_mask)


# ---------------------------------------------------------------------------
# apply_action — all four CPU branches
# ---------------------------------------------------------------------------


def test_apply_action_matches_cpu_all_branches():
    from topograph.sim_cpu.dynamics import apply_action as cpu_apply

    world = make_world(0, GridCityConfig(grid_shape=(10, 10)))
    base = reset(world)
    # Pre-activate edge 0 and set a tight budget so some edges are unaffordable.
    mask0 = np.zeros(world.n_candidate_edges, dtype=np.bool_)
    mask0[0] = True
    tight = dataclasses.replace(base, edge_mask=mask0, budget_remaining=0.5)

    # action 0 -> already active (no-op); action 1 -> maybe unaffordable;
    # no-op action -> no-op; an affordable edge under full budget -> activate.
    full = dataclasses.replace(base, budget_remaining=world.initial_budget)
    cases = [
        (tight, 0),                      # already active
        (tight, 1),                      # likely unaffordable at budget 0.5
        (tight, world.no_op_action),     # explicit no-op
        (full, 2),                       # affordable activation
    ]
    states = [s for s, _ in cases]
    acts = [a for _, a in cases]

    cpu = [cpu_apply(s, a) for s, a in cases]
    cpu_masks = np.stack([m for m, _, _ in cpu])
    cpu_budgets = np.array([b for _, b, _ in cpu])
    cpu_did = np.array([d for _, _, d in cpu])

    new_mask, new_budget, did = apply_action_batched(
        jnp.asarray(np.stack([s.edge_mask for s in states])),
        _f32(np.array([s.budget_remaining for s in states])),
        jnp.asarray(np.array(acts, dtype=np.int32)),
        _f32(world.cost_per_edge),
    )
    np.testing.assert_array_equal(np.asarray(new_mask), cpu_masks)
    np.testing.assert_allclose(np.asarray(new_budget), cpu_budgets, rtol=1e-4, atol=1e-3)
    np.testing.assert_array_equal(np.asarray(did), cpu_did)


# ---------------------------------------------------------------------------
# On-device direct-distance construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("density", [0.0, 0.3, 1.0])
def test_build_direct_distance_matches_cpu(density):
    world = make_world(0, GridCityConfig(grid_shape=(10, 10)))
    rng = np.random.default_rng(7)
    mask = np.zeros(world.n_candidate_edges, dtype=np.bool_)
    if density > 0:
        k = max(1, int(round(density * world.n_candidate_edges)))
        mask[rng.choice(world.n_candidate_edges, size=k, replace=False)] = True

    cpu = _build_direct_distance_matrix(world, mask)
    wa = world_arrays_from_config(world)
    gpu = np.asarray(build_direct_distance_single(jnp.asarray(mask), wa))

    np.testing.assert_allclose(gpu, cpu, rtol=RTOL, atol=ATOL,
                               err_msg=f"direct-distance mismatch at density={density}")


# ---------------------------------------------------------------------------
# freeze_mask ablation
# ---------------------------------------------------------------------------


def test_freeze_mask_keeps_topology_static():
    world = make_world(0, GridCityConfig(grid_shape=(10, 10)))
    wa = world_arrays_from_config(world)
    gstate = reset_batched(wa, 1, _f32(world.initial_activity), world.initial_budget)

    # Even with edge-activating actions, frozen topology must not change the
    # mask or the budget across the rollout.
    actions_tb = jnp.zeros((world.horizon, 1), dtype=jnp.int32)  # all "activate edge 0"
    final_state, _ = rollout_batched(
        gstate, actions_tb, wa, horizon=world.horizon, freeze_mask=True
    )
    assert int(np.asarray(final_state.edge_mask)[0].sum()) == 0
    np.testing.assert_allclose(
        float(np.asarray(final_state.budget)[0]), world.initial_budget,
        rtol=0, atol=0, err_msg="freeze_mask must not spend budget",
    )
