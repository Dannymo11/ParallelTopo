"""On-device MLP policy + end-to-end driver correctness (M4).

Covers `sim_gpu/mlp_policy.py`. The agent is not trained to do anything
useful (an M4 non-goal), so these tests pin *mechanism*, not policy
quality:

* feature / logit / param shapes are what the rollout expects;
* `rollout_mlp` only ever samples LEGAL actions, runs end-to-end, builds
  networks, and is deterministic in its PRNG key;
* `train_step` produces a finite loss, changes every parameter (gradients
  actually flow through the scanned rollout to the policy net), and is
  itself deterministic in its key.

Skipped if JAX isn't installed (a `[gpu]` extra).
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from topograph.sim_cpu import GridCityConfig, make_world, valid_action_mask, reset
from topograph.sim_gpu import (
    DEFAULT_HIDDEN,
    feature_dim,
    init_mlp_params,
    mlp_forward,
    n_actions,
    reset_batched,
    rollout_mlp,
    state_features,
    train_step,
    valid_action_mask_batched,
    world_arrays_from_config,
)


def _f32(x):
    return jnp.asarray(np.asarray(x, dtype=np.float32))


def _setup(grid=(10, 10), B=16, seed=0):
    world = make_world(seed, GridCityConfig(grid_shape=grid))
    wa = world_arrays_from_config(world)
    budget0 = float(world.initial_budget)
    state = reset_batched(wa, B, _f32(world.initial_activity), budget0)
    params = init_mlp_params(
        jax.random.PRNGKey(seed + 1),
        in_dim=feature_dim(wa),
        out_dim=n_actions(wa),
        hidden=DEFAULT_HIDDEN,
    )
    return world, wa, state, params, budget0, B


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------


def test_feature_and_logit_shapes():
    world, wa, state, params, budget0, B = _setup()
    feats = state_features(state, horizon=world.horizon, budget0=budget0)
    assert feats.shape == (B, feature_dim(wa))
    assert np.all(np.isfinite(np.asarray(feats)))

    logits = mlp_forward(params, feats)
    assert logits.shape == (B, n_actions(wa))
    # n_actions == candidate edges + 1 (the no-op).
    assert n_actions(wa) == world.n_candidate_edges + 1


# ---------------------------------------------------------------------------
# rollout_mlp: legality, runs, builds networks, determinism
# ---------------------------------------------------------------------------


def test_rollout_mlp_samples_only_legal_actions():
    world, wa, state, params, budget0, B = _setup()
    key = jax.random.PRNGKey(0)
    final, rewards, actions = rollout_mlp(
        params, state, key, wa, horizon=world.horizon, budget0=budget0
    )

    assert rewards.shape == (world.horizon, B)
    assert actions.shape == (world.horizon, B)
    assert np.all(np.isfinite(np.asarray(rewards)))
    # Every env built at least one edge under the (untrained) policy.
    assert np.all(np.asarray(final.edge_mask).sum(axis=1) > 0)

    # Re-derive legality at each step from a fresh replay and confirm every
    # sampled action was legal in the state it was taken from.
    actions_np = np.asarray(actions)
    st = state
    k = key
    for t in range(world.horizon):
        k, sub = jax.random.split(k)
        legal = np.asarray(
            valid_action_mask_batched(st.edge_mask, st.budget, wa.cost_per_edge)
        )
        a_t = actions_np[t]
        assert legal[np.arange(B), a_t].all(), f"illegal action sampled at step {t}"
        # advance with the recorded actions to reach the next state
        from topograph.sim_gpu import step_batched
        st, _ = step_batched(st, jnp.asarray(a_t), wa, horizon=world.horizon)


def test_rollout_mlp_is_deterministic_in_key():
    world, wa, state, params, budget0, B = _setup()
    key = jax.random.PRNGKey(3)
    _f1, r1, a1 = rollout_mlp(params, state, key, wa, horizon=world.horizon, budget0=budget0)
    _f2, r2, a2 = rollout_mlp(params, state, key, wa, horizon=world.horizon, budget0=budget0)
    np.testing.assert_array_equal(np.asarray(a1), np.asarray(a2))
    np.testing.assert_array_equal(np.asarray(r1), np.asarray(r2))

    _f3, _r3, a3 = rollout_mlp(
        params, state, jax.random.PRNGKey(4), wa, horizon=world.horizon, budget0=budget0
    )
    assert not np.array_equal(np.asarray(a1), np.asarray(a3))


# ---------------------------------------------------------------------------
# train_step: finite loss, gradients reach every parameter, determinism
# ---------------------------------------------------------------------------


def test_train_step_updates_all_params():
    world, wa, state, params, budget0, B = _setup(B=32)
    key = jax.random.PRNGKey(0)
    new_params, loss = train_step(
        params, state, key, wa, horizon=world.horizon, budget0=budget0, lr=1e-2
    )
    assert np.isfinite(float(loss))

    # Every leaf must have moved — i.e. a gradient reached it through the
    # scanned rollout. (lr * grad == 0 only if grad is exactly 0, which would
    # mean the parameter never influenced any log-prob.)
    for old, new in zip(params, new_params):
        old_np, new_np = np.asarray(old), np.asarray(new)
        assert old_np.shape == new_np.shape
        assert not np.allclose(old_np, new_np), "a parameter received no gradient"


def test_train_step_is_deterministic_in_key():
    world, wa, state, params, budget0, B = _setup(B=16)
    key = jax.random.PRNGKey(7)
    p1, l1 = train_step(params, state, key, wa, horizon=world.horizon, budget0=budget0)
    p2, l2 = train_step(params, state, key, wa, horizon=world.horizon, budget0=budget0)
    np.testing.assert_allclose(float(l1), float(l2), rtol=0, atol=0)
    for a, b in zip(p1, p2):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
