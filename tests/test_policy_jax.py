"""Vectorized random policy + on-device rollout driver correctness.

Covers the GPU `policy.py`:

* `valid_action_mask_batched` matches the CPU `valid_action_mask` per env.
* the random sampler only ever returns legal actions, and is ~uniform over
  the legal set.
* `rollout_random` is deterministic in its PRNG key, runs end-to-end, and
  actually builds networks.
* aggregate reward parity: the mean episode return of the GPU random policy
  matches the CPU `RandomLegalPolicy` mean return on the same workload
  (this is the end-to-end check that the assembled GPU simulator + policy
  reproduce the CPU simulator's reward scale, since the per-step dynamics
  are equivalence-tested elsewhere).

Skipped if JAX isn't installed (a `[gpu]` extra).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from topograph.policies import RandomLegalPolicy
from topograph.sim_cpu import (
    GridCityConfig,
    make_world,
    reset,
    run_episode,
    valid_action_mask,
)
from topograph.sim_gpu import (
    episode_returns,
    random_legal_action_batched,
    reset_batched,
    rollout_random,
    valid_action_mask_batched,
    world_arrays_from_config,
)

RTOL = 2e-3
ATOL = 2e-3


def _f32(x):
    return jnp.asarray(np.asarray(x, dtype=np.float32))


# ---------------------------------------------------------------------------
# Legality mask parity
# ---------------------------------------------------------------------------


def test_valid_action_mask_matches_cpu():
    world = make_world(0, GridCityConfig(grid_shape=(10, 10)))
    rng = np.random.default_rng(0)
    n_envs = 8

    cpu_masks, states = [], []
    for i in range(n_envs):
        mask = rng.random(world.n_candidate_edges) < (0.1 * i)
        budget = float(world.initial_budget - 6 * i)  # some envs near/at zero budget
        s = dataclasses.replace(
            reset(world), edge_mask=mask.astype(np.bool_), budget_remaining=budget
        )
        states.append(s)
        cpu_masks.append(valid_action_mask(s))

    gpu = np.asarray(
        valid_action_mask_batched(
            jnp.asarray(np.stack([s.edge_mask for s in states])),
            _f32(np.array([s.budget_remaining for s in states])),
            _f32(world.cost_per_edge),
        )
    )
    np.testing.assert_array_equal(gpu, np.stack(cpu_masks))


# ---------------------------------------------------------------------------
# Sampler: legality + approximate uniformity
# ---------------------------------------------------------------------------


def test_random_policy_only_samples_legal():
    # A legality mask where only a known subset is legal; sample many times
    # and assert every draw is legal and all legal actions appear.
    A = 12
    legal = np.zeros(A, dtype=bool)
    legal_idx = np.array([1, 4, 7, 11])  # 11 is the (always-legal) no-op slot
    legal[legal_idx] = True

    n = 4000
    keys = jax.random.split(jax.random.PRNGKey(0), n)
    legal_batch = jnp.broadcast_to(jnp.asarray(legal), (n, A))
    actions = np.asarray(random_legal_action_batched(keys, legal_batch))

    assert set(np.unique(actions)).issubset(set(legal_idx.tolist()))
    assert set(np.unique(actions)) == set(legal_idx.tolist())  # all legal seen
    # Roughly uniform over the 4 legal actions (loose band).
    counts = np.array([(actions == k).sum() for k in legal_idx])
    fracs = counts / n
    assert np.all(np.abs(fracs - 0.25) < 0.05)


# ---------------------------------------------------------------------------
# rollout_random: shape, determinism, builds a network
# ---------------------------------------------------------------------------


def test_rollout_random_runs_and_is_deterministic():
    world = make_world(0, GridCityConfig(grid_shape=(10, 10)))
    wa = world_arrays_from_config(world)
    B = 16
    state = reset_batched(wa, B, _f32(world.initial_activity), world.initial_budget)

    key = jax.random.PRNGKey(0)
    final, rewards, actions = rollout_random(state, key, wa, horizon=world.horizon)

    assert rewards.shape == (world.horizon, B)
    assert actions.shape == (world.horizon, B)
    assert np.all(np.isfinite(np.asarray(rewards)))
    # Each env built at least one edge.
    assert np.all(np.asarray(final.edge_mask).sum(axis=1) > 0)

    # Same key -> identical; different key -> different actions.
    final2, rewards2, actions2 = rollout_random(state, key, wa, horizon=world.horizon)
    np.testing.assert_array_equal(np.asarray(actions), np.asarray(actions2))
    np.testing.assert_array_equal(np.asarray(rewards), np.asarray(rewards2))

    _, _, actions3 = rollout_random(
        state, jax.random.PRNGKey(1), wa, horizon=world.horizon
    )
    assert not np.array_equal(np.asarray(actions), np.asarray(actions3))


# ---------------------------------------------------------------------------
# Aggregate reward parity with the CPU random policy
# ---------------------------------------------------------------------------


def test_random_rollout_return_matches_cpu_distribution():
    """GPU random-policy mean return ~= CPU RandomLegalPolicy mean return.

    Both run the same workload; the dynamics are equivalence-tested, so the
    reward *scale* should agree. RNG differs (JAX vs NumPy), so this compares
    distribution means, not trajectories. Tolerance is a generous relative
    band that comfortably covers the sampling spread at these sample sizes.
    """
    cfg = GridCityConfig(grid_shape=(10, 10))
    world = make_world(0, cfg)

    # CPU reference: mean return over several seeds.
    cpu_policy = RandomLegalPolicy()
    cpu_returns = np.array(
        [run_episode(reset(world), cpu_policy, rng=s).episode_return for s in range(12)]
    )
    cpu_mean = cpu_returns.mean()

    # GPU: mean per-env return over a batch.
    wa = world_arrays_from_config(world)
    B = 128
    state = reset_batched(wa, B, _f32(world.initial_activity), world.initial_budget)
    _, rewards, _ = rollout_random(state, jax.random.PRNGKey(0), wa, horizon=world.horizon)
    gpu_returns = np.asarray(episode_returns(rewards))
    gpu_mean = gpu_returns.mean()

    rel = abs(gpu_mean - cpu_mean) / cpu_mean
    assert rel < 0.05, (
        f"GPU random-return mean {gpu_mean:.3f} vs CPU {cpu_mean:.3f} "
        f"(relative diff {rel:.3%}) — exceeds 5% parity band"
    )
    # Non-trivial: both clearly above zero (walking-floor sanity).
    assert gpu_mean > 0 and cpu_mean > 0
