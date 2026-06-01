"""Vectorized policies + on-device rollout driver (M3).

The GPU analog of `policies/random.RandomLegalPolicy` plus a `lax.scan`
rollout that advances a whole batch of environments under it, entirely
on-device. This is the driver that turns the assembled `step` (see
`step.py`) into full episodes — and the skeleton an MLP policy slots into
for M4 (replace `random_legal_action_batched` with a network forward pass
under the same scan).

Action legality mirrors the CPU `sim_cpu.api.valid_action_mask`: the no-op
(index `E`) is always legal; candidate edge `i` is legal iff it is
currently inactive AND its cost fits the remaining budget. The random
policy samples uniformly over the legal set, exactly like the CPU
`RandomLegalPolicy` — via a Gumbel-free "uniform-keys, mask-and-argmax"
trick that is branch-free and vmap/jit-friendly.

RNG is JAX-native (`jax.random`), so GPU rollouts won't reproduce a CPU
NumPy rollout action-for-action; correctness is instead pinned by (a) the
fixed-action-schedule equivalence in `test_step_jax.py`, (b) legality of
every sampled action, and (c) aggregate reward parity with the CPU random
policy (`test_policy_jax.py`).

fp32 throughout.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from .apsp import DEFAULT_K_ITERATIONS
from .step import GPUState, WorldArrays, step_batched


# ---------------------------------------------------------------------------
# Action legality   [mirrors sim_cpu.api.valid_action_mask]
# ---------------------------------------------------------------------------


def valid_action_mask_single(
    edge_mask: jax.Array, budget: jax.Array, cost_per_edge: jax.Array
) -> jax.Array:
    """`(E+1,)` boolean legality mask for one env (no-op at index E)."""
    affordable = cost_per_edge <= budget
    edge_legal = (~edge_mask) & affordable
    noop_legal = jnp.ones((1,), dtype=bool)  # no-op is always legal
    return jnp.concatenate([edge_legal, noop_legal])


@partial(jax.jit, static_argnames=())
def valid_action_mask_batched(
    edge_mask: jax.Array, budget: jax.Array, cost_per_edge: jax.Array
) -> jax.Array:
    """Batched legality: `(B, E)` masks / `(B,)` budgets -> `(B, E+1)` bool."""
    return jax.vmap(valid_action_mask_single, in_axes=(0, 0, None))(
        edge_mask, budget, cost_per_edge
    )


# ---------------------------------------------------------------------------
# Uniform-over-legal random policy
# ---------------------------------------------------------------------------


def random_legal_action_single(key: jax.Array, legal: jax.Array) -> jax.Array:
    """Sample uniformly among the True entries of `legal` (shape `(A,)`).

    Assigns iid uniform keys to legal actions and `-1` to illegal ones, then
    takes the argmax — every legal action is equally likely to hold the max,
    illegal actions can never win. No-op (always legal) is the guaranteed
    fallback. Returns an int32 action index.
    """
    u = jax.random.uniform(key, shape=(legal.shape[0],))
    u = jnp.where(legal, u, -1.0)
    return jnp.argmax(u).astype(jnp.int32)


@partial(jax.jit, static_argnames=())
def random_legal_action_batched(keys: jax.Array, legal: jax.Array) -> jax.Array:
    """Per-env sample: `(B, 2)` keys + `(B, A)` legality -> `(B,)` actions."""
    return jax.vmap(random_legal_action_single, in_axes=(0, 0))(keys, legal)


# ---------------------------------------------------------------------------
# On-device random-policy rollout
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnames=("horizon", "k_iterations", "freeze_mask"))
def rollout_random(
    state: GPUState,
    key: jax.Array,
    world: WorldArrays,
    horizon: int,
    k_iterations: int = DEFAULT_K_ITERATIONS,
    freeze_mask: bool = False,
) -> tuple[GPUState, jax.Array, jax.Array]:
    """Run `horizon` steps of the uniform-random-legal policy on a batch.

    `state` is a batched `GPUState` (B envs); `key` is a single PRNGKey.
    Each step splits the key, draws a fresh per-env subkey, samples a legal
    action per env from the *current* state, and advances via `step_batched`.
    Returns `(final_state, rewards (T, B), actions (T, B))`.
    """
    batch_size = state.budget.shape[0]

    def body(carry, _):
        st, k = carry
        k, sub = jax.random.split(k)
        keys = jax.random.split(sub, batch_size)
        legal = valid_action_mask_batched(st.edge_mask, st.budget, world.cost_per_edge)
        actions = random_legal_action_batched(keys, legal)
        next_st, rewards = step_batched(
            st, actions, world,
            horizon=horizon, k_iterations=k_iterations, freeze_mask=freeze_mask,
        )
        return (next_st, k), (rewards, actions)

    (final_state, _), (rewards, actions) = jax.lax.scan(
        body, (state, key), xs=None, length=horizon
    )
    return final_state, rewards, actions


def episode_returns(rewards: jax.Array) -> jax.Array:
    """Sum a `(T, B)` reward trace to per-env returns `(B,)`."""
    return jnp.sum(rewards, axis=0)


__all__ = [
    "valid_action_mask_single",
    "valid_action_mask_batched",
    "random_legal_action_single",
    "random_legal_action_batched",
    "rollout_random",
    "episode_returns",
]
