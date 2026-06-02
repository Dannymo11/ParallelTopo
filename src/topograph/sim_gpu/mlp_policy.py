"""On-device MLP policy + end-to-end rollout/training drivers (M4).

This is the M4 counterpart to the random-policy driver in `policy.py`. It
swaps the uniform-random sampler for a small MLP forward pass under the
*same* `lax.scan` rollout, so the policy and the simulator stay on one
device behind one compiled program — the PureJaxRL-style design the M4
spec calls for, chosen over an SB3 `VectorEnv` precisely to avoid the
host-device transfer per step that would otherwise be the thing we
accidentally measure.

What this module answers (RQ3): does the simulator's throughput survive
once a policy is in the loop, or does the policy / host boundary dominate?
Three drivers let `scripts/m4_train_throughput.py` separate the costs:

  * `rollout_random`  (in policy.py)  — simulator only, the M5 Fig-1 number
  * `rollout_mlp`                      — simulator + MLP *inference* per step
  * `train_step`                       — simulator + inference + REINFORCE
                                         backward + an SGD update

The agent is deliberately NOT trained to do anything useful (an explicit
M4 non-goal): the policy exists to consume the simulator at a realistic
rate so the boundary cost is real. Correctness here means (a) every sampled
action is legal, (b) determinism under a fixed key, and (c) gradients flow
to every parameter — not policy quality.

The MLP is hand-rolled (no flax/haiku/optax) so the Modal image stays the
proven `jax[cuda12]==0.5.3`-only build with no extra dependencies.

fp32 throughout, matching the rest of sim_gpu.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .apsp import DEFAULT_K_ITERATIONS
from .policy import valid_action_mask_batched
from .step import GPUState, WorldArrays, step_batched

# Logits for illegal actions are pushed to this (finite, not -inf, so
# log_softmax never produces a NaN — the no-op action is always legal, so at
# least one entry per row is finite regardless).
_NEG_INF = -1.0e9

DEFAULT_HIDDEN: int = 64


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


class MLPParams(NamedTuple):
    """A 2-hidden-layer MLP as an explicit PyTree (JAX threads it through
    `jit`/`grad`/`vmap` with no registration)."""

    w1: jax.Array
    b1: jax.Array
    w2: jax.Array
    b2: jax.Array
    w3: jax.Array
    b3: jax.Array


def feature_dim(world: WorldArrays) -> int:
    """Length of the per-env feature vector this policy consumes."""
    n_zones = world.walk_times.shape[0]
    n_candidate = world.cost_per_edge.shape[0]
    return n_zones + n_candidate + 2  # activity, edge_mask, budget frac, step frac


def n_actions(world: WorldArrays) -> int:
    """Action-space size: one per candidate edge plus the no-op."""
    return int(world.cost_per_edge.shape[0]) + 1


def init_mlp_params(
    key: jax.Array, in_dim: int, out_dim: int, hidden: int = DEFAULT_HIDDEN
) -> MLPParams:
    """He-initialized weights, zero biases, fp32."""
    k1, k2, k3 = jax.random.split(key, 3)

    def he(k, shape):
        fan_in = shape[0]
        return jax.random.normal(k, shape, dtype=jnp.float32) * jnp.sqrt(2.0 / fan_in)

    return MLPParams(
        w1=he(k1, (in_dim, hidden)),
        b1=jnp.zeros((hidden,), jnp.float32),
        w2=he(k2, (hidden, hidden)),
        b2=jnp.zeros((hidden,), jnp.float32),
        w3=he(k3, (hidden, out_dim)),
        b3=jnp.zeros((out_dim,), jnp.float32),
    )


# ---------------------------------------------------------------------------
# Features and forward pass
# ---------------------------------------------------------------------------


def state_features(
    state: GPUState, horizon: int, budget0: float
) -> jax.Array:
    """`(B, F)` bounded feature matrix from a batched state.

    Concatenates per-env: land-use activity `(N,)`, the edge-existence mask
    as floats `(E,)`, remaining-budget fraction, and episode progress. All
    terms are O(1) so the untrained MLP stays numerically tame.
    """
    budget_frac = (state.budget / jnp.float32(budget0))[:, None]
    step_frac = (state.step.astype(jnp.float32) / jnp.float32(horizon))[:, None]
    return jnp.concatenate(
        [state.activity, state.edge_mask.astype(jnp.float32), budget_frac, step_frac],
        axis=-1,
    )


def mlp_forward(params: MLPParams, feats: jax.Array) -> jax.Array:
    """`(B, F)` features -> `(B, A)` action logits."""
    h = jax.nn.relu(feats @ params.w1 + params.b1)
    h = jax.nn.relu(h @ params.w2 + params.b2)
    return h @ params.w3 + params.b3


def masked_logits(logits: jax.Array, legal: jax.Array) -> jax.Array:
    """Set illegal-action logits to a large negative value."""
    return jnp.where(legal, logits, jnp.float32(_NEG_INF))


# ---------------------------------------------------------------------------
# Rollout core (returns log-probs; used by both inference and training)
# ---------------------------------------------------------------------------


def _rollout_collect(
    params: MLPParams,
    state: GPUState,
    key: jax.Array,
    world: WorldArrays,
    horizon: int,
    k_iterations: int,
    budget0: float,
):
    """Run `horizon` steps under the MLP policy on a batch of B envs.

    Returns `(final_state, rewards (T, B), actions (T, B), logps (T, B))`.
    Not jitted: it is the shared body for the jitted inference driver and
    the differentiated REINFORCE loss.
    """
    batch_size = state.budget.shape[0]

    def body(carry, _):
        st, k = carry
        k, sub = jax.random.split(k)
        feats = state_features(st, horizon, budget0)
        legal = valid_action_mask_batched(st.edge_mask, st.budget, world.cost_per_edge)
        logits = masked_logits(mlp_forward(params, feats), legal)
        actions = jax.random.categorical(sub, logits, axis=-1).astype(jnp.int32)
        logp = jnp.take_along_axis(
            jax.nn.log_softmax(logits, axis=-1), actions[:, None], axis=-1
        )[:, 0]
        next_st, rewards = step_batched(
            st, actions, world, horizon=horizon, k_iterations=k_iterations
        )
        return (next_st, k), (rewards, actions, logp)

    (final_state, _), (rewards, actions, logps) = jax.lax.scan(
        body, (state, key), xs=None, length=horizon
    )
    return final_state, rewards, actions, logps


@partial(jax.jit, static_argnames=("horizon", "k_iterations", "budget0"))
def rollout_mlp(
    params: MLPParams,
    state: GPUState,
    key: jax.Array,
    world: WorldArrays,
    horizon: int,
    k_iterations: int = DEFAULT_K_ITERATIONS,
    budget0: float = 1.0,
) -> tuple[GPUState, jax.Array, jax.Array]:
    """Inference-only rollout: simulator + MLP forward pass, fully on-device.

    The end-to-end *inference* throughput driver (no gradient, no host
    transfer). Returns `(final_state, rewards (T, B), actions (T, B))`.
    """
    final_state, rewards, actions, _logps = _rollout_collect(
        params, state, key, world, horizon, k_iterations, budget0
    )
    return final_state, rewards, actions


# ---------------------------------------------------------------------------
# REINFORCE training step (forward rollout + policy-gradient update)
# ---------------------------------------------------------------------------


def reinforce_loss(
    params: MLPParams,
    state: GPUState,
    key: jax.Array,
    world: WorldArrays,
    horizon: int,
    k_iterations: int,
    budget0: float,
) -> jax.Array:
    """Scalar REINFORCE loss with a batch-mean baseline.

    `loss = -mean_b[ (R_b - b) * sum_t logp_{t,b} ]`, where the return
    `R_b` (and hence the advantage) is treated as a constant — gradients
    flow only through the log-probabilities, the standard score-function
    estimator. Reward magnitude is irrelevant to M4: this exists to put a
    realistic forward+backward through the simulator, not to learn.
    """
    _final, rewards, _actions, logps = _rollout_collect(
        params, state, key, world, horizon, k_iterations, budget0
    )
    returns = jnp.sum(rewards, axis=0)                  # (B,)
    advantage = jax.lax.stop_gradient(returns - jnp.mean(returns))
    logp_sum = jnp.sum(logps, axis=0)                   # (B,)
    return -jnp.mean(advantage * logp_sum)


@partial(jax.jit, static_argnames=("horizon", "k_iterations", "budget0", "lr"))
def train_step(
    params: MLPParams,
    state: GPUState,
    key: jax.Array,
    world: WorldArrays,
    horizon: int,
    k_iterations: int = DEFAULT_K_ITERATIONS,
    budget0: float = 1.0,
    lr: float = 1e-3,
) -> tuple[MLPParams, jax.Array]:
    """One full training iteration: rollout -> REINFORCE loss -> SGD update.

    Hand-rolled SGD (no optax). Returns `(new_params, loss)`. This is the
    driver whose env-steps/sec is the honest end-to-end *training*
    throughput — forward rollout plus the backward pass and parameter
    update, all under one `jit`.
    """
    loss, grads = jax.value_and_grad(reinforce_loss)(
        params, state, key, world, horizon, k_iterations, budget0
    )
    new_params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)
    return new_params, loss


__all__ = [
    "MLPParams",
    "DEFAULT_HIDDEN",
    "feature_dim",
    "n_actions",
    "init_mlp_params",
    "state_features",
    "mlp_forward",
    "masked_logits",
    "rollout_mlp",
    "reinforce_loss",
    "train_step",
]
