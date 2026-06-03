"""JAX per-step dynamics for the batched GPU simulator (M3).

Ports the three CPU dynamics components that follow APSP in a `step` —
`compute_accessibility`, `update_activity`, `compute_demand` — to JAX, in
the `vmap`-over-the-batch-axis style established by `apsp.py`. Each
component is written as a **single-environment** function operating on
`(N,)` / `(N, N)` arrays plus the shared (non-batched) scalar world
parameters; the batched public entry points are `jax.vmap`'d over a
leading batch axis with the scalars held fixed (`in_axes=None`).

Correctness contract: at fp32 these reproduce the CPU baseline in
`sim_cpu/dynamics.py` within float32 round-off, asserted slot-by-slot by
`tests/test_dynamics_jax.py`. The batched tests deliberately use
per-environment-distinct inputs so that the highest-risk bug — doing the
`update_activity` normalization as a whole-batch reduction instead of a
per-env one — fails loudly instead of silently producing plausible
numbers.

"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# 3. Accessibility (exp-decay kernel)   [mirrors sim_cpu.dynamics.compute_accessibility]
# ---------------------------------------------------------------------------


def compute_accessibility_single(
    activity: jax.Array,
    travel_times: jax.Array,
    decay: jax.Array | float,
) -> jax.Array:
    """Per-zone accessibility under an exp-decay kernel, single environment.

        a_i = sum_j A_j * exp(-decay * t_ij)

    `activity` is `(N,)`, `travel_times` is `(N, N)`. The diagonal
    contributes `A_i` (t_ii = 0) exactly as on the CPU. Returns `(N,)`.
    """
    kernel = jnp.exp(-decay * travel_times)
    return kernel @ activity


@partial(jax.jit, static_argnames=())
def compute_accessibility_batched(
    activity: jax.Array,
    travel_times: jax.Array,
    decay: jax.Array | float,
) -> jax.Array:
    """Batched accessibility across `(B, N)` activity and `(B, N, N)` times.

    `decay` is a shared scalar (non-batched). Returns `(B, N)`.
    """
    return jax.vmap(compute_accessibility_single, in_axes=(0, 0, None))(
        activity, travel_times, decay
    )


# ---------------------------------------------------------------------------
# 4. Land-use update   [mirrors sim_cpu.dynamics.update_activity]
# ---------------------------------------------------------------------------


def update_activity_single(
    activity: jax.Array,
    accessibility: jax.Array,
    growth_rate: jax.Array | float,
) -> jax.Array:
    """Multiplicative land-use growth toward higher-accessibility zones.

        A_i' = A_i * (1 + growth_rate * (a_i / mean(a) - 1))

    then clip negatives and renormalize so total activity is preserved
    exactly. Single environment: `activity`, `accessibility` are `(N,)`,
    `mean(a)` is the mean over **this env's** zones — NOT a batch-wide mean.
    Under `vmap` the reductions below (`jnp.mean`, `jnp.sum`) run over the
    single remaining `(N,)` axis, which is what keeps the normalization
    per-environment. Returns `(N,)`.

    The two CPU guards (`mean_acc <= 0` and `total_after <= 0` both return
    the input unchanged) become a `jnp.where` so the function stays
    jit/vmap-pure with no data-dependent Python branch.
    """
    mean_acc = jnp.mean(accessibility)
    factor = 1.0 + growth_rate * (accessibility / mean_acc - 1.0)
    new = jnp.maximum(activity * factor, 0.0)  # guard against negative growth

    total_before = jnp.sum(activity)
    total_after = jnp.sum(new)
    renormalized = new * (total_before / total_after)

    safe = (mean_acc > 0.0) & (total_after > 0.0)
    return jnp.where(safe, renormalized, activity)


@partial(jax.jit, static_argnames=())
def update_activity_batched(
    activity: jax.Array,
    accessibility: jax.Array,
    growth_rate: jax.Array | float,
) -> jax.Array:
    """Batched land-use update across `(B, N)` activity and accessibility.

    `growth_rate` is a shared scalar (non-batched). The per-env mean/sum
    inside `update_activity_single` make each env normalize against itself;
    a whole-batch reduction here would be silently wrong and is exactly what
    the batched equivalence test guards against. Returns `(B, N)`.
    """
    return jax.vmap(update_activity_single, in_axes=(0, 0, None))(
        activity, accessibility, growth_rate
    )


# ---------------------------------------------------------------------------
# 6. Gravity demand   [mirrors sim_cpu.dynamics.compute_demand]
# ---------------------------------------------------------------------------


def compute_demand_single(
    activity: jax.Array,
    travel_times: jax.Array,
    alpha: jax.Array | float,
    beta: jax.Array | float,
) -> jax.Array:
    """Gravity-model OD demand matrix, single environment.

        D_ij = alpha * A_i * A_j / max(t_ij, 1e-9)^beta,   D_ii = 0

    `activity` is `(N,)`, `travel_times` is `(N, N)`. The `1e-9` floor
    matches the CPU (the zero diagonal would otherwise divide by zero); the
    diagonal is then forced to zero, so the floored values never survive.
    Returns `(N, N)`.
    """
    t_safe = jnp.maximum(travel_times, 1e-9)
    demand = alpha * (activity[:, None] * activity[None, :]) / (t_safe ** beta)
    n = activity.shape[0]
    return jnp.where(jnp.eye(n, dtype=bool), 0.0, demand)


@partial(jax.jit, static_argnames=())
def compute_demand_batched(
    activity: jax.Array,
    travel_times: jax.Array,
    alpha: jax.Array | float,
    beta: jax.Array | float,
) -> jax.Array:
    """Batched gravity demand across `(B, N)` activity and `(B, N, N)` times.

    `alpha`, `beta` are shared scalars (non-batched). Returns `(B, N, N)`.
    """
    return jax.vmap(compute_demand_single, in_axes=(0, 0, None, None))(
        activity, travel_times, alpha, beta
    )


# ---------------------------------------------------------------------------
# 1. Action application   [mirrors sim_cpu.dynamics.apply_action]
# ---------------------------------------------------------------------------


def apply_action_single(
    edge_mask: jax.Array,
    budget: jax.Array,
    action: jax.Array,
    cost_per_edge: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Apply one action to a single env's `(E,)` edge mask + scalar budget.

    Action convention (matches the CPU): `action == E` (== `n_candidate`,
    the no-op index) is a no-op; an in-range action activates candidate edge
    `action` iff it is currently inactive AND affordable, otherwise it
    silently degrades to a no-op. Returns `(new_mask, new_budget,
    did_activate)`.

    The CPU's three early-return branches become branchless masking so the
    function is vmap/jit-pure. `action == E` would index out of bounds into
    `cost_per_edge` (length E), so the gather index is clamped to 0 and the
    result discarded via `is_noop`.
    """
    n_candidate = cost_per_edge.shape[0]
    is_noop = action >= n_candidate
    safe_idx = jnp.where(is_noop, 0, action)

    cost = cost_per_edge[safe_idx]
    already_active = edge_mask[safe_idx]
    affordable = cost <= budget
    activate = (~is_noop) & (~already_active) & affordable

    new_mask = jnp.where(activate, edge_mask.at[safe_idx].set(True), edge_mask)
    new_budget = jnp.where(activate, budget - cost, budget)
    return new_mask, new_budget, activate


@partial(jax.jit, static_argnames=())
def apply_action_batched(
    edge_mask: jax.Array,
    budget: jax.Array,
    action: jax.Array,
    cost_per_edge: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Batched action application across `(B, E)` masks / `(B,)` budgets+actions.

    `cost_per_edge` is the shared `(E,)` world array (non-batched). Returns
    `(new_mask (B, E), new_budget (B,), did_activate (B,))`.
    """
    return jax.vmap(apply_action_single, in_axes=(0, 0, 0, None))(
        edge_mask, budget, action, cost_per_edge
    )


# ---------------------------------------------------------------------------
# 5. Welfare / reward   [mirrors sim_cpu.dynamics.compute_welfare]
# ---------------------------------------------------------------------------


def compute_welfare_single(activity: jax.Array, accessibility: jax.Array) -> jax.Array:
    """Population-weighted average accessibility (the reward), single env.

        W = sum_i A_i a_i / sum_i A_i,   0 if total activity is non-positive.

    `activity`, `accessibility` are `(N,)`. Returns a scalar.
    """
    total = jnp.sum(activity)
    return jnp.where(total > 0.0, jnp.sum(activity * accessibility) / total, 0.0)


@partial(jax.jit, static_argnames=())
def compute_welfare_batched(activity: jax.Array, accessibility: jax.Array) -> jax.Array:
    """Batched welfare across `(B, N)` activity and accessibility. Returns `(B,)`."""
    return jax.vmap(compute_welfare_single, in_axes=(0, 0))(activity, accessibility)


__all__ = [
    "compute_accessibility_single",
    "compute_accessibility_batched",
    "update_activity_single",
    "update_activity_batched",
    "compute_demand_single",
    "compute_demand_batched",
    "apply_action_single",
    "apply_action_batched",
    "compute_welfare_single",
    "compute_welfare_batched",
]
