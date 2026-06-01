"""Assembled vmapped GPU simulator step (M3).

Composes the M2 APSP kernel and the M3 dynamics components into a single
`step(states, actions) -> (next_states, rewards, dones)` that advances a
whole batch of environments in lockstep on-device, mirroring the CPU
`sim_cpu.api.step` composition exactly:

    1. apply_action          -> updated edge mask + budget
    2. build direct-distance -> walking floor with active-transit overrides
    3. apsp (K=5 squaring)   -> all-pairs shortest-path travel times
    4. compute_accessibility -> exp-decay kernel @ *old* activity
    5. update_activity       -> land-use redistribution (per-env normalized)
    6. compute_accessibility -> kernel @ *new* activity (for the reward)
    7. compute_welfare       -> population-weighted accessibility = reward

Demand (`compute_demand`) is info-only in the CPU step and is not on the
reward path, so it is omitted here; callers that need it can call
`compute_demand_batched` on the returned travel times.

**State layout (ECS-like, à la Madrona).** State is a `NamedTuple` of
per-attribute arrays with the batch axis leading — `jnp`-friendly, no
ragged tensors, no Python loop over environments. Static world structure
lives in a separate non-batched `WorldArrays` PyTree. JAX treats both
NamedTuples as PyTrees automatically, so `vmap`/`jit` thread them without
custom registration.

**Direct-distance construction is now on-device** (`build_direct_distance_*`),
closing the M3 deliverable that the M2 throughput script flagged: the
benchmark no longer builds the matrix on CPU and transfers it.

**`freeze_mask`** is the M5 Figure-4 ablation knob. When True, step skips
the edge-mask update entirely (topology frozen at episode start), so the
only difference vs. the production path is the per-step mask scatter — the
overhead this measures against fixed-topology batched simulators.

fp32 throughout, matching the M2 APSP kernel decision.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .apsp import DEFAULT_K_ITERATIONS, apsp_matrix_squaring
from .dynamics import (
    apply_action_single,
    compute_accessibility_single,
    compute_welfare_single,
    update_activity_single,
)


class WorldArrays(NamedTuple):
    """Static, non-batched world structure the step needs, as JAX arrays.

    Built from a CPU `WorldConfig` via `world_arrays_from_config`. Scalars
    (decay, growth_rate) are carried as 0-d arrays so they ride along the
    same PyTree; `horizon` and `k_iterations` are passed separately as
    static (compile-time) ints.
    """

    walk_times: jax.Array            # (N, N) f32
    candidate_edges: jax.Array       # (E, 2) int32
    candidate_edge_weights: jax.Array  # (E,) f32
    cost_per_edge: jax.Array         # (E,) f32
    accessibility_decay: jax.Array   # scalar f32
    growth_rate: jax.Array           # scalar f32


class GPUState(NamedTuple):
    """Batched per-step state. Each leaf has a leading batch axis of size B."""

    edge_mask: jax.Array   # (B, E) bool
    activity: jax.Array    # (B, N) f32
    budget: jax.Array      # (B,) f32
    step: jax.Array        # (B,) int32
    done: jax.Array        # (B,) bool


def world_arrays_from_config(world) -> WorldArrays:
    """Convert a CPU `WorldConfig` into device-resident fp32/int32 arrays."""
    return WorldArrays(
        walk_times=jnp.asarray(world.walk_times, dtype=jnp.float32),
        candidate_edges=jnp.asarray(world.candidate_edges, dtype=jnp.int32),
        candidate_edge_weights=jnp.asarray(
            world.candidate_edge_weights, dtype=jnp.float32
        ),
        cost_per_edge=jnp.asarray(world.cost_per_edge, dtype=jnp.float32),
        accessibility_decay=jnp.asarray(world.accessibility_decay, dtype=jnp.float32),
        growth_rate=jnp.asarray(world.growth_rate, dtype=jnp.float32),
    )


def reset_batched(
    world: WorldArrays,
    batch_size: int,
    initial_activity: jax.Array,
    initial_budget: float,
) -> GPUState:
    """Initial batched state: no active edges, uniform activity, full budget."""
    n_candidate = world.cost_per_edge.shape[0]
    n_zones = world.walk_times.shape[0]
    activity = jnp.broadcast_to(
        jnp.asarray(initial_activity, dtype=jnp.float32), (batch_size, n_zones)
    )
    return GPUState(
        edge_mask=jnp.zeros((batch_size, n_candidate), dtype=bool),
        activity=activity,
        budget=jnp.full((batch_size,), float(initial_budget), dtype=jnp.float32),
        step=jnp.zeros((batch_size,), dtype=jnp.int32),
        done=jnp.zeros((batch_size,), dtype=bool),
    )


# ---------------------------------------------------------------------------
# Direct-distance matrix (on-device): walking floor + active-transit overrides
# ---------------------------------------------------------------------------


def build_direct_distance_single(edge_mask: jax.Array, world: WorldArrays) -> jax.Array:
    """`(N, N)` direct-hop distances: walk_times, min'd with active edges.

    Mirrors `sim_cpu.dynamics._build_direct_distance_matrix`. Inactive
    candidate edges contribute `+inf`, so the scatter-min leaves the walking
    floor untouched there; active edges override with `min(walk, transit)`.
    Each undirected pair is written to both `(i, j)` and `(j, i)`.
    """
    i = world.candidate_edges[:, 0]
    j = world.candidate_edges[:, 1]
    active_w = jnp.where(
        edge_mask, world.candidate_edge_weights, jnp.asarray(jnp.inf, jnp.float32)
    )
    direct = world.walk_times
    direct = direct.at[i, j].min(active_w)
    direct = direct.at[j, i].min(active_w)
    return direct


# ---------------------------------------------------------------------------
# Single-env step (the thing that gets vmapped)
# ---------------------------------------------------------------------------


def step_single(
    state: GPUState,
    action: jax.Array,
    world: WorldArrays,
    horizon: int,
    k_iterations: int,
    freeze_mask: bool,
) -> tuple[GPUState, jax.Array]:
    """Advance one environment by one step. Returns `(next_state, reward)`."""
    if freeze_mask:
        # M5 Fig-4 ablation: topology frozen, no mask scatter / budget change.
        new_mask = state.edge_mask
        new_budget = state.budget
    else:
        new_mask, new_budget, _ = apply_action_single(
            state.edge_mask, state.budget, action, world.cost_per_edge
        )

    direct = build_direct_distance_single(new_mask, world)
    travel_times = apsp_matrix_squaring(direct, k_iterations=k_iterations)

    # Accessibility under the OLD activity drives land-use growth...
    acc_pre = compute_accessibility_single(
        state.activity, travel_times, world.accessibility_decay
    )
    new_activity = update_activity_single(state.activity, acc_pre, world.growth_rate)

    # ...then accessibility under the NEW activity is the reward basis.
    final_acc = compute_accessibility_single(
        new_activity, travel_times, world.accessibility_decay
    )
    reward = compute_welfare_single(new_activity, final_acc)

    next_step = state.step + 1
    next_done = next_step >= horizon
    next_state = GPUState(
        edge_mask=new_mask,
        activity=new_activity,
        budget=new_budget,
        step=next_step,
        done=next_done,
    )
    return next_state, reward


@partial(jax.jit, static_argnames=("horizon", "k_iterations", "freeze_mask"))
def step_batched(
    state: GPUState,
    actions: jax.Array,
    world: WorldArrays,
    horizon: int,
    k_iterations: int = DEFAULT_K_ITERATIONS,
    freeze_mask: bool = False,
) -> tuple[GPUState, jax.Array]:
    """Batched `step`. `actions` is `(B,)`; returns `(next_state, rewards (B,))`.

    `world` is the shared non-batched `WorldArrays` (bound into the vmapped
    function, so it is broadcast, not mapped). `horizon`/`k_iterations`/
    `freeze_mask` are static so the K-loop unrolls and the freeze branch is
    resolved at compile time.
    """

    def fn(s: GPUState, a: jax.Array) -> tuple[GPUState, jax.Array]:
        return step_single(s, a, world, horizon, k_iterations, freeze_mask)

    return jax.vmap(fn, in_axes=(0, 0))(state, actions)


@partial(jax.jit, static_argnames=("horizon", "k_iterations", "freeze_mask"))
def rollout_batched(
    state: GPUState,
    actions: jax.Array,
    world: WorldArrays,
    horizon: int,
    k_iterations: int = DEFAULT_K_ITERATIONS,
    freeze_mask: bool = False,
) -> tuple[GPUState, jax.Array]:
    """Run `horizon` steps over a pre-specified action schedule.

    `actions` is `(T, B)` (step-major), letting a caller replay a fixed
    action sequence — used by the B=1 numerical-equivalence test, and the
    skeleton for the M3 vectorized random-policy rollout driver (swap the
    indexed `actions[t]` for a policy call under the same `lax.scan`).
    Returns `(final_state, rewards (T, B))`.
    """

    def body(carry: GPUState, a_t: jax.Array) -> tuple[GPUState, jax.Array]:
        next_state, reward = step_batched(
            carry, a_t, world, horizon=horizon,
            k_iterations=k_iterations, freeze_mask=freeze_mask,
        )
        return next_state, reward

    return jax.lax.scan(body, state, actions)


__all__ = [
    "WorldArrays",
    "GPUState",
    "world_arrays_from_config",
    "reset_batched",
    "build_direct_distance_single",
    "step_single",
    "step_batched",
    "rollout_batched",
]
