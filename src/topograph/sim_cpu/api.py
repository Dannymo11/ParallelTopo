"""Canonical simulator API: `make_world`, `reset`, `step`, `run_episode`.

This module pins down the function signatures the GPU port (M3) will mirror.
The dynamics in `step` are a deliberate **no-op stub** at this checkpoint —
the action is recorded, the step counter advances, and reward is zero.
Filling in real grid-city dynamics is task #8.

Why a stub instead of leaving `step` empty:

* Lets the benchmark harness, sanity-test scaffolding, and `run_episode`
  driver be written and exercised end-to-end at this checkpoint.
* Provides the "evaluation code is at least running on a trivial baseline"
  artifact the CS348K checkpoint asks for. The trivial baseline produces
  an identifiably-empty result (zero reward, zero edges added), which the
  sanity tests in task #3 will reject when we plug in real dynamics.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .dynamics import (
    apply_action,
    compute_accessibility,
    compute_demand,
    compute_travel_times,
    compute_welfare,
    update_activity,
)
from .types import Action, GridCityConfig, State, Trajectory, WorldConfig

# A policy is a pure callable that maps (state, rng) to an action index. We
# don't dictate that policies be stateless — closing over learned weights is
# fine — only that the action returned be valid in `state.world.action_dim`.
Policy = Callable[[State, np.random.Generator], Action]


# ---------------------------------------------------------------------------
# World generation
# ---------------------------------------------------------------------------


def make_world(seed: int, cfg: GridCityConfig) -> WorldConfig:
    """Build a deterministic synthetic grid city.

    Layout: zones placed on a regular `rows x cols` grid with `cell_size_m`
    spacing. Walk distances are Euclidean. Candidate transit edges are the
    unordered pairs `(i, j), i < j` connecting each zone to its
    `candidate_k_nearest` Euclidean neighbors (deduplicated). Each edge's
    transit weight is `euclidean_distance / transit_speed_mps` and its cost
    is the same Euclidean distance — so a fixed budget buys a roughly
    constant total length of transit network regardless of where it goes.

    The `seed` is consumed only by future stochastic extensions (e.g.,
    perturbed grid layouts). At M1 the generator is fully deterministic
    given `cfg`; the seed is wired in now so we don't have to widen the
    signature later.
    """
    rng = np.random.default_rng(seed)  # noqa: F841 — reserved for stochastic layouts
    rows, cols = cfg.grid_shape
    n_zones = rows * cols

    # Zones on a regular grid, in meters.
    yy, xx = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    zone_positions = (
        np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float64) * cfg.cell_size_m
    )

    # Pairwise Euclidean distances and walking times.
    diffs = zone_positions[:, None, :] - zone_positions[None, :, :]
    walk_distances = np.linalg.norm(diffs, axis=-1).astype(np.float64)
    walk_times = walk_distances / cfg.walk_speed_mps

    # Candidate transit edges: each zone -> its k nearest neighbors (excluding
    # itself), deduplicated to undirected pairs.
    k = min(cfg.candidate_k_nearest, n_zones - 1)
    # argsort along axis=1 puts the zone itself at index 0 (distance zero);
    # take indices 1..k+1 for the k nearest.
    nearest = np.argsort(walk_distances, axis=1)[:, 1 : k + 1]
    pair_set: set[tuple[int, int]] = set()
    for i in range(n_zones):
        for j in nearest[i]:
            a, b = (i, int(j)) if i < j else (int(j), i)
            pair_set.add((a, b))
    candidate_edges = np.array(sorted(pair_set), dtype=np.int64)
    edge_distances = walk_distances[candidate_edges[:, 0], candidate_edges[:, 1]]
    candidate_edge_weights = edge_distances / cfg.transit_speed_mps
    # Edge cost is expressed in *grid units* (distance / cell_size_m) rather
    # than raw meters, so the budget number in `GridCityConfig` is meaningful
    # at any cell size. A grid-neighbor edge costs ~1; a diagonal ~sqrt(2).
    cost_per_edge = (edge_distances / cfg.cell_size_m).astype(np.float64)

    # Uniform initial activity.
    initial_activity = np.full(
        n_zones, cfg.initial_activity_total / n_zones, dtype=np.float64
    )

    return WorldConfig(
        zone_positions=zone_positions,
        walk_distances=walk_distances,
        walk_times=walk_times,
        candidate_edges=candidate_edges,
        candidate_edge_weights=candidate_edge_weights,
        cost_per_edge=cost_per_edge,
        initial_activity=initial_activity,
        initial_budget=cfg.initial_budget,
        horizon=cfg.horizon,
        gravity_alpha=cfg.gravity_alpha,
        gravity_beta=cfg.gravity_beta,
        accessibility_decay=cfg.accessibility_decay,
        growth_rate=cfg.growth_rate,
    )


# ---------------------------------------------------------------------------
# Episode lifecycle
# ---------------------------------------------------------------------------


def reset(world: WorldConfig) -> State:
    """Return the initial state for an episode in the given world.

    Initial conditions: no transit edges active, activity at the world's
    `initial_activity`, full budget, step counter at zero, not done.
    """
    return State(
        world=world,
        edge_mask=np.zeros(world.n_candidate_edges, dtype=np.bool_),
        activity=world.initial_activity.copy(),
        budget_remaining=world.initial_budget,
        step=0,
        done=False,
    )


def step(state: State, action: Action) -> tuple[State, float, bool, dict[str, Any]]:
    """Advance the simulation by one step under the full M1 dynamics.

    Composition:

      1. apply_action          — mutate edge_mask, budget under the action.
      2. compute_travel_times  — APSP under the (possibly updated) network.
      3. compute_accessibility — exp-decay kernel @ activity.
      4. update_activity       — multiplicative growth toward higher-acc zones,
                                 renormalized so the total population is preserved.
      5. compute_welfare       — population-weighted average accessibility,
                                 used as the reward.
      6. compute_demand        — informational gravity demand (placed in `info`).

    Returns ``(next_state, reward, done, info)``. The ``info`` dict carries
    the per-step travel-times, accessibility, and demand matrices for
    downstream visualization and the M5 ablations; it is not part of the
    state contract.
    """
    if not (0 <= action <= state.world.no_op_action):
        raise ValueError(
            f"action {action} out of range [0, {state.world.no_op_action}]"
        )
    if state.done:
        raise RuntimeError("step called on a terminated state")

    # 1. Action -> updated mask + budget (silent no-op for illegal-but-in-range).
    new_mask, new_budget, did_activate = apply_action(state, action)

    # 2-3. APSP and accessibility under the (possibly updated) network.
    travel_times = compute_travel_times(state.world, new_mask)
    accessibility = compute_accessibility(state.world, state.activity, travel_times)

    # 4. Activity grows toward higher-access zones.
    new_activity = update_activity(
        state.activity, accessibility, state.world.growth_rate
    )

    # 5. Reward = population-weighted accessibility under the network and the
    #    *post-growth* activity. The pre-growth accessibility kernel is reused
    #    (same travel_times) so the only extra work is one matrix-vector multiply.
    final_acc = compute_accessibility(state.world, new_activity, travel_times)
    reward = compute_welfare(new_activity, final_acc)

    # 6. Demand surfaced for info / future ridership reward.
    demand = compute_demand(state.world, new_activity, travel_times)

    next_step = state.step + 1
    next_done = next_step >= state.world.horizon

    next_state = State(
        world=state.world,
        edge_mask=new_mask,
        activity=new_activity,
        budget_remaining=new_budget,
        step=next_step,
        done=next_done,
    )
    info: dict[str, Any] = {
        "did_activate": did_activate,
        "travel_times": travel_times,
        "accessibility": final_acc,
        "demand": demand,
    }
    return next_state, reward, next_done, info


def run_episode(
    state: State,
    policy: Policy,
    rng: np.random.Generator | int,
) -> Trajectory:
    """Drive a full episode from `state` to termination under `policy`.

    `rng` may be an int seed (a fresh `default_rng` will be made) or an
    existing `Generator` (passed through to the policy each step).

    The driver is implementation-agnostic about the simulator's dynamics —
    it works against the stubbed `step` today and against the real
    dynamics tomorrow.
    """
    if isinstance(rng, int):
        rng = np.random.default_rng(rng)

    horizon = state.world.horizon
    actions = np.empty(horizon, dtype=np.int64)
    rewards = np.empty(horizon, dtype=np.float64)

    for t in range(horizon):
        action = int(policy(state, rng))
        next_state, reward, done, _info = step(state, action)
        actions[t] = action
        rewards[t] = reward
        state = next_state
        if done:
            break
    else:
        # `else` on a `for` runs only if the loop wasn't broken out of.
        # Reaching here without `done` means the horizon contract is broken.
        if not state.done:
            raise RuntimeError(
                "run_episode finished horizon without state.done; "
                "step termination contract is violated"
            )

    return Trajectory(actions=actions, rewards=rewards, final_state=state)


# ---------------------------------------------------------------------------
# Convenience: vectorized action validity (used by future policies)
# ---------------------------------------------------------------------------


def valid_action_mask(state: State) -> NDArray[np.bool_]:
    """Boolean mask of which actions are *legal* in this state.

    An action is legal iff (a) it is the no-op, or (b) the candidate edge it
    refers to is currently inactive *and* its cost fits in the remaining
    budget. The stub `step` doesn't enforce this — it accepts any in-range
    action — but real policies and sanity tests should use this mask to
    avoid generating unreachable transitions.
    """
    n_candidate = state.world.n_candidate_edges
    mask = np.empty(state.world.action_dim, dtype=np.bool_)
    affordable = state.world.cost_per_edge <= state.budget_remaining
    mask[:n_candidate] = (~state.edge_mask) & affordable
    mask[n_candidate] = True  # no-op is always legal
    return mask


def find_edge_index(world: WorldConfig, a: int, b: int) -> int:
    """Return the index of edge (a, b) in `world.candidate_edges`.

    Lookup is undirected: ``find_edge_index(w, a, b) == find_edge_index(w, b, a)``.
    Raises ``KeyError`` if the pair is not in the candidate set (e.g., the
    zones are too far apart under the k-nearest construction).

    Used by scripted policies and tests that need to refer to specific edges
    by their endpoint zones rather than by candidate index.
    """
    lo, hi = (a, b) if a < b else (b, a)
    matches = np.where(
        (world.candidate_edges[:, 0] == lo) & (world.candidate_edges[:, 1] == hi)
    )[0]
    if matches.size == 0:
        raise KeyError(f"edge ({a}, {b}) not in candidate set")
    return int(matches[0])
