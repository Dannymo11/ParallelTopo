"""Dataclasses defining the simulator's state surface.

The state surface is the *contract* between the CPU baseline and the GPU port:
both implement the same `make_world / reset / step / run_episode` API on the
same dataclasses. The CPU version (M1) operates on a single environment; the
GPU version (M3) batches these via JAX `vmap`, where each leaf of the PyTree
gains a leading batch dimension.

Design choices worth remembering:

* All dynamic state lives in `State`. Static world structure (graph layout,
  candidate edges, walk distances, costs) lives in `WorldConfig`. `State`
  carries a reference to its `WorldConfig` for ergonomic single-env use; the
  GPU port restructures this so config is a non-batched argument and state
  is the batched PyTree.
* The action space encodes "activate candidate edge `i`" for `i ∈ [0,
  n_candidate)` and reserves `i == n_candidate` as the no-op action. Storing
  no-op as a regular action index (rather than `-1`) keeps the action space
  contiguous and JAX-friendly.
* Dataclasses are `frozen=True, slots=True` so each step returns a new state
  instance via `dataclasses.replace`. This mirrors how the GPU port has to
  work (pure functions, no in-place mutation under `jit`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# An action is a single integer index. ``a == n_candidate`` means no-op;
# ``a < n_candidate`` means "activate candidate_edges[a] on this step".
Action = int


# ---------------------------------------------------------------------------
# Generator-side config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GridCityConfig:
    """Parameters controlling synthetic-grid-city generation.

    `make_world(seed, GridCityConfig)` produces a deterministic `WorldConfig`
    from these parameters. Kept separate from `WorldConfig` so the same world
    can be reused across many resets without regenerating geometry, and so a
    sweep over grid sizes only varies this object.
    """

    # Grid shape: ``(rows, cols)``. Total zones = rows * cols. M1 default is
    # 10x10 = 100 zones; M5 sweep goes up to 20x20 = 400.
    grid_shape: tuple[int, int] = (10, 10)

    # Spacing between adjacent grid cells (meters). Sets the geometric scale
    # of the city; affects walk vs. transit travel times.
    cell_size_m: float = 500.0

    # Effective speeds. Walking is the always-available fallback; transit
    # edges, when activated, are faster. The ratio governs how attractive
    # transit edges are vs. walking.
    walk_speed_mps: float = 1.4  # ~5 km/h
    transit_speed_mps: float = 8.0  # ~30 km/h

    # Gravity-model demand: D_ij ∝ A_i * A_j / (travel_time_ij ** beta).
    # Alpha is a normalization knob (left explicit so it doesn't get baked
    # into other constants).
    gravity_alpha: float = 1.0
    gravity_beta: float = 1.5

    # Accessibility decay: accessibility_i = sum_j A_j * exp(-decay * t_ij).
    # Smaller `decay` -> longer-range accessibility -> bigger benefit from
    # any edge that shortens travel time.
    accessibility_decay: float = 1.0 / 600.0  # exp(-1) at 10 minutes

    # Land-use growth rate per step. Each zone's activity is updated as
    # ``A_i' = A_i * (1 + growth_rate * (a_i / mean(a) - 1))``, normalized so
    # total activity stays bounded.
    growth_rate: float = 0.1

    # Total initial activity, distributed uniformly across zones at reset.
    initial_activity_total: float = 100.0

    # Episode horizon (number of steps). M1 default = 15.
    horizon: int = 15

    # Total budget for edge additions across the whole episode. Per-edge cost
    # is set inside `make_world`; together they determine how many edges an
    # episode can afford.
    initial_budget: float = 50.0

    # Candidate-edge generation: a "k-nearest" candidate set keeps the action
    # space tractable and avoids cluttering the simulator with edges that
    # would never make sense (e.g., across-the-grid bee-line edges with very
    # long walking-equivalent times). M1 uses k = 8.
    candidate_k_nearest: int = 8


# ---------------------------------------------------------------------------
# Static world structure
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WorldConfig:
    """Static, immutable world structure produced by `make_world`.

    Shapes (with N = number of zones, E = number of candidate edges):

    * `zone_positions`           : (N, 2) float64 — Euclidean coordinates.
    * `walk_distances`           : (N, N) float64 — base walking distances.
    * `walk_times`               : (N, N) float64 — `walk_distances / walk_speed`.
    * `candidate_edges`          : (E, 2) int64   — undirected (i, j) pairs, i < j.
    * `candidate_edge_weights`   : (E,)   float64 — transit travel time if active.
    * `cost_per_edge`            : (E,)   float64 — budget cost to activate.
    * `initial_activity`         : (N,)   float64 — per-zone activity at reset.
    * `initial_budget`           : float           — total episode budget.
    * `horizon`                  : int             — episode length.

    Derived parameters (gravity / accessibility / growth) live in
    `GridCityConfig`-derived attributes attached here so the simulator only
    needs `WorldConfig` at step time.
    """

    zone_positions: NDArray[np.float64]
    walk_distances: NDArray[np.float64]
    walk_times: NDArray[np.float64]
    candidate_edges: NDArray[np.int64]
    candidate_edge_weights: NDArray[np.float64]
    cost_per_edge: NDArray[np.float64]
    initial_activity: NDArray[np.float64]
    initial_budget: float
    horizon: int

    # Carried through from GridCityConfig so step dynamics (M1, task #8) can
    # read them without going back to the generator config.
    gravity_alpha: float
    gravity_beta: float
    accessibility_decay: float
    growth_rate: float

    @property
    def n_zones(self) -> int:
        return int(self.zone_positions.shape[0])

    @property
    def n_candidate_edges(self) -> int:
        return int(self.candidate_edges.shape[0])

    @property
    def action_dim(self) -> int:
        """Total action-space size, including the no-op action."""
        return self.n_candidate_edges + 1

    @property
    def no_op_action(self) -> int:
        """The integer index of the no-op action."""
        return self.n_candidate_edges


# ---------------------------------------------------------------------------
# Mutable per-step state
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class State:
    """Mutable per-step state. Returned (as a fresh instance) by every `step`.

    Minimal — derived quantities (accessibility scores, demand matrix,
    effective travel times) are *not* stored. They are recomputed on demand
    inside `step` and surfaced through the `info` dict.

    Shapes (with E = `world.n_candidate_edges`, N = `world.n_zones`):

    * `edge_mask`        : (E,) bool    — which candidate edges are active.
    * `activity`         : (N,) float64 — current per-zone activity.
    * `budget_remaining` : float        — budget left for further edges.
    * `step`             : int          — 0-indexed step counter (== horizon at termination).
    * `done`             : bool         — terminal flag.
    """

    world: WorldConfig
    edge_mask: NDArray[np.bool_]
    activity: NDArray[np.float64]
    budget_remaining: float
    step: int
    done: bool


# ---------------------------------------------------------------------------
# Episode trajectory (used by run_episode and tests)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Trajectory:
    """The record of a single completed episode.

    * `actions`     : (T,) int64   — per-step action taken.
    * `rewards`     : (T,) float64 — per-step reward.
    * `final_state` : the State after the last step (with done=True).

    where T == final_state.step (which equals world.horizon under the M1
    contract — episodes always run for the full horizon).
    """

    actions: NDArray[np.int64]
    rewards: NDArray[np.float64]
    final_state: State

    @property
    def episode_return(self) -> float:
        return float(self.rewards.sum())

    @property
    def length(self) -> int:
        return int(self.actions.shape[0])
