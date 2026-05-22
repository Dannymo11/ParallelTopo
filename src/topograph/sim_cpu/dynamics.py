"""Per-component dynamics for the CPU simulator.

Each function is **pure**: it takes raw arrays / scalars (not a `State`
wrapper) and returns the updated quantity. This is deliberate.

* It lets each component be unit-tested in isolation, without round-tripping
  through `step`.
* It makes the per-component time breakdown the profiling pass produces in
  task #5 directly map to function names — no inlining games to worry about.
* It mirrors how the GPU port (M3) will look: the JAX version's
  ``vmap``-able functions accept arrays + a non-batched ``WorldConfig`` and
  return arrays, in exactly this shape.

Composition order in a single `step`, defined in `api.py`:

    1. apply_action          : mutate edge_mask, budget under the chosen action.
    2. compute_travel_times  : APSP with walking floor + active transit. <- HOT.
    3. compute_accessibility : exp-decay kernel @ activity.
    4. update_activity       : multiplicative growth toward higher-acc zones.
    5. compute_welfare       : population-weighted average accessibility.
    6. compute_demand        : informational gravity demand (info-only at M1).

The reward returned by `step` is the welfare from (5). Demand is computed
once per step and surfaced through ``info`` for downstream visualization
and for the M5 ridership-reward variant; it's not in the reward path itself.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.csgraph import floyd_warshall

from .types import State, WorldConfig

# ---------------------------------------------------------------------------
# 1. Action application
# ---------------------------------------------------------------------------


def apply_action(
    state: State, action: int
) -> tuple[NDArray[np.bool_], float, bool]:
    """Apply ``action`` to the edge mask + budget. Returns the updated edge
    mask, updated remaining budget, and a flag indicating whether an edge
    was actually activated.

    Convention: *out-of-range* actions must be caught by the caller (`step`
    raises). *In-range but illegal* actions (already-active edge,
    unaffordable cost, no-op) silently degrade to no-op — matching how
    standard RL simulators (Brax, dm_control, gymnasium) treat invalid
    actions. Policies that use `valid_action_mask` never hit the silent
    paths; this is a safety net for unmasked policies.
    """
    if action == state.world.no_op_action:
        return state.edge_mask, state.budget_remaining, False

    cost = float(state.world.cost_per_edge[action])
    if state.edge_mask[action]:
        return state.edge_mask, state.budget_remaining, False
    if cost > state.budget_remaining:
        return state.edge_mask, state.budget_remaining, False

    new_mask = state.edge_mask.copy()
    new_mask[action] = True
    return new_mask, state.budget_remaining - cost, True


# ---------------------------------------------------------------------------
# 2. APSP travel times
# ---------------------------------------------------------------------------


def _build_direct_distance_matrix(
    world: WorldConfig,
    edge_mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Direct-hop distance matrix: walk_times everywhere, overridden by
    transit_times on currently-active candidate edges.

    Shared by both `compute_travel_times` (scipy backend) and
    `compute_travel_times_numpy` (reference NumPy backend) so the only
    difference between them is the APSP routine itself.
    """
    direct = world.walk_times.copy()
    if edge_mask.any():
        active = world.candidate_edges[edge_mask]
        weights = world.candidate_edge_weights[edge_mask]
        i = active[:, 0]
        j = active[:, 1]
        # Replace the direct-hop weight with min(walk, transit).
        # Note: ``np.minimum(direct[i, j], weights, out=direct[i, j])`` does
        # NOT work — fancy indexing on the LHS returns a copy, so the in-
        # place write lands on a temporary. Use plain assignment instead.
        direct[i, j] = np.minimum(direct[i, j], weights)
        direct[j, i] = np.minimum(direct[j, i], weights)
    return direct


def compute_travel_times(
    world: WorldConfig,
    edge_mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """All-pairs shortest-path travel times under the active network.

    Direct hops use ``min(walking_time, transit_time)`` where a transit edge
    is active; multi-hop paths are found via Floyd-Warshall over the dense
    walking floor. The walking floor guarantees every pair has a finite
    travel time regardless of which transit edges are active.

    **Backend: scipy.sparse.csgraph.floyd_warshall.** This is the M1 default
    because it is the most competitive single-environment CPU baseline —
    the scipy implementation is C-coded and runs ~1.7× faster end-to-end
    than the hand-vectorized NumPy version (see
    `compute_travel_times_numpy` and `scripts/apsp_baseline_comparison.py`).
    A weak CPU baseline would artificially inflate the M3 GPU speedup
    number; that's exactly the thing the systems-paper write-up cannot
    afford to be wrong about.

    Returns a symmetric N×N matrix with zero diagonal. This is the M2
    feasibility-study target: the GPU port will replace this with batched
    approximate APSP (K iterations of Bellman-Ford-style relaxation).
    """
    direct = _build_direct_distance_matrix(world, edge_mask)
    return floyd_warshall(direct, directed=False, overwrite=True)


def compute_travel_times_numpy(
    world: WorldConfig,
    edge_mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Hand-vectorized NumPy Floyd-Warshall — kept as a reference implementation.

    Same inputs, same outputs (numerically identical to
    `compute_travel_times` — verified to floating-point zero) but ~1.7×
    slower end-to-end. Two reasons it stays in the public surface:

    1. **GPU-port symmetry.** The JAX batched Bellman-Ford in M3 will
       look much more like this loop than like a scipy C call, so this
       function is the reference that the GPU version's correctness
       tests cross-check against.
    2. **Reproducibility of the baseline comparison.**
       `scripts/apsp_baseline_comparison.py` benches scipy vs. this
       function head-to-head to confirm the speedup ratio hasn't drifted.
    """
    d = _build_direct_distance_matrix(world, edge_mask)
    n = d.shape[0]
    for k in range(n):
        d = np.minimum(d, d[:, k : k + 1] + d[k : k + 1, :])
    return d


# ---------------------------------------------------------------------------
# 3. Accessibility (exp-decay kernel)
# ---------------------------------------------------------------------------


def compute_accessibility(
    world: WorldConfig,
    activity: NDArray[np.float64],
    travel_times: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Per-zone accessibility under an exponential-decay kernel.

    .. math::
        a_i = \\sum_j A_j \\cdot \\exp(-\\lambda \\cdot t_{ij})

    where :math:`\\lambda` is ``world.accessibility_decay``. The diagonal
    contributes ``A_i`` (since ``t_ii = 0``); we leave it in to keep the
    formula uniform and matrix-multiply-friendly.
    """
    kernel = np.exp(-world.accessibility_decay * travel_times)
    return kernel @ activity


# ---------------------------------------------------------------------------
# 4. Land-use update
# ---------------------------------------------------------------------------


def update_activity(
    activity: NDArray[np.float64],
    accessibility: NDArray[np.float64],
    growth_rate: float,
) -> NDArray[np.float64]:
    """Multiplicative land-use growth toward higher-accessibility zones.

    Step:
        ``A_i' = A_i * (1 + growth_rate * (a_i / mean(a) - 1))``

    Then renormalize so the total activity is preserved exactly. The
    renormalization is what makes "land-use shifts toward high access"
    rather than "land-use grows because access is high" — the simulator
    models redistribution under a fixed population, which is the simpler
    and more comparable-across-steps choice.
    """
    mean_acc = float(accessibility.mean())
    if mean_acc <= 0.0:
        return activity.copy()
    factor = 1.0 + growth_rate * (accessibility / mean_acc - 1.0)
    new = activity * factor
    np.clip(new, 0.0, None, out=new)  # guard against negative growth
    total_before = float(activity.sum())
    total_after = float(new.sum())
    if total_after <= 0.0:
        return activity.copy()
    new *= total_before / total_after
    return new


# ---------------------------------------------------------------------------
# 5. Welfare / reward
# ---------------------------------------------------------------------------


def compute_welfare(
    activity: NDArray[np.float64],
    accessibility: NDArray[np.float64],
) -> float:
    """Population-weighted average accessibility.

    .. math::
        W = \\frac{\\sum_i A_i \\cdot a_i}{\\sum_i A_i}

    With activity preserved across steps (see `update_activity`), the
    denominator is constant, so per-step welfare is directly comparable
    across an episode. With uniform activity, ``W == mean(accessibility)``.
    """
    total = float(activity.sum())
    if total <= 0.0:
        return 0.0
    return float((activity * accessibility).sum() / total)


# ---------------------------------------------------------------------------
# 6. Gravity demand (info-only at M1)
# ---------------------------------------------------------------------------


def compute_demand(
    world: WorldConfig,
    activity: NDArray[np.float64],
    travel_times: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Gravity-model OD demand matrix.

    .. math::
        D_{ij} = \\alpha \\cdot \\frac{A_i \\, A_j}{t_{ij}^{\\beta}}

    Diagonal forced to zero. Currently surfaced through ``info`` only —
    the M1 reward uses accessibility-weighted welfare, not demand-weighted
    ridership. The M5 ablation may add a ridership reward variant that
    uses this directly.
    """
    a = activity
    t_safe = np.maximum(travel_times, 1e-9)  # diagonal is zero; avoid div-by-zero
    demand = world.gravity_alpha * (a[:, None] * a[None, :]) / (
        t_safe ** world.gravity_beta
    )
    np.fill_diagonal(demand, 0.0)
    return demand
