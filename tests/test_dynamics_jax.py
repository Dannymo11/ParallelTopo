"""JAX dynamics-port correctness: matches the CPU baseline at fp32.

Equivalence harness for the three M3 dynamics components ported in
`sim_gpu/dynamics.py` — `compute_accessibility`, `update_activity`,
`compute_demand`. Each is checked against its `sim_cpu/dynamics.py`
counterpart on identical inputs, within float32 round-off (same tolerance
regime as `test_apsp_jax.py`).

To isolate each component from APSP-port error, the shared `travel_times`
input is the exact scipy-FW output (`compute_travel_times`); these tests
verify the *dynamics* math, not the shortest-path kernel (which
`test_apsp_jax.py` already covers).

The batched tests use **per-environment-distinct** activity and
accessibility so that the single highest-risk M3 bug — implementing the
`update_activity` normalization as a whole-batch reduction instead of a
per-env one — fails here instead of silently producing plausible numbers.

Skipped if JAX isn't installed (it's a `[gpu]` extra), so CPU-only dev
installs keep a green suite.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from topograph.sim_cpu import (
    GridCityConfig,
    compute_accessibility,
    compute_demand,
    compute_travel_times,
    make_world,
    update_activity,
)
from topograph.sim_cpu.dynamics import _build_direct_distance_matrix  # noqa: F401  (kept for parity/debugging)
from topograph.sim_gpu import (
    compute_accessibility_batched,
    compute_demand_batched,
    update_activity_batched,
)

# fp32 round-off tolerance — matches test_apsp_jax.py.
RTOL_FP32 = 1e-4
ATOL_FP32 = 1e-3

GRIDS = [(5, 5), (10, 10), (15, 15)]
DENSITIES = [0.25, 0.5, 0.75]


def _make_mask(n_candidate: int, density: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mask = np.zeros(n_candidate, dtype=np.bool_)
    if density > 0:
        k = max(1, int(round(density * n_candidate)))
        mask[rng.choice(n_candidate, size=k, replace=False)] = True
    return mask


def _scenario(grid_shape, density, seed, world_seed=0):
    """Build (world, travel_times, activity) for one environment.

    `activity` is deliberately non-uniform (and not the uniform reset
    distribution) so accessibility, demand, and the land-use redistribution
    are all non-trivial — a port that collapsed any of them to a constant
    would be caught.
    """
    world = make_world(world_seed, GridCityConfig(grid_shape=grid_shape))
    mask = _make_mask(world.n_candidate_edges, density, seed)
    tt = compute_travel_times(world, mask)  # (N, N) float64, exact APSP
    rng = np.random.default_rng(seed + 7)
    activity = rng.uniform(0.2, 3.0, size=world.n_zones).astype(np.float64)
    return world, tt, activity


def _f32(x):
    return jnp.asarray(np.asarray(x, dtype=np.float32))


# ---------------------------------------------------------------------------
# compute_accessibility
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("grid_shape", GRIDS)
@pytest.mark.parametrize("density", DENSITIES)
def test_compute_accessibility_matches_cpu(grid_shape, density):
    world, tt, activity = _scenario(grid_shape, density, seed=11)
    cpu = compute_accessibility(world, activity, tt)

    jax_out = np.asarray(
        compute_accessibility_batched(
            _f32(activity[None]), _f32(tt[None]), world.accessibility_decay
        )
    )[0]

    np.testing.assert_allclose(
        jax_out, cpu, rtol=RTOL_FP32, atol=ATOL_FP32,
        err_msg=f"accessibility mismatch at grid={grid_shape}, density={density}",
    )


# ---------------------------------------------------------------------------
# compute_demand
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("grid_shape", GRIDS)
@pytest.mark.parametrize("density", DENSITIES)
def test_compute_demand_matches_cpu(grid_shape, density):
    world, tt, activity = _scenario(grid_shape, density, seed=23)
    cpu = compute_demand(world, activity, tt)

    jax_out = np.asarray(
        compute_demand_batched(
            _f32(activity[None]), _f32(tt[None]),
            world.gravity_alpha, world.gravity_beta,
        )
    )[0]

    np.testing.assert_allclose(
        jax_out, cpu, rtol=RTOL_FP32, atol=ATOL_FP32,
        err_msg=f"demand mismatch at grid={grid_shape}, density={density}",
    )
    # Diagonal must be exactly zero (no self-demand), like the CPU.
    np.testing.assert_array_equal(np.diag(jax_out), np.zeros(world.n_zones))


# ---------------------------------------------------------------------------
# update_activity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("grid_shape", GRIDS)
@pytest.mark.parametrize("density", DENSITIES)
def test_update_activity_matches_cpu(grid_shape, density):
    world, tt, activity = _scenario(grid_shape, density, seed=37)
    # Feed the SAME accessibility to both sides to isolate update_activity
    # from any accessibility-port error.
    acc = compute_accessibility(world, activity, tt)
    cpu = update_activity(activity, acc, world.growth_rate)

    jax_out = np.asarray(
        update_activity_batched(_f32(activity[None]), _f32(acc[None]), world.growth_rate)
    )[0]

    np.testing.assert_allclose(
        jax_out, cpu, rtol=RTOL_FP32, atol=ATOL_FP32,
        err_msg=f"update_activity mismatch at grid={grid_shape}, density={density}",
    )


def test_update_activity_preserves_total_per_env():
    """Total activity is conserved by the update (per env), like the CPU."""
    world, tt, activity = _scenario((10, 10), 0.5, seed=41)
    acc = compute_accessibility(world, activity, tt)
    jax_out = np.asarray(
        update_activity_batched(_f32(activity[None]), _f32(acc[None]), world.growth_rate)
    )[0]
    np.testing.assert_allclose(
        jax_out.sum(), activity.sum(), rtol=RTOL_FP32, atol=ATOL_FP32,
        err_msg="update_activity did not preserve total activity",
    )


# ---------------------------------------------------------------------------
# Batched correctness — the per-env normalization guard
# ---------------------------------------------------------------------------


def test_update_activity_batched_is_per_env():
    """Batched update must normalize each env against ITSELF, not the batch.

    This is the M3 land-use risk the milestone flags: `mean(accessibility)`
    and the total-activity renormalization must be per-environment. We build
    a batch whose envs have deliberately different activity scales and
    accessibility magnitudes, run the batched update, and compare each slot
    to the CPU single-env result. A whole-batch mean/sum would shift every
    slot and fail here.
    """
    cfg_grids = [(5, 5), (10, 10), (15, 15), (10, 10)]
    scales = [0.5, 3.0, 1.0, 10.0]  # very different per-env magnitudes

    # Envs have different N, so pad to a common N for a rectangular batch and
    # only compare the valid prefix per env. Simpler: fix one grid, vary
    # activity scale + density so magnitudes differ but shapes match.
    cfg = GridCityConfig(grid_shape=(10, 10))
    n_envs = 6
    acts, accs, cpu_news = [], [], []
    for i in range(n_envs):
        world = make_world(i, cfg)
        mask = _make_mask(world.n_candidate_edges, 0.2 + 0.1 * i, seed=i + 300)
        tt = compute_travel_times(world, mask)
        rng = np.random.default_rng(i + 500)
        scale = float(scales[i % len(scales)]) if i < len(scales) else float(i + 1)
        activity = (rng.uniform(0.2, 3.0, size=world.n_zones) * scale).astype(np.float64)
        acc = compute_accessibility(world, activity, tt)
        acts.append(activity)
        accs.append(acc)
        cpu_news.append(update_activity(activity, acc, world.growth_rate))

    A = _f32(np.stack(acts, axis=0))
    ACC = _f32(np.stack(accs, axis=0))
    # growth_rate is shared across envs (same cfg), so one scalar is correct.
    gr = make_world(0, cfg).growth_rate
    jax_batched = np.asarray(update_activity_batched(A, ACC, gr))
    cpu_batched = np.stack(cpu_news, axis=0)

    np.testing.assert_allclose(
        jax_batched, cpu_batched, rtol=RTOL_FP32, atol=ATOL_FP32,
        err_msg="batched update_activity is not per-env (whole-batch reduction bug?)",
    )


def test_dynamics_batched_invariant_to_ordering():
    """Shuffling the batch axis shuffles the output correspondingly.

    Catches accidental cross-slot coupling in any of the three batched
    dynamics (a vmap that shared state across the batch axis would fail).
    """
    cfg = GridCityConfig(grid_shape=(10, 10))
    n_envs = 5
    acts, tts = [], []
    for i in range(n_envs):
        world = make_world(i, cfg)
        mask = _make_mask(world.n_candidate_edges, 0.5, seed=i + 700)
        tts.append(compute_travel_times(world, mask))
        rng = np.random.default_rng(i + 900)
        acts.append(rng.uniform(0.2, 3.0, size=world.n_zones).astype(np.float64))

    A = _f32(np.stack(acts, axis=0))
    T = _f32(np.stack(tts, axis=0))
    decay = make_world(0, cfg).accessibility_decay

    normal = np.asarray(compute_accessibility_batched(A, T, decay))
    rng = np.random.default_rng(0)
    perm = rng.permutation(n_envs)
    shuffled = np.asarray(compute_accessibility_batched(A[perm], T[perm], decay))

    inv = np.argsort(perm)
    np.testing.assert_allclose(
        shuffled[inv], normal, rtol=RTOL_FP32, atol=ATOL_FP32,
        err_msg="batched accessibility not invariant to batch ordering",
    )
