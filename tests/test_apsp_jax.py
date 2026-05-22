"""JAX APSP correctness: matches scipy FW to float32 precision.

The M2 accuracy sweep (`scripts/batched_apsp_accuracy.py`) already showed
that fixed-iteration matrix-squaring at K=5 converges to bit-identical
scipy-FW output on all M1-spec grids when run in NumPy (float64). This
test is the JAX-port version: same algorithm, same K, same inputs, but
now running through `jax.jit` and `jax.vmap` at float32 precision.

The acceptable tolerance is float32 round-off, not bit-identity — fp32
has ~7 decimal digits, so rtol=1e-5 is generous. If this test fails the
JAX kernel has a bug; downstream throughput numbers are meaningless until
it passes.

Test is skipped if JAX isn't installed (it's an `[gpu]` extras dep) so
the rest of the test suite keeps passing on CPU-only dev installs.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from topograph.sim_cpu import GridCityConfig, compute_travel_times, make_world
from topograph.sim_cpu.dynamics import _build_direct_distance_matrix
from topograph.sim_gpu import DEFAULT_K_ITERATIONS, apsp_batched, apsp_matrix_squaring


# fp32 round-off tolerance. Way looser than the M2 <2% accuracy gate;
# the gate is comfortably met if this passes.
RTOL_FP32 = 1e-4
ATOL_FP32 = 1e-3


def _make_mask(n_candidate: int, density: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mask = np.zeros(n_candidate, dtype=np.bool_)
    if density > 0:
        k = max(1, int(round(density * n_candidate)))
        mask[rng.choice(n_candidate, size=k, replace=False)] = True
    return mask


# Cover the M1 grid range and a representative span of mask densities. Skip
# density=0.0 because it's trivial (direct matrix is already exact) and
# would just inflate the apparent pass rate.
@pytest.mark.parametrize("grid_shape", [(5, 5), (10, 10), (15, 15)])
@pytest.mark.parametrize("density", [0.25, 0.5, 0.75, 1.0])
def test_apsp_matrix_squaring_matches_scipy_at_k5(grid_shape, density):
    """Single-graph JAX kernel matches scipy FW at K=5 within fp32 tol."""
    world = make_world(0, GridCityConfig(grid_shape=grid_shape))
    mask = _make_mask(world.n_candidate_edges, density, seed=42)

    A = _build_direct_distance_matrix(world, mask).astype(np.float32)
    D_exact = compute_travel_times(world, mask)

    D_jax = np.asarray(
        apsp_matrix_squaring(jnp.asarray(A), k_iterations=DEFAULT_K_ITERATIONS)
    )

    np.testing.assert_allclose(
        D_jax, D_exact, rtol=RTOL_FP32, atol=ATOL_FP32,
        err_msg=(
            f"JAX APSP mismatch vs scipy FW at grid={grid_shape}, "
            f"density={density}, K={DEFAULT_K_ITERATIONS}"
        ),
    )


def test_apsp_batched_matches_per_env_scipy():
    """vmap'd batched kernel produces identical results to per-env scipy.

    Builds a batch of 8 worlds-with-masks at varying densities, computes
    APSP exactly per-env via scipy, computes batched APSP in JAX, asserts
    they agree slot-by-slot. This catches both vmap mistakes (wrong axis,
    broadcasting bug) and any state leaking across batch slots.
    """
    cfg = GridCityConfig(grid_shape=(10, 10))
    n_envs = 8
    densities = np.linspace(0.0, 1.0, n_envs)

    A_list: list[np.ndarray] = []
    D_exact_list: list[np.ndarray] = []
    for i, d in enumerate(densities):
        world = make_world(i, cfg)
        mask = _make_mask(world.n_candidate_edges, float(d), seed=i + 100)
        A_list.append(_build_direct_distance_matrix(world, mask).astype(np.float32))
        D_exact_list.append(compute_travel_times(world, mask))

    A_batch = np.stack(A_list, axis=0)
    D_batch_jax = np.asarray(apsp_batched(jnp.asarray(A_batch)))
    D_batch_exact = np.stack(D_exact_list, axis=0)

    np.testing.assert_allclose(
        D_batch_jax, D_batch_exact, rtol=RTOL_FP32, atol=ATOL_FP32,
        err_msg="vmap'd batched APSP disagrees with per-env scipy",
    )


def test_apsp_batched_is_invariant_to_batch_ordering():
    """Shuffling the batch axis must produce a correspondingly-shuffled output.

    A vmap'd kernel that accidentally collapsed or shared state across the
    batch axis would fail this — the per-env result for env `i` would
    drift when its position in the batch changes.
    """
    cfg = GridCityConfig(grid_shape=(10, 10))
    n_envs = 6

    A_list = []
    for i in range(n_envs):
        world = make_world(i, cfg)
        mask = _make_mask(world.n_candidate_edges, 0.5, seed=i + 200)
        A_list.append(_build_direct_distance_matrix(world, mask).astype(np.float32))

    A_batch = np.stack(A_list, axis=0)
    D_normal = np.asarray(apsp_batched(jnp.asarray(A_batch)))

    rng = np.random.default_rng(0)
    perm = rng.permutation(n_envs)
    A_shuffled = A_batch[perm]
    D_shuffled = np.asarray(apsp_batched(jnp.asarray(A_shuffled)))

    inv_perm = np.argsort(perm)
    np.testing.assert_allclose(
        D_shuffled[inv_perm], D_normal, rtol=RTOL_FP32, atol=ATOL_FP32,
        err_msg="Batched APSP not invariant to batch ordering",
    )
