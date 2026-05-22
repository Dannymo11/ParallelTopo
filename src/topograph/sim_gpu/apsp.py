"""JAX batched APSP via fixed-iteration min-plus matrix-squaring.

This is the core M2/M3 kernel: given a batch of direct-distance matrices
(walking floor + active transit overrides), compute all-pairs shortest-path
travel times in lockstep on the GPU.

**Algorithm — matrix-squaring.**
Starting from the direct-distance matrix A, iterate:

    D_0 = A
    D_{t+1} = min(D_t, D_t  ⊕  D_t)            where ⊕ is min-plus matmul

After K iterations, D contains shortest paths using up to 2^K hops. The
M2 accuracy study (`scripts/batched_apsp_accuracy.py`, 2026-05-22) showed
this converges to **bit-identical** scipy-FW output at K=5 on all M1-spec
grid sizes (5x5 through 15x15). At fp32 precision, "bit-identical" becomes
"within float32 round-off" — well below the M2 <2% mean-error gate.

**Why matrix-squaring instead of Bellman-Ford-style.**
The M2 sweep compared both (`D <- min(D, D⊕A)` vs `D <- min(D, D⊕D)`)
and matrix-squaring needed ~4x fewer iterations for the same convergence
(K=5 for 15x15 vs K=16). Same answer, 4x fewer min-plus matmuls per
APSP. There is no reason to implement BF in the GPU path.

**Why fp32.**
GPU fp64 is slow (typically 1/32 of fp32 throughput on consumer cards).
The M2 accuracy result holds at fp32: the relaxation converges to exact
shortest paths within fp32 round-off, which is many orders of magnitude
tighter than the 2% gate. The writeup line is "exact at fp32 precision."

**Memory note (M2 memory bar).**
The min-plus matmul intermediate is shape (B, N, N, N) which for the
worst M1 case (B=256, N=225) is ~11.7 GB at fp32. Fits a 24 GB GPU but
not by a huge margin; larger batches or N=400 (M5 sweep) may need
tiling on the K (intermediate-node) axis or bf16. Not addressed here —
flagged for M3.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

# The M2 decision: K=5 covers walking-floor diameters up to 32 hops (2^5),
# which is comfortably above the ~28-hop diameter of a 15x15 grid city.
# One safety iteration past the empirical first-convergence K=4 from the
# M2 sweep. Caller can override; almost no reason to.
DEFAULT_K_ITERATIONS: int = 5


# ---------------------------------------------------------------------------
# Single-graph kernel (the thing that gets vmapped)
# ---------------------------------------------------------------------------


def min_plus_matmul(A: jax.Array, B: jax.Array) -> jax.Array:
    """(A ⊕ B)[i, j] = min_k (A[i, k] + B[k, j]).

    3D broadcast: temporary tensor is (N, N, N). On GPU under vmap the
    batched version becomes (B_, N, N, N) — the dominant memory cost of
    the whole APSP. See the module docstring for the bound.
    """
    return jnp.min(A[:, :, None] + B[None, :, :], axis=1)


@partial(jax.jit, static_argnames=("k_iterations",))
def apsp_matrix_squaring(
    A: jax.Array,
    k_iterations: int = DEFAULT_K_ITERATIONS,
) -> jax.Array:
    """Single-graph APSP. `A` is the (N, N) direct-distance matrix.

    K iterations of `D <- min(D, D ⊕ D)`. K is a static (compile-time)
    argument so the loop unrolls cleanly under JIT. With K=5 the unroll
    is trivially small.
    """
    D = A
    for _ in range(k_iterations):
        D = jnp.minimum(D, min_plus_matmul(D, D))
    return D


# ---------------------------------------------------------------------------
# Batched kernel (what the simulator's step function calls)
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnames=("k_iterations",))
def apsp_batched(
    A_batch: jax.Array,
    k_iterations: int = DEFAULT_K_ITERATIONS,
) -> jax.Array:
    """Batched APSP across a (B, N, N) stack of direct-distance matrices.

    `A_batch[b]` is the direct-distance matrix for environment `b`. Returns
    a (B, N, N) tensor where entry `[b, i, j]` is the shortest-path travel
    time from zone i to zone j in env b under env b's current edge mask.

    Implementation: `jax.vmap` over the batch axis applies the single-graph
    kernel to each (N, N) slice in parallel on the GPU. The whole thing
    fuses into one compiled call under `jit`.
    """
    return jax.vmap(apsp_matrix_squaring, in_axes=(0, None))(A_batch, k_iterations)


__all__ = [
    "DEFAULT_K_ITERATIONS",
    "apsp_batched",
    "apsp_matrix_squaring",
    "min_plus_matmul",
]
