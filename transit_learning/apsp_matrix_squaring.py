"""PyTorch port of TopoGraph's batched APSP kernel (min-plus matrix-squaring).

This is a faithful translation of ParallelTopo's JAX kernel
(`src/topograph/sim_gpu/apsp.py`) into PyTorch, so the *exact same algorithm*
that delivered the 50-60x GPU speedup in TopoGraph can be benchmarked against
transit_learning's `torch_utils.floyd_warshall` on the same inputs and inside
the same (PyTorch) runtime.

Algorithm -- matrix-squaring
----------------------------
Starting from the direct-distance (edge-cost) matrix A, iterate:

    D_0       = A
    D_{t+1}   = min(D_t, D_t  (+)  D_t)        where (+) is min-plus matmul

After K iterations, D contains shortest paths using up to 2^K hops. Compared
with a Bellman-Ford-style relaxation (`D <- min(D, D (+) A)`, +1 hop per
iteration), matrix-squaring needs ~log2(diameter) iterations instead of
~diameter -- the whole reason TopoGraph chose it.

Differences vs transit_learning's Floyd-Warshall
-------------------------------------------------
* Floyd-Warshall is exact in N sequential outer iterations and ALSO returns the
  `nexts` predecessor matrix for path reconstruction. Matrix-squaring returns
  *distances only*. Anything downstream that needs the actual path sequences
  (transit_learning's `reconstruct_all_paths` / `shortest_path_sequences`)
  still needs predecessor tracking -- this kernel is a drop-in for the distance
  half of FW, not the path half.
* "Exact at K" holds only when 2^K >= graph diameter (in hops). TopoGraph uses
  K=5 because its grid-city walking floor bounds the hop-diameter at <=~28.
  Arbitrary transit graphs need their K validated -- see `compare.py`'s
  convergence-K sweep.

Conventions (matching transit_learning.floyd_warshall)
------------------------------------------------------
* Input is an edge-cost matrix: entry (i, j) is the cost of edge i->j, or
  +inf if no such edge. Diagonal should be 0.
* Accepts (N, N) or batched (B, N, N). Returns the same rank.
"""

from __future__ import annotations

import math

import torch

# TopoGraph's default; covers 2^5 = 32 hops. Valid for M1-spec grid cities and
# (as compare.py confirms) for Mandl/Mumford, but should be checked per graph.
DEFAULT_K_ITERATIONS: int = 5


def k_for_n_nodes(n_nodes: int, safety: int = 1) -> int:
    """Smallest K with 2^K >= (n_nodes - 1), plus `safety` extra iterations.

    n_nodes - 1 is the worst-case hop-diameter of any connected graph, so this
    K guarantees convergence to exact shortest paths for *any* topology of that
    size -- the conservative choice when the real diameter is unknown.
    """
    if n_nodes <= 2:
        return 1 + safety
    return int(math.ceil(math.log2(n_nodes - 1))) + safety


def min_plus_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """(A (+) B)[..., i, j] = min_k (A[..., i, k] + B[..., k, j]).

    Works for (N, N) and batched (B, N, N). The intermediate broadcast tensor
    is (..., N, N, N) -- the dominant memory cost, exactly as flagged in
    TopoGraph's apsp.py. XLA fused this away on GPU; PyTorch eager will
    materialize it, so very large N at large batch can be memory-heavy (the
    harness reports peak memory so this is measurable).
    """
    if A.dim() == 2:
        # (i, k, 1) + (1, k, j) -> (i, k, j); min over k (axis 1)
        return torch.amin(A[:, :, None] + B[None, :, :], dim=1)
    # batched: (b, i, k, 1) + (b, 1, k, j) -> (b, i, k, j); min over k (axis 2)
    return torch.amin(A[:, :, :, None] + B[:, None, :, :], dim=2)


def apsp_matrix_squaring(
    A: torch.Tensor,
    k_iterations: int = DEFAULT_K_ITERATIONS,
) -> torch.Tensor:
    """All-pairs shortest-path distances via K min-plus matrix-squarings.

    `A` is an (N, N) or (B, N, N) edge-cost matrix (+inf for missing edges,
    0 on the diagonal). Returns shortest-path distances of the same shape.
    """
    D = A
    for _ in range(k_iterations):
        D = torch.minimum(D, min_plus_matmul(D, D))
    return D


__all__ = [
    "DEFAULT_K_ITERATIONS",
    "k_for_n_nodes",
    "min_plus_matmul",
    "apsp_matrix_squaring",
]
