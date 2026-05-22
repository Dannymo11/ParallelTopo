"""GPU batched simulator (M2 kernels + M3 scaffolding).

M2 deliverable: `apsp_batched`, the JAX batched all-pairs shortest-path
kernel that lets a population of environments with *different active edge
sets* advance in lockstep. See `apsp.py` for the algorithm rationale.

M3 will add the rest of the per-step dynamics (demand, accessibility,
land-use update, reward, action application) and a vmapped `step` function.
"""

from .apsp import (
    DEFAULT_K_ITERATIONS,
    apsp_batched,
    apsp_matrix_squaring,
    min_plus_matmul,
)

__all__ = [
    "DEFAULT_K_ITERATIONS",
    "apsp_batched",
    "apsp_matrix_squaring",
    "min_plus_matmul",
]
