"""Inputs for the APSP comparison: real Mandl/Mumford graphs + synthetic batches.

The real city graphs (Mandl, Mumford0-3) are the *exact* edge-cost matrices
transit_learning evaluates on -- whitespace-separated travel-time matrices with
"Inf" for missing edges and 0 on the diagonal. These are the honest
single-graph correctness inputs.

For throughput we also need *batches* of graphs (TopoGraph's whole speedup is
batching APSP across many environments). Two ways to get a batch:
  * `replicate_batch`  -- the same city graph stacked B times (apples-to-apples
    with the real topology; a stand-in for "many similar cities").
  * `synthetic_batch`  -- B random connected geometric graphs of N nodes, to
    sweep batch/size independently of the fixed real datasets.

Everything returns plain torch tensors so it feeds both transit_learning's
`floyd_warshall` and the matrix-squaring kernel unchanged.
"""

from __future__ import annotations

from pathlib import Path

import torch

# Where transit_learning keeps the Mumford archive's extracted Instances.
DEFAULT_INSTANCES_DIR = Path(
    "/Users/dannymo/dev/transit_learning/datasets/mumford_dataset/Instances"
)

CITY_NODES = {
    "Mandl": 15,
    "Mumford0": 30,
    "Mumford1": 70,
    "Mumford2": 110,
    "Mumford3": 127,
}


def load_city_edge_costs(
    city: str,
    instances_dir: Path | str = DEFAULT_INSTANCES_DIR,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Load a city's (N, N) edge-cost matrix: travel time, +inf off-edge, 0 diag.

    Parses `<city>TravelTimes.txt`. Values "Inf"/"inf" -> +inf. The diagonal is
    forced to exactly 0 (some files store 0 already; this guards the rest).
    """
    path = Path(instances_dir) / f"{city}TravelTimes.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Point --instances-dir at the Mumford "
            f"'Instances' directory (see transit_learning/README.md)."
        )
    rows: list[list[float]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append([float("inf") if t.lower() == "inf" else float(t)
                     for t in line.split()])
    mat = torch.tensor(rows, dtype=dtype)
    if mat.shape[0] != mat.shape[1]:
        raise ValueError(f"{city}: non-square matrix {tuple(mat.shape)}")
    mat.fill_diagonal_(0.0)
    return mat


def available_cities(instances_dir: Path | str = DEFAULT_INSTANCES_DIR) -> list[str]:
    """Cities whose TravelTimes file is present, in node-count order."""
    d = Path(instances_dir)
    return [c for c in CITY_NODES if (d / f"{c}TravelTimes.txt").exists()]


def replicate_batch(edge_costs: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Stack a single (N, N) edge-cost matrix into a (B, N, N) batch."""
    return edge_costs.unsqueeze(0).expand(batch_size, -1, -1).contiguous()


def synthetic_batch(
    batch_size: int,
    n_nodes: int,
    edge_keep_prob: float = 0.3,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """B random *connected* geometric graphs as a (B, N, N) edge-cost batch.

    Nodes are random 2D points; an edge exists with prob `edge_keep_prob`, with
    cost = Euclidean distance. A ring (i -> i+1) is always added so every graph
    is connected (otherwise FW and matrix-squaring agree only on +inf, an
    uninteresting comparison). Symmetric, 0 on the diagonal.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    pos = torch.rand(batch_size, n_nodes, 2, generator=g)
    # pairwise euclidean distances, (B, N, N)
    dist = torch.cdist(pos, pos)
    keep = torch.rand(batch_size, n_nodes, n_nodes, generator=g) < edge_keep_prob
    keep = keep & keep.transpose(1, 2)  # symmetric
    # guarantee connectivity with a ring
    ring = torch.zeros(n_nodes, n_nodes, dtype=torch.bool)
    idx = torch.arange(n_nodes)
    ring[idx, (idx + 1) % n_nodes] = True
    ring = ring | ring.T
    keep = keep | ring
    edge_costs = torch.where(keep, dist, torch.full_like(dist, float("inf")))
    eye = torch.eye(n_nodes, dtype=torch.bool)
    edge_costs[:, eye] = 0.0
    return edge_costs.to(dtype=dtype, device=device)


__all__ = [
    "DEFAULT_INSTANCES_DIR",
    "CITY_NODES",
    "load_city_edge_costs",
    "available_cities",
    "replicate_batch",
    "synthetic_batch",
]
