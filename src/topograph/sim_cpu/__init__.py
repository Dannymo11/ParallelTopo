"""CPU baseline simulator (M1).

NumPy/SciPy implementation of the slim transit simulator. Exposes the canonical
``reset(world) -> state``, ``step(state, action) -> (state, reward, done, info)``
API that the GPU port (``sim_gpu``) mirrors at batch dimension 1.

Public surface kept deliberately small so the contract is easy to audit.
"""

from .api import (
    Policy,
    find_edge_index,
    make_world,
    reset,
    run_episode,
    step,
    valid_action_mask,
)
from .types import Action, GridCityConfig, State, Trajectory, WorldConfig

__all__ = [
    "Action",
    "GridCityConfig",
    "Policy",
    "State",
    "Trajectory",
    "WorldConfig",
    "find_edge_index",
    "make_world",
    "reset",
    "run_episode",
    "step",
    "valid_action_mask",
]
