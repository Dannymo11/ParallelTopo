"""GPU batched simulator (M2 kernels + M3 scaffolding).

M2 deliverable: `apsp_batched`, the JAX batched all-pairs shortest-path
kernel that lets a population of environments with *different active edge
sets* advance in lockstep. See `apsp.py` for the algorithm rationale.

M3 (in progress): the per-step dynamics are ported component by component,
each as a single-env function plus a vmapped batched entry point,
equivalence-tested against the CPU baseline. Landed: `compute_accessibility`,
`update_activity`, `compute_demand`, `apply_action`, `compute_welfare`
(`dynamics.py`), plus on-device direct-distance construction and the
assembled vmapped `step` / `rollout` over a batched ECS-like state
(`step.py`, with the M5 Fig-4 `freeze_mask` ablation flag), and the
vectorized uniform-random-legal policy + on-device `rollout_random` driver
(`policy.py`). Still to come: the M3 throughput re-run (time the full step
on device, feeding M5 Figure 1) and the M4 MLP-policy variant.
"""

from .apsp import (
    DEFAULT_K_ITERATIONS,
    apsp_batched,
    apsp_matrix_squaring,
    min_plus_matmul,
)
from .dynamics import (
    apply_action_batched,
    apply_action_single,
    compute_accessibility_batched,
    compute_accessibility_single,
    compute_demand_batched,
    compute_demand_single,
    compute_welfare_batched,
    compute_welfare_single,
    update_activity_batched,
    update_activity_single,
)
from .step import (
    GPUState,
    WorldArrays,
    build_direct_distance_single,
    reset_batched,
    rollout_batched,
    step_batched,
    step_single,
    world_arrays_from_config,
)
from .policy import (
    episode_returns,
    random_legal_action_batched,
    random_legal_action_single,
    rollout_random,
    valid_action_mask_batched,
    valid_action_mask_single,
)
from .mlp_policy import (
    DEFAULT_HIDDEN,
    MLPParams,
    feature_dim,
    init_mlp_params,
    masked_logits,
    mlp_forward,
    n_actions,
    reinforce_loss,
    rollout_mlp,
    state_features,
    train_step,
)

__all__ = [
    "DEFAULT_K_ITERATIONS",
    "apsp_batched",
    "apsp_matrix_squaring",
    "min_plus_matmul",
    "compute_accessibility_single",
    "compute_accessibility_batched",
    "update_activity_single",
    "update_activity_batched",
    "compute_demand_single",
    "compute_demand_batched",
    "apply_action_single",
    "apply_action_batched",
    "compute_welfare_single",
    "compute_welfare_batched",
    "WorldArrays",
    "GPUState",
    "world_arrays_from_config",
    "reset_batched",
    "build_direct_distance_single",
    "step_single",
    "step_batched",
    "rollout_batched",
    "valid_action_mask_single",
    "valid_action_mask_batched",
    "random_legal_action_single",
    "random_legal_action_batched",
    "rollout_random",
    "episode_returns",
    "MLPParams",
    "DEFAULT_HIDDEN",
    "feature_dim",
    "n_actions",
    "init_mlp_params",
    "state_features",
    "mlp_forward",
    "masked_logits",
    "rollout_mlp",
    "reinforce_loss",
    "train_step",
]
