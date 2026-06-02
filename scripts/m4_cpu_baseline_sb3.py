"""M4 item 4 (SKETCH): CPU training-throughput baseline via SB3 VecEnv.

STATUS: scaffold, not yet wired into the results pipeline. This is the
realistic CPU *training* baseline the M4 plan calls for — a multi-process
Stable-Baselines3 `SubprocVecEnv` running PPO on the same transit task —
so the end-to-end training speedup can be reported as

    GPU end-to-end (scripts/m4_train_throughput.py, train_step driver)
    --------------------------------------------------------------------
    CPU end-to-end (this script, SubprocVecEnv env-steps/sec)

It is deliberately the single-process scipy CPU number's honest successor:
SB3 with N worker processes is what a practitioner would actually run on a
CPU box, and its host-side `VectorEnv` step + the policy forward on CPU are
exactly the boundary costs the on-device GPU design avoids.

WHY IT IS A SKETCH (not run here):
  * It adds heavy dependencies NOT in the project's GPU image —
    `stable-baselines3`, `gymnasium`, `torch`. Keep these out of the Modal
    image; run this baseline on a CPU box only.
        pip install "stable-baselines3>=2.3" "gymnasium>=0.29" torch
  * The point is env-steps/sec at the training boundary, NOT convergence —
    same non-goals as the rest of M4 (no tuning, no policy-quality eval).

The Gymnasium env below wraps the existing `sim_cpu` simulator with the
SAME observation features as the GPU MLP policy (`sim_gpu.mlp_policy
.state_features`), so the two sides are measured on equivalent work.

TODO before this produces a headline number:
  1. Confirm the SB3 import path + VecEnv worker count for the target CPU.
  2. Decide net arch to match the GPU side (64-unit MLP) for a fair compare.
  3. Time env-steps/sec over a fixed number of PPO `learn` steps, warm.
  4. Emit results/m4_cpu_baseline/<ts>.json with the same schema spirit as
     m4_train_throughput.py, then divide in a small combine step.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from topograph.sim_cpu import GridCityConfig, make_world, reset, step

HORIZON = 15
GRID_SHAPE = (10, 10)
HIDDEN_UNITS = 64
SEED = 0


def _features(state) -> np.ndarray:
    """Match `sim_gpu.mlp_policy.state_features`: activity, edge_mask, budget
    fraction, step fraction — so the CPU and GPU sides see equivalent input.
    """
    world = state.world
    budget_frac = state.budget_remaining / float(world.initial_budget)
    step_frac = state.step / float(world.horizon)
    return np.concatenate([
        np.asarray(state.activity, dtype=np.float32),
        state.edge_mask.astype(np.float32),
        np.array([budget_frac, step_frac], dtype=np.float32),
    ])


def make_gym_env():
    """Build a Gymnasium env wrapping `sim_cpu`. Imports gymnasium lazily so
    the module is importable without the SB3 stack installed."""
    try:
        import gymnasium as gym
        from gymnasium import spaces
    except ImportError as e:  # pragma: no cover - sketch guard
        raise SystemExit(
            "This baseline needs gymnasium + stable-baselines3 + torch:\n"
            '  pip install "stable-baselines3>=2.3" "gymnasium>=0.29" torch'
        ) from e

    class TransitEnv(gym.Env):
        """One transit-construction episode as a Gymnasium env.

        Observation: the shared feature vector. Action: Discrete(E+1); an
        illegal-but-in-range action is a silent no-op (the sim_cpu contract),
        so no action masking is required for a throughput baseline.
        """

        metadata = {"render_modes": []}

        def __init__(self):
            super().__init__()
            self.world = make_world(
                SEED, GridCityConfig(grid_shape=GRID_SHAPE, horizon=HORIZON)
            )
            n_actions = self.world.no_op_action + 1
            obs_dim = self.world.n_zones + self.world.n_candidate_edges + 2
            self.action_space = spaces.Discrete(n_actions)
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )
            self.state = None

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.state = reset(self.world)
            return _features(self.state), {}

        def step(self, action):
            self.state, reward, done, _info = step(self.state, int(action))
            obs = _features(self.state)
            return obs, float(reward), bool(done), False, {}

    return TransitEnv


def main() -> None:
    EnvCls = make_gym_env()
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import SubprocVecEnv
    except ImportError as e:  # pragma: no cover - sketch guard
        raise SystemExit(
            "This baseline needs stable-baselines3 + torch:\n"
            '  pip install "stable-baselines3>=2.3" "gymnasium>=0.29" torch'
        ) from e

    # TODO(1): tune n_envs to the target CPU's core count.
    n_envs = 8
    vec_env = SubprocVecEnv([EnvCls for _ in range(n_envs)])

    # TODO(2): match the GPU side's 64-unit MLP for a fair comparison.
    model = PPO(
        "MlpPolicy", vec_env, seed=SEED, device="cpu",
        policy_kwargs=dict(net_arch=[HIDDEN_UNITS, HIDDEN_UNITS]),
        n_steps=HORIZON, batch_size=n_envs * HORIZON, verbose=0,
    )

    # TODO(3): warm once, then time env-steps/sec over a fixed budget.
    warmup_steps = n_envs * HORIZON * 4
    timed_steps = n_envs * HORIZON * 40
    model.learn(total_timesteps=warmup_steps)  # warm caches / JIT-free CPU
    t0 = time.perf_counter()
    model.learn(total_timesteps=timed_steps, reset_num_timesteps=False)
    elapsed = time.perf_counter() - t0

    env_steps_per_sec = timed_steps / elapsed
    print(f"CPU SB3 SubprocVecEnv ({n_envs} workers): "
          f"{env_steps_per_sec:,.0f} env-steps/sec")
    print("Compare against the GPU train_step env-steps/sec from "
          "scripts/m4_train_throughput.py for the RQ3 end-to-end speedup.")

    # TODO(4): write results/m4_cpu_baseline/<ts>.json and add a small
    # combine step that divides the GPU train-step number by this one.
    vec_env.close()


if __name__ == "__main__":
    main()
