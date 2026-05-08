"""One-step lookahead greedy policy.

At each step, the policy iterates over the legal actions, calls `step` once
per action to peek at the resulting reward, and picks the action that
maximizes immediate reward. Ties are broken by lowest action index for
determinism.

This is a *baseline* heuristic — it ignores the multi-step structure of the
problem entirely — but it's exactly the right comparator for the
"greedy beats random" sanity test: if a one-step-greedy heuristic can't
beat uniform random on the simulator's own reward, the reward landscape
itself is degenerate.

Cost: O(action_dim) `step` calls per environment step. Not vectorized; the
M3 GPU port will replace this with a proper batched lookahead.
"""

from __future__ import annotations

import numpy as np

from topograph.sim_cpu import State, step, valid_action_mask


class OneStepGreedyPolicy:
    def __call__(self, state: State, rng: np.random.Generator) -> int:
        del rng  # deterministic
        mask = valid_action_mask(state)
        legal = np.flatnonzero(mask)
        # Default: no-op (always legal), reward 0 if step returns 0.
        best_action = int(state.world.no_op_action)
        best_reward = -np.inf
        for a in legal:
            _next, reward, _done, _info = step(state, int(a))
            # Strict ">" preserves lowest-index tie-break.
            if reward > best_reward:
                best_reward = float(reward)
                best_action = int(a)
        return best_action
