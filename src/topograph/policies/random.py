"""Uniform-random-over-legal-actions policy.

Used as the throughput-floor policy in the benchmark harness, and as the
"baseline that should be beaten" comparator in sanity tests.

The policy is stateless and reads its randomness from the `rng` argument
passed by `run_episode`, so behavior is fully reproducible under a fixed
seed.
"""

from __future__ import annotations

import numpy as np

from topograph.sim_cpu import State, valid_action_mask


class RandomLegalPolicy:
    """Pick uniformly at random among the actions that pass `valid_action_mask`.

    Falls back to no-op if (somehow) no legal action exists — though in
    practice no-op is always legal, so the fallback is a defensive guard
    rather than expected behavior.
    """

    def __call__(self, state: State, rng: np.random.Generator) -> int:
        mask = valid_action_mask(state)
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            return state.world.no_op_action
        return int(rng.choice(legal))
