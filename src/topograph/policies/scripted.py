"""Scripted (replay) policy.

Plays back a fixed sequence of action indices, one per step. Past the end
of the sequence, falls back to the no-op (or a caller-provided fallback).

Stateless: action lookup is keyed by `state.step`, not by an internal
counter, so the same policy instance is reusable across runs without an
explicit reset.

Used by sanity tests to construct deterministic "hand-built corridor"
trajectories and by the bench harness for replay-mode timing.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from topograph.sim_cpu import Action, State


class ScriptedPolicy:
    def __init__(
        self,
        actions: Sequence[Action],
        fallback: Action | None = None,
    ) -> None:
        self._actions = np.asarray(list(actions), dtype=np.int64)
        self._fallback = fallback

    def __call__(self, state: State, rng: np.random.Generator) -> int:
        del rng
        t = state.step
        if t < self._actions.shape[0]:
            return int(self._actions[t])
        if self._fallback is not None:
            return int(self._fallback)
        return int(state.world.no_op_action)
