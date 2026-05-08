"""Policies for benchmarking and sanity testing.

Includes ``random``, ``greedy``, and ``scripted``. Policies are plain
callables of signature ``policy(state, rng) -> action`` so the benchmark
harness can stay agnostic about how the action was produced.
"""

from .greedy import OneStepGreedyPolicy
from .random import RandomLegalPolicy
from .scripted import ScriptedPolicy

__all__ = [
    "OneStepGreedyPolicy",
    "RandomLegalPolicy",
    "ScriptedPolicy",
]
