"""Policies for benchmarking and sanity testing.

Includes ``random``, ``greedy``, and (later) learned policies. Policies are
plain callables of signature ``policy(state, rng) -> action`` so the benchmark
harness can stay agnostic about how the action was produced.
"""
