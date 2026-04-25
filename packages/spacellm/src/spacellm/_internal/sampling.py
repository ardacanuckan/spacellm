"""Reproducible random sampling primitives.

We use ``numpy.random.Generator`` (the modern API since NumPy 1.17) rather than
``random`` or the legacy ``numpy.random.RandomState``. PyTorch's RNG is kept
deliberately separate, environments seed *fault* RNG; users seed *model* RNG.
Confusing the two is the most common cause of irreproducible "fault injection"
experiments.
"""

from __future__ import annotations

import numpy as np


class RandomState:
    """Thin, typed wrapper around :class:`numpy.random.Generator`.

    Args:
        seed: Seed for the underlying generator. ``None`` → unseeded
            (non-deterministic).
    """

    __slots__ = ("_rng",)

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def poisson(self, mean: float) -> int:
        """Sample one Poisson-distributed non-negative integer with the given mean."""
        if mean <= 0.0:
            return 0
        return int(self._rng.poisson(mean))

    def randint(self, low: int, high: int) -> int:
        """Uniform integer in ``[low, high)``. Half-open like Python's range."""
        if not high > low:
            raise ValueError(f"high ({high}) must be greater than low ({low})")
        return int(self._rng.integers(low, high))

    def random(self) -> float:
        """Uniform float in ``[0.0, 1.0)``."""
        return float(self._rng.random())


__all__ = ["RandomState"]
