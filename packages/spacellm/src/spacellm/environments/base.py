"""Abstract base class for radiation environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

import torch

from spacellm.types import FaultEvent, RadiationStep


class Environment(ABC):
    """Common interface for all environments.

    Subclasses choose how to fulfil the contract, pure statistics (L1),
    physics models (L2), external simulators (L3), but every consumer of an
    environment relies on these three operations:

    1. :meth:`step`: advance simulated time by ``dt`` and return the
       instantaneous flux/dose snapshot.
    2. :meth:`sample_faults`: draw concrete bit-flip events for a window of
       duration ``dt`` over a tensor population.
    3. :meth:`reset`: restore initial state for a fresh run.

    Implementations should be deterministic given the same seed, so that runs
    are reproducible, irreproducible fault injection is a useless benchmark.
    """

    @abstractmethod
    def step(self, dt: float) -> RadiationStep:
        """Advance simulated time by ``dt`` seconds.

        Args:
            dt: Time delta in seconds, must be positive.

        Returns:
            The :class:`~spacellm.types.RadiationStep` characterising flux and
            dose at the new instant.

        Raises:
            ValueError: If ``dt`` is non-positive.
        """

    @abstractmethod
    def sample_faults(
        self,
        tensors: Mapping[str, torch.Tensor],
        dt: float,
    ) -> list[FaultEvent]:
        """Sample bit-flip events for a window of duration ``dt``.

        ``tensors`` is a mapping of ``tensor_id`` → tensor. The mapping is *not*
        modified by this call; the caller (or a protection hook) is responsible
        for replaying the returned events onto live state.

        Args:
            tensors: Named tensors that may receive faults.
            dt: Window length in seconds, must be positive.

        Returns:
            A list of :class:`~spacellm.types.FaultEvent`; possibly empty.

        Raises:
            ValueError: If ``dt`` is non-positive.
        """

    def reset(self) -> None:  # noqa: B027, intentional no-op default
        """Reset internal state. Default implementation is a no-op."""

    @property
    @abstractmethod
    def t(self) -> float:
        """Current simulated time in seconds since the last :meth:`reset`."""


__all__ = ["Environment"]
