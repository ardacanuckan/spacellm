"""Level-1 statistical fault injection.

Models bit upsets as a homogeneous Poisson process with a configurable rate
per bit per second. No physics, just calibrated probabilistic sampling. The
fastest and most reproducible environment, appropriate for unit tests,
regression suites, and protection-strategy A/B comparisons.

For physics-calibrated rates derived from orbit + device, see
``spacellm.environments.physics_lite`` (lands in v0.3 per
``CHANGELOG.md``).
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from spacellm._internal.bitops import flip_bit_uniform
from spacellm._internal.sampling import RandomState
from spacellm.environments.base import Environment
from spacellm.types import FaultCause, FaultEvent, RadiationStep


class StatisticalEnvironment(Environment):
    """Poisson bit-flip injector.

    Args:
        rate_per_bit_per_s: Mean number of bit upsets per bit per second.
            Realistic frame of reference: COTS DRAM at LEO sees roughly
            1e-12 to 1e-10 (highly device-dependent, see
            ``packages/docs/docs/physics-primer.md``). For demonstrations, rates
            of 1e-6 to 1e-3 produce visible accuracy effects within seconds.
        seed: RNG seed for reproducibility. ``None`` is non-deterministic.
        cause: Attribution label for produced :class:`FaultEvent` instances.

    Raises:
        ValueError: If ``rate_per_bit_per_s`` is negative.
    """

    def __init__(
        self,
        rate_per_bit_per_s: float = 1e-9,
        *,
        seed: int | None = None,
        cause: FaultCause = FaultCause.SYNTHETIC,
    ) -> None:
        if rate_per_bit_per_s < 0:
            raise ValueError(
                f"rate_per_bit_per_s must be non-negative, got {rate_per_bit_per_s}",
            )
        self._rate = float(rate_per_bit_per_s)
        self._cause = cause
        self._t = 0.0
        self._rng = RandomState(seed=seed)

    @property
    def t(self) -> float:
        return self._t

    def step(self, dt: float) -> RadiationStep:
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        self._t += dt
        return RadiationStep(
            t=self._t,
            proton_flux=0.0,
            heavy_ion_flux=0.0,
            seu_rate_per_bit_per_s=self._rate,
            tid_rate_gy_per_s=0.0,
            inside_saa=False,
            source="statistical",
        )

    def sample_faults(
        self,
        tensors: Mapping[str, torch.Tensor],
        dt: float,
    ) -> list[FaultEvent]:
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        if self._rate == 0.0:
            return []
        events: list[FaultEvent] = []
        for name, tensor in tensors.items():
            if tensor.numel() == 0:
                continue
            n_bits = tensor.numel() * tensor.element_size() * 8
            mean = n_bits * self._rate * dt
            count = self._rng.poisson(mean)
            for _ in range(count):
                bit_index = self._rng.randint(0, n_bits)
                events.append(self._build_event(name, tensor, bit_index))
        return events

    def reset(self) -> None:
        self._t = 0.0

    def _build_event(
        self,
        name: str,
        tensor: torch.Tensor,
        bit_index: int,
    ) -> FaultEvent:
        before, after, bit_position = flip_bit_uniform(tensor, bit_index)
        return FaultEvent(
            t=self._t,
            tensor_id=name,
            bit_index=bit_index,
            bit_position=bit_position,
            before=before,
            after=after,
            cause=self._cause,
        )


__all__ = ["StatisticalEnvironment"]
