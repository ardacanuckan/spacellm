"""Level-2 physics-calibrated radiation environment."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch

from spacellm.environments.base import Environment
from spacellm.environments.orbits import OrbitProfile
from spacellm.environments.physics import irpp_seu_rate_per_cell
from spacellm.environments.statistical import StatisticalEnvironment
from spacellm.types import DeviceModel, FaultCause, FaultEvent, RadiationStep


class PhysicsLiteEnvironment(Environment):
    """Physics-calibrated bit-flip environment.

    Combines a :class:`~spacellm.types.DeviceModel`'s Weibull cross-section
    with an :class:`~spacellm.environments.orbits.OrbitProfile`'s LET
    spectrum to derive a calibrated SEU rate per bit per second, then
    delegates fault sampling to a :class:`StatisticalEnvironment` driven
    by that rate.

    Computation
    -----------
    1. ``rate_per_cell_per_s = ∫ σ(L) · dF/dL · dL`` (IRPP).
    2. ``rate_per_bit_per_s = rate_per_cell / bits_per_cell``.
    3. ``effective_rate = rate_per_bit_per_s × time_acceleration``.

    Args:
        device: Silicon characterisation. Provides Weibull cross-section
            parameters.
        orbit: Orbit characterisation. Provides LET spectrum + dose rate.
        bits_per_cell: How many bits one sensitive cell represents. For
            6T SRAM and most DRAM, ``cell ≈ bit``; defaults to 1.0.
        time_acceleration: Multiplier on the SEU rate. Realistic LEO
            rates (~1e-12 / bit / s) would never produce events in the
            seconds typical of a unit test; setting ``time_acceleration``
            > 1 lets demos compress mission time. Set to 1.0 for
            mission-realistic injection.
        seed: RNG seed for reproducibility.

    Raises:
        ValueError: If ``bits_per_cell`` or ``time_acceleration`` are
            non-positive.
    """

    def __init__(
        self,
        device: DeviceModel,
        orbit: OrbitProfile,
        *,
        bits_per_cell: float = 1.0,
        time_acceleration: float = 1.0,
        seed: int | None = None,
    ) -> None:
        if bits_per_cell <= 0:
            raise ValueError(f"bits_per_cell must be positive, got {bits_per_cell}")
        if time_acceleration <= 0:
            raise ValueError(f"time_acceleration must be positive, got {time_acceleration}")

        self.device = device
        self.orbit = orbit
        self.bits_per_cell = float(bits_per_cell)
        self.time_acceleration = float(time_acceleration)

        rate_per_cell_per_s = irpp_seu_rate_per_cell(
            device=device,
            let_grid=orbit.let_grid_mev_cm2_per_mg,
            flux_per_let_per_s=orbit.flux_per_let_per_s,
        )
        self._mission_rate_per_bit_per_s = rate_per_cell_per_s / self.bits_per_cell
        effective_rate = self._mission_rate_per_bit_per_s * self.time_acceleration

        self._heavy_ion_flux = float(
            np.trapezoid(orbit.flux_per_let_per_s, orbit.let_grid_mev_cm2_per_mg),
        )

        self._inner = StatisticalEnvironment(
            rate_per_bit_per_s=effective_rate,
            seed=seed,
            cause=FaultCause.SEU_HEAVY_ION,
        )

    @property
    def mission_seu_rate_per_bit_per_s(self) -> float:
        """Pre-acceleration SEU rate. The realistic-mission number."""
        return self._mission_rate_per_bit_per_s

    @property
    def effective_seu_rate_per_bit_per_s(self) -> float:
        """Mission rate × time_acceleration, what actually drives sampling."""
        return self._mission_rate_per_bit_per_s * self.time_acceleration

    @property
    def t(self) -> float:
        return self._inner.t

    def step(self, dt: float) -> RadiationStep:
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        self._inner.step(dt)
        return RadiationStep(
            t=self._inner.t,
            proton_flux=self.orbit.proton_flux_above_10mev_per_cm2_per_s,
            heavy_ion_flux=self._heavy_ion_flux,
            seu_rate_per_bit_per_s=self.effective_seu_rate_per_bit_per_s,
            tid_rate_gy_per_s=self.orbit.tid_rate_gy_per_s,
            inside_saa=False,
            source=f"physics_lite:{self.orbit.name}",
        )

    def sample_faults(
        self,
        tensors: Mapping[str, torch.Tensor],
        dt: float,
    ) -> list[FaultEvent]:
        return self._inner.sample_faults(tensors, dt)

    def reset(self) -> None:
        self._inner.reset()


__all__ = ["PhysicsLiteEnvironment"]
