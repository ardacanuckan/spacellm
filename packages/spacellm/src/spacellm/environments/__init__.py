"""Radiation environments, fault sources for SpaceLLM.

The environment hierarchy follows three implementation tiers (per
``packages/docs/docs/architecture.md``):

* **Level 1, statistical** (``StatisticalEnvironment``): the simplest
  Poisson-process bit-flip injector. Available in v0.1.
* **Level 2, physics-lite**: orbital flux models (AE-9 / AP-9) convolved with
  device-specific Weibull cross-sections. Lands in v0.3.
* **Level 3, external bridges**: ingest SPENVIS / OLTARIS / Geant4 outputs.
  Lands in v0.9.

All environments share the same :class:`Environment` ABC, so any module that
consumes a fault source (``profiling``, ``bench``, ``runtime.harden``) treats
them interchangeably.
"""

from __future__ import annotations

from spacellm.environments import devices, orbits, physics
from spacellm.environments.base import Environment
from spacellm.environments.mcu import MCUDistribution, MCUEnvironment
from spacellm.environments.orbits import (
    GEO_QUIET,
    LEO_ISS_NOMINAL,
    MARS_TRANSIT,
    OrbitProfile,
)
from spacellm.environments.physics_lite import PhysicsLiteEnvironment
from spacellm.environments.statistical import StatisticalEnvironment

__all__ = [
    "GEO_QUIET",
    "LEO_ISS_NOMINAL",
    "MARS_TRANSIT",
    "Environment",
    "MCUDistribution",
    "MCUEnvironment",
    "OrbitProfile",
    "PhysicsLiteEnvironment",
    "StatisticalEnvironment",
    "devices",
    "orbits",
    "physics",
]
