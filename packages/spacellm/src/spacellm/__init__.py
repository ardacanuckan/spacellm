"""SpaceLLM, the open framework for radiation-tolerant transformer training and inference in space.

Public API surface (stable through 0.x patch releases):

- ``spacellm.__version__``
- ``spacellm.types``, ``RadiationStep``, ``FaultEvent``, ``RunRecord``, ``DeviceModel``
- ``spacellm.environments``, fault sources
- ``spacellm.protection``, hardening strategies (added in v0.1)
- ``spacellm.runtime.harden``, high-level entry point (added in v0.1)
- ``spacellm.profiling``, sensitivity profiler (added in v0.1)
- ``spacellm.observability``, run tracking (added in v0.1)
- ``spacellm.nn``, PyTorch-style hardened modules (added in v0.1)
- ``spacellm.training``, training-side protection (v0.7+)
- ``spacellm.bench``, SpaceBench (v0.6+)
"""

from __future__ import annotations

from spacellm import (
    availability,
    bench,
    environments,
    mission,
    nn,
    observability,
    profiling,
    protection,
    reliability,
    runtime,
)
from spacellm._version import __version__
from spacellm.availability import (
    AvailabilityProfile,
    AvailabilityReport,
    simulate_availability,
)
from spacellm.mission import MissionCostReport, MissionProfile, estimate_mission_cost
from spacellm.reliability import (
    ReliabilityStack,
    ResidualRateReport,
    compose_residual_rate,
)
from spacellm.runtime.harden import HardenedModel, harden
from spacellm.types import (
    DeviceModel,
    FaultCause,
    FaultEvent,
    RadiationStep,
    RunRecord,
)

__all__ = [
    "AvailabilityProfile",
    "AvailabilityReport",
    "DeviceModel",
    "FaultCause",
    "FaultEvent",
    "HardenedModel",
    "MissionCostReport",
    "MissionProfile",
    "RadiationStep",
    "ReliabilityStack",
    "ResidualRateReport",
    "RunRecord",
    "__version__",
    "availability",
    "bench",
    "compose_residual_rate",
    "environments",
    "estimate_mission_cost",
    "harden",
    "mission",
    "nn",
    "observability",
    "profiling",
    "protection",
    "reliability",
    "runtime",
    "simulate_availability",
]
