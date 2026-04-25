"""Sensitivity profiler, input to top-k protection ranking.

Two modes are exposed in v0.1:

* :func:`static_profile`, a cheap pass that ranks every leaf module by
  parameter count and bit-area. Useful as a sanity check and as the
  default ranking source for :class:`spacellm.protection.SelectiveTMR`.

* :func:`sensitivity_profile`, a dynamic mode that runs the model under
  a forward function, then per-layer injects ``n_flips`` random bit-flips
  and measures output divergence (mean / max squared deviation) from the
  clean baseline. v0.5 will swap this for a faster gradient-based approx.
"""

from __future__ import annotations

from spacellm.profiling.report import LayerSensitivity, ProfileReport
from spacellm.profiling.sensitivity import sensitivity_profile
from spacellm.profiling.state_dict import state_dict_profile
from spacellm.profiling.static import static_profile

__all__ = [
    "LayerSensitivity",
    "ProfileReport",
    "sensitivity_profile",
    "state_dict_profile",
    "static_profile",
]
