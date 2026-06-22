"""Mission availability / FDIR: eclipse duty-cycle + single-event latch-up.

Fault masking keeps results correct but says nothing about whether the
compute was powered. Availability, the fraction of mission time the payload
is up, is a separate budget, driven by scheduled and unplanned downtime.

Two effects dominate availability for an LEO compute payload:

* **Eclipse duty-cycle.** A solar-powered satellite loses (part of) its
  power budget every orbit while in Earth's shadow. That is *scheduled*
  downtime, deterministic and periodic.
* **Single-event latch-up (SEL).** A heavy-ion strike above the device's
  SEL threshold can latch a parasitic structure; recovery is a
  power-cycle, i.e. ``reboot_time_s`` of *unplanned* downtime. SpaceLLM
  does not mask SEL (latch-up is a hardware concern); this module budgets
  it.

:func:`simulate_availability` runs a discrete-event simulation:
deterministic eclipse windows plus a Poisson stream of SEL events (a SEL
that lands while the box is already down is masked), unions the downtime
intervals, and reports availability, MTBF, and MTTR.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from spacellm._internal.sampling import RandomState


@dataclass(frozen=True, slots=True)
class AvailabilityProfile:
    """Inputs for an availability simulation.

    Attributes:
        sel_rate_per_s: Single-event-latch-up rate (events/second) for the
            device in this orbit. Derive from the heavy-ion flux above the
            device's ``sel_threshold_mev_cm2_per_mg`` × the SEL
            cross-section, or supply a measured rate.
        reboot_time_s: Downtime per SEL event (detect + power-cycle +
            reboot).
        orbit_period_s: Orbital period. LEO ISS ≈ 5560 s.
        eclipse_fraction: Fraction of each orbit spent in power-limited
            eclipse, in ``[0, 1)``. LEO ≈ 0.35.
        mission_duration_s: Total mission wall-clock to simulate.

    Raises:
        ValueError: On non-positive magnitudes or out-of-range fractions.
    """

    sel_rate_per_s: float
    reboot_time_s: float
    orbit_period_s: float
    eclipse_fraction: float
    mission_duration_s: float

    def __post_init__(self) -> None:
        if self.sel_rate_per_s < 0:
            raise ValueError("sel_rate_per_s must be non-negative")
        if self.reboot_time_s <= 0:
            raise ValueError("reboot_time_s must be positive")
        if self.orbit_period_s <= 0:
            raise ValueError("orbit_period_s must be positive")
        if not 0.0 <= self.eclipse_fraction < 1.0:
            raise ValueError("eclipse_fraction must be in [0, 1)")
        if self.mission_duration_s <= 0:
            raise ValueError("mission_duration_s must be positive")


@dataclass(frozen=True, slots=True)
class AvailabilityReport:
    """Availability budget for one mission.

    Attributes:
        availability: Fraction of mission time the compute was up, ``[0, 1]``.
        uptime_s: Total powered, non-rebooting seconds.
        total_downtime_s: Union of eclipse + reboot downtime.
        eclipse_downtime_s: Scheduled eclipse downtime.
        reboot_downtime_s: Unplanned downtime added by SEL reboots beyond
            what eclipse already covered.
        n_eclipse_cycles: Number of eclipse windows in the mission.
        n_sel_events: Total SEL events sampled.
        n_effective_reboots: SEL events that actually triggered a reboot
            (i.e. occurred while powered).
        n_masked_sel: SEL events masked because the box was already down.
        mtbf_s: Mean powered time between effective SEL reboots (``inf`` if
            none).
        mttr_s: Mean time to recover from an SEL (the reboot time).
    """

    availability: float
    uptime_s: float
    total_downtime_s: float
    eclipse_downtime_s: float
    reboot_downtime_s: float
    n_eclipse_cycles: int
    n_sel_events: int
    n_effective_reboots: int
    n_masked_sel: int
    mtbf_s: float
    mttr_s: float

    def summary(self) -> str:
        """A compact human-readable availability block."""
        mtbf = "∞" if math.isinf(self.mtbf_s) else f"{self.mtbf_s:.0f} s"
        return (
            "SpaceLLM availability budget\n"
            f"  availability        : {self.availability * 100:.3f} %\n"
            f"  eclipse downtime    : {self.eclipse_downtime_s:.0f} s "
            f"({self.n_eclipse_cycles} cycles)\n"
            f"  reboot downtime     : {self.reboot_downtime_s:.0f} s "
            f"({self.n_effective_reboots} reboots, {self.n_masked_sel} masked)\n"
            f"  MTBF (SEL)          : {mtbf}\n"
            f"  MTTR                : {self.mttr_s:.0f} s"
        )


def _eclipse_windows(profile: AvailabilityProfile) -> list[tuple[float, float]]:
    """Deterministic eclipse windows: sunlit first, then eclipse, per orbit."""
    if profile.eclipse_fraction == 0.0:
        return []
    period = profile.orbit_period_s
    sunlit = period * (1.0 - profile.eclipse_fraction)
    mission = profile.mission_duration_s
    n_orbits = math.ceil(mission / period)
    windows: list[tuple[float, float]] = []
    for k in range(n_orbits):
        start = k * period + sunlit
        end = (k + 1) * period
        start = min(start, mission)
        end = min(end, mission)
        if end > start:
            windows.append((start, end))
    return windows


def _union_length(intervals: list[tuple[float, float]], horizon: float) -> float:
    """Total length of the union of ``intervals`` clipped to ``[0, horizon]``."""
    if not intervals:
        return 0.0
    clipped = sorted(
        (max(0.0, s), min(horizon, e)) for s, e in intervals if min(horizon, e) > max(0.0, s)
    )
    if not clipped:
        return 0.0
    total = 0.0
    cur_start, cur_end = clipped[0]
    for s, e in clipped[1:]:
        if s > cur_end:
            total += cur_end - cur_start
            cur_start, cur_end = s, e
        else:
            cur_end = max(cur_end, e)
    total += cur_end - cur_start
    return total


def _covered(t: float, intervals: list[tuple[float, float]]) -> bool:
    return any(s <= t < e for s, e in intervals)


def simulate_availability(
    profile: AvailabilityProfile,
    *,
    seed: int | None = None,
) -> AvailabilityReport:
    """Run the discrete-event availability simulation.

    Args:
        profile: Mission inputs.
        seed: RNG seed for the SEL Poisson stream. ``None`` is
            non-deterministic.

    Returns:
        An :class:`AvailabilityReport`.
    """
    mission = profile.mission_duration_s
    eclipse = _eclipse_windows(profile)
    eclipse_downtime = _union_length(eclipse, mission)

    rng = RandomState(seed=seed)
    # A homogeneous Poisson process: N ~ Poisson(λT), event times i.i.d.
    # Uniform[0, T] (sorted). Uses only the existing sampling primitives.
    n_sel = rng.poisson(profile.sel_rate_per_s * mission)
    sel_times = sorted(rng.random() * mission for _ in range(n_sel))

    reboot_intervals: list[tuple[float, float]] = []
    n_masked = 0
    for te in sel_times:
        # Masked if the box is already down (eclipse or an active reboot).
        if _covered(te, eclipse) or _covered(te, reboot_intervals):
            n_masked += 1
            continue
        reboot_intervals.append((te, min(te + profile.reboot_time_s, mission)))

    total_downtime = _union_length(eclipse + reboot_intervals, mission)
    reboot_downtime = total_downtime - eclipse_downtime
    uptime = mission - total_downtime
    availability = uptime / mission

    n_effective = len(reboot_intervals)
    sunlit_time = mission - eclipse_downtime
    mtbf = sunlit_time / n_effective if n_effective > 0 else math.inf

    return AvailabilityReport(
        availability=availability,
        uptime_s=uptime,
        total_downtime_s=total_downtime,
        eclipse_downtime_s=eclipse_downtime,
        reboot_downtime_s=reboot_downtime,
        n_eclipse_cycles=len(eclipse),
        n_sel_events=n_sel,
        n_effective_reboots=n_effective,
        n_masked_sel=n_masked,
        mtbf_s=mtbf,
        mttr_s=profile.reboot_time_s,
    )


__all__ = ["AvailabilityProfile", "AvailabilityReport", "simulate_availability"]
