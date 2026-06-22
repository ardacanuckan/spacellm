"""Mission economics: turn an SEU rate into kWh, rollbacks, and a budget.

``spacellm.environments`` gives an SEU rate per bit per second. This module
maps that onto the operational cost of an on-orbit training run: re-flown
compute energy, rollback count, and the checkpoint interval that minimises
wasted work.

The chain is small and every step is a one-line formula:

1. **Fault rate.** ``λ = seu_rate_per_bit_per_s · n_bits · corrupting_fraction``
   faults per second, where ``corrupting_fraction`` is the share of raw
   upsets that actually produce a *silent, training-corrupting* fault (an
   exponent / high-MSB flip; mantissa-LSB flips are individually harmless,
   see the protection-module docs). ``MTBF = 1 / λ``.
2. **Optimal checkpoint interval.** Young's first-order result,
   ``τ_opt = √(2 · δ · MTBF)`` with ``δ`` the checkpoint write cost. This
   is the interval that minimises wasted work; checkpoint more often and
   you burn power writing checkpoints, less often and you re-fly more
   compute after each fault.
3. **Wasted energy with detection.** Each detected fault rolls back to the
   last checkpoint, re-flying ``τ/2`` of work on average plus a restore:
   ``E_waste = faults · (τ/2 + δ) · power``.
4. **Wasted energy without detection.** A *silent* corruption is the
   expensive case: with no integrity check you cannot trust any checkpoint
   taken after the (unknown) corruption, so discovery at run end forces a
   re-fly of the whole run. ``E_silent = P(≥1 fault) · power · run``. The
   gap between (4) and (3) is exactly the value SpaceLLM's detection layer
   buys, expressed in kWh.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_SECONDS_PER_DAY = 86_400.0
_JOULES_PER_KWH = 3.6e6


@dataclass(frozen=True, slots=True)
class MissionProfile:
    """Inputs describing one on-orbit training workload.

    Attributes:
        seu_rate_per_bit_per_s: Calibrated upset rate (e.g.
            ``PhysicsLiteEnvironment.mission_seu_rate_per_bit_per_s``).
        n_parameters: Number of model parameters.
        bytes_per_parameter: Storage width per parameter (2 for
            FP16/BF16, 4 for FP32). Defaults to 2.
        step_time_s: Wall-clock seconds per training step.
        power_w: Compute power draw of the workload, in watts.
        checkpoint_cost_s: Seconds to write (or restore) one checkpoint.
        corrupting_fraction: Fraction of raw SEUs that produce a silent,
            training-corrupting fault. Defaults to 0.25 (the FP32
            exponent-bit fraction, a reasonable first cut; tune from a
            sensitivity profile).
        run_duration_s: Total training-run wall-clock length, used for the
            no-detection comparison.

    Raises:
        ValueError: On any non-positive magnitude or out-of-range fraction.
    """

    seu_rate_per_bit_per_s: float
    n_parameters: int
    step_time_s: float
    power_w: float
    checkpoint_cost_s: float
    run_duration_s: float
    bytes_per_parameter: int = 2
    corrupting_fraction: float = 0.25

    def __post_init__(self) -> None:
        if self.seu_rate_per_bit_per_s < 0:
            raise ValueError("seu_rate_per_bit_per_s must be non-negative")
        if self.n_parameters <= 0:
            raise ValueError("n_parameters must be positive")
        if self.bytes_per_parameter <= 0:
            raise ValueError("bytes_per_parameter must be positive")
        if self.step_time_s <= 0:
            raise ValueError("step_time_s must be positive")
        if self.power_w <= 0:
            raise ValueError("power_w must be positive")
        if self.checkpoint_cost_s <= 0:
            raise ValueError("checkpoint_cost_s must be positive")
        if self.run_duration_s <= 0:
            raise ValueError("run_duration_s must be positive")
        if not 0.0 <= self.corrupting_fraction <= 1.0:
            raise ValueError("corrupting_fraction must be in [0, 1]")

    @property
    def n_bits(self) -> int:
        return self.n_parameters * self.bytes_per_parameter * 8


@dataclass(frozen=True, slots=True)
class MissionCostReport:
    """Budget numbers an operator can put in a mission-planning slide.

    Attributes:
        n_bits: Model state size in bits.
        fault_rate_per_s: Corrupting-fault rate ``λ``.
        mtbf_s: Mean time between corrupting faults (``inf`` if rate is 0).
        expected_faults_per_day: ``λ · 86400``.
        optimal_checkpoint_interval_s: Young-optimal interval ``τ_opt``.
        used_checkpoint_interval_s: Interval the rest of the report assumes
            (the supplied override, else ``optimal_checkpoint_interval_s``).
        expected_rollbacks_per_day: Detected faults forcing a rollback per
            day (equals ``expected_faults_per_day`` under full detection).
        wasted_energy_kwh_per_day: Re-flown compute per day *with*
            detection + checkpointing.
        checkpoint_overhead_kwh_per_day: Energy spent writing checkpoints
            per day.
        silent_redo_energy_kwh_per_run: Expected re-flown energy across the
            whole run if corruption is *undetected* (no SpaceLLM layer).
        detection_savings_kwh_per_run: ``silent_redo`` minus the
            with-detection rollback energy over the run, the kWh the
            integrity layer is worth.
    """

    n_bits: int
    fault_rate_per_s: float
    mtbf_s: float
    expected_faults_per_day: float
    optimal_checkpoint_interval_s: float
    used_checkpoint_interval_s: float
    expected_rollbacks_per_day: float
    wasted_energy_kwh_per_day: float
    checkpoint_overhead_kwh_per_day: float
    silent_redo_energy_kwh_per_run: float
    detection_savings_kwh_per_run: float

    def summary(self) -> str:
        """A compact human-readable budget block."""
        mtbf = "∞" if math.isinf(self.mtbf_s) else f"{self.mtbf_s:.0f} s"
        return (
            "SpaceLLM mission cost budget\n"
            f"  model state            : {self.n_bits / 8e9:.2f} GB ({self.n_bits:.3e} bits)\n"
            f"  corrupting-fault rate  : {self.fault_rate_per_s:.3e} /s  (MTBF {mtbf})\n"
            f"  faults / mission day   : {self.expected_faults_per_day:.3f}\n"
            f"  optimal checkpoint     : every {self.optimal_checkpoint_interval_s:.0f} s\n"
            f"  rollbacks / day        : {self.expected_rollbacks_per_day:.3f}\n"
            f"  wasted energy / day    : {self.wasted_energy_kwh_per_day:.4f} kWh "
            f"(+{self.checkpoint_overhead_kwh_per_day:.4f} kWh checkpoint overhead)\n"
            f"  silent redo / run      : {self.silent_redo_energy_kwh_per_run:.3f} kWh "
            f"if undetected\n"
            f"  detection saves / run  : {self.detection_savings_kwh_per_run:.3f} kWh"
        )


def estimate_mission_cost(
    profile: MissionProfile,
    *,
    checkpoint_interval_s: float | None = None,
) -> MissionCostReport:
    """Compute the mission energy / rollback budget for one workload.

    Args:
        profile: The workload description.
        checkpoint_interval_s: Override the checkpoint cadence. ``None``
            (default) uses the Young-optimal interval.

    Returns:
        A :class:`MissionCostReport`.

    Raises:
        ValueError: If ``checkpoint_interval_s`` is supplied but
            non-positive.
    """
    if checkpoint_interval_s is not None and checkpoint_interval_s <= 0:
        raise ValueError("checkpoint_interval_s must be positive")

    lam = profile.seu_rate_per_bit_per_s * profile.n_bits * profile.corrupting_fraction
    delta = profile.checkpoint_cost_s
    power = profile.power_w
    run = profile.run_duration_s

    if lam <= 0.0:
        # No corrupting faults expected: nothing to checkpoint against.
        tau_opt = run
        used_tau = checkpoint_interval_s if checkpoint_interval_s is not None else run
        return MissionCostReport(
            n_bits=profile.n_bits,
            fault_rate_per_s=0.0,
            mtbf_s=math.inf,
            expected_faults_per_day=0.0,
            optimal_checkpoint_interval_s=tau_opt,
            used_checkpoint_interval_s=used_tau,
            expected_rollbacks_per_day=0.0,
            wasted_energy_kwh_per_day=0.0,
            checkpoint_overhead_kwh_per_day=(_SECONDS_PER_DAY / used_tau)
            * delta
            * power
            / _JOULES_PER_KWH,
            silent_redo_energy_kwh_per_run=0.0,
            detection_savings_kwh_per_run=0.0,
        )

    mtbf = 1.0 / lam
    # Young's first-order optimal checkpoint interval, capped at run length.
    tau_opt = min(math.sqrt(2.0 * delta * mtbf), run)
    used_tau = checkpoint_interval_s if checkpoint_interval_s is not None else tau_opt

    faults_per_day = lam * _SECONDS_PER_DAY
    # Each detected fault re-flies τ/2 of work on average plus a restore.
    rework_time_per_fault = 0.5 * used_tau + delta
    wasted_time_per_day = faults_per_day * rework_time_per_fault
    wasted_energy_kwh_per_day = wasted_time_per_day * power / _JOULES_PER_KWH

    checkpoints_per_day = _SECONDS_PER_DAY / used_tau
    checkpoint_overhead_kwh_per_day = checkpoints_per_day * delta * power / _JOULES_PER_KWH

    # Without detection: a silent corruption is invisible until eval, and no
    # checkpoint after the (unknown) corruption is trustworthy, so a corrupted
    # run must be re-flown. This is the conservative *single re-fly* cost:
    # P(≥1 fault) · whole-run energy. (It is a lower bound, a badly corrupted
    # run can need several re-flights, but a bounded number is what a planning
    # slide wants.)
    p_corruption_in_run = 1.0 - math.exp(-lam * run)
    run_energy_kwh = power * run / _JOULES_PER_KWH
    silent_redo_energy_kwh_per_run = p_corruption_in_run * run_energy_kwh

    # With detection across the run: roll back ~τ/2 + δ of work per fault
    # instead of re-flying the whole run.
    detect_rework_kwh_per_run = lam * run * rework_time_per_fault * power / _JOULES_PER_KWH
    # When detection-rework meets or exceeds a single re-fly, the fault rate is
    # so high that detection alone is marginal, an honest 0, not a negative.
    detection_savings_kwh_per_run = max(
        silent_redo_energy_kwh_per_run - detect_rework_kwh_per_run,
        0.0,
    )

    return MissionCostReport(
        n_bits=profile.n_bits,
        fault_rate_per_s=lam,
        mtbf_s=mtbf,
        expected_faults_per_day=faults_per_day,
        optimal_checkpoint_interval_s=tau_opt,
        used_checkpoint_interval_s=used_tau,
        expected_rollbacks_per_day=faults_per_day,
        wasted_energy_kwh_per_day=wasted_energy_kwh_per_day,
        checkpoint_overhead_kwh_per_day=checkpoint_overhead_kwh_per_day,
        silent_redo_energy_kwh_per_run=silent_redo_energy_kwh_per_run,
        detection_savings_kwh_per_run=detection_savings_kwh_per_run,
    )


__all__ = ["MissionCostReport", "MissionProfile", "estimate_mission_cost"]
