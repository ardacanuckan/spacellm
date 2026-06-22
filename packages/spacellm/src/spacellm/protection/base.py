"""Strategy ABC and the per-strategy telemetry record."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from torch import nn


@dataclass(slots=True)
class ProtectionReport:
    """Telemetry snapshot for one strategy at one point in time.

    Mutable so the runtime can update counters in place during long runs.
    Aggregated across strategies in :class:`spacellm.runtime.RuntimeReport`
    (lands in v0.1 task #4).

    Attributes:
        strategy_name: Stable identifier for the strategy that produced
            this report (e.g. ``"selective_tmr"``).
        n_corruptions_detected: Mismatches found by integrity checks.
        n_corruptions_recovered: Detected corruptions that the strategy
            corrected (e.g. via majority vote). For *detect-only*
            strategies this is always zero.
        n_checks_performed: How many integrity checks ran during the
            covered window.
        overhead_bytes: Memory overhead introduced by this strategy
            (replicas, parity tensors, fingerprints).
        extra: Free-form additional metrics keyed by name.
    """

    strategy_name: str
    n_corruptions_detected: int = 0
    n_corruptions_recovered: int = 0
    n_checks_performed: int = 0
    overhead_bytes: int = 0
    extra: dict[str, float] = field(default_factory=dict)


class Strategy(ABC):
    """Base class for all protection strategies.

     A strategy is described declaratively, then materialised by
     :meth:`apply` mutating the model in place (or returning a wrapped one).
     After application, the model behaves as a normal :class:`torch.nn.Module`
    , protected layers are wrapped or hooked, transparent to downstream
     code. :meth:`collect_report` returns telemetry, and :meth:`reset` clears
     accumulated counters without unwinding the structural changes.

     Strategies should be cheap to construct and have no side-effects until
     :meth:`apply` runs.
    """

    name: str = "strategy"

    @abstractmethod
    def apply(self, model: nn.Module) -> nn.Module:
        """Mutate ``model`` to apply this strategy.

        Returns:
            The (same) model. Returned for fluent composition.
        """

    @abstractmethod
    def collect_report(self) -> ProtectionReport:
        """Return a snapshot of telemetry accumulated since the last reset."""

    def reset(self) -> None:  # noqa: B027, intentional no-op default
        """Reset accumulated telemetry. Default is a no-op."""


__all__ = ["ProtectionReport", "Strategy"]
