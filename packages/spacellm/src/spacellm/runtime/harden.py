"""High-level :func:`harden` entry point and :class:`HardenedModel` handle."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from spacellm.protection.base import ProtectionReport, Strategy

if TYPE_CHECKING:
    from torch import nn

    from spacellm.environments.base import Environment


@dataclass
class HardenedModel:
    """A lightweight handle returned by :func:`harden`.

    The underlying ``model`` is mutated in place by the strategies, but the
    handle gives uniform access to telemetry, an attached environment, and
    the strategies themselves. Calling the handle delegates to the model.

    Attributes:
        model: The protected model. Calling forward via ``hardened(x)`` is
            equivalent to ``hardened.model(x)``.
        strategies: The strategies that were applied, in the order applied.
        environment: An optional :class:`~spacellm.environments.base.Environment`
            attached at hardening time, handy for evaluation runs that
            need to be reproducible from the model alone.
    """

    model: nn.Module
    strategies: list[Strategy] = field(default_factory=list)
    environment: Environment | None = None

    def collect_reports(self) -> list[ProtectionReport]:
        """Snapshot telemetry from every attached strategy."""
        return [s.collect_report() for s in self.strategies]

    def reset_telemetry(self) -> None:
        """Reset accumulated telemetry on every strategy. Structural
        wrapping is *not* unwound, strategies are designed to be
        applied once."""
        for s in self.strategies:
            s.reset()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)


def harden(
    model: nn.Module,
    *,
    strategies: Sequence[Strategy],
    environment: Environment | None = None,
) -> HardenedModel:
    """Apply protection strategies to ``model`` and return a handle.

    The model is mutated in place, wrapping, hooks, fingerprints, and any
    other side effects each strategy implements. The strategies are applied
    in order; if two strategies select overlapping modules, the second one
    sees the first one's output (e.g. wrapping a TMR-protected module with
    a checksum is fine; the checksum sees the wrapped module).

    Example
    -------

    .. code-block:: python

        import spacellm as sl
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
        hardened = sl.harden(
            model,
            strategies=[
                sl.protection.SelectiveTMR(top_k_percent=5),
                sl.protection.AttentionChecksum(),
                sl.protection.KVCacheParity(),
            ],
        )
        out = hardened.generate(input_ids, max_new_tokens=128)
        for r in hardened.collect_reports():
            print(r)

    Args:
        model: The :class:`torch.nn.Module` to harden.
        strategies: Strategies to apply, in order.
        environment: An optional environment attached to the handle for
            reproducibility, strategies do not consume it directly.

    Returns:
        A :class:`HardenedModel` handle.
    """
    for s in strategies:
        s.apply(model)
    return HardenedModel(model=model, strategies=list(strategies), environment=environment)


__all__ = ["HardenedModel", "harden"]
