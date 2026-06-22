"""Training-time silent-data-corruption (SDC) guard.

The other strategies in :mod:`spacellm.protection` defend inference: weights
and KV-caches. Training adds a second failure mode, silent data corruption:
a bit-flip in a gradient or in optimizer state (Adam's ``exp_avg`` /
``exp_avg_sq``) during the step itself. The loss curve stays monotone, the
run keeps producing tokens, and the corruption is written into the next
checkpoint. (See the README FAQ on checkpointing, and the "silent data
corruption at scale" literature.)

``GradientGuard`` is a primitive (like :class:`KVCacheParity`): ``apply``
is a no-op because gradients and optimizer state live *outside* the
``nn.Module`` graph and only exist transiently inside the training loop.
The user calls :meth:`check_gradients` after ``loss.backward()`` and
:meth:`check_optimizer_state` after / around ``optimizer.step()``.

Two detectors, both deterministic and history-free on the first call:

* **Non-finite.** An exponent flip that drives a value to ``inf`` / ``nan``
  is unambiguous corruption. Always flagged.
* **Magnitude spike.** A per-tensor exponential moving average of the
  L2 norm; a finite value that jumps past ``spike_factor`` × the running
  average is the fingerprint of a high-exponent-bit flip in one element.

With ``recover=True`` the guard *sanitises*: non-finite entries are zeroed
and a spiking tensor is renormalised back to the spike threshold, turning
a run-poisoning corruption into a bounded, logged perturbation.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from spacellm.protection.base import ProtectionReport, Strategy

_TINY = 1e-12


class GradientGuard(Strategy):
    """Detect (and optionally sanitise) corruption in gradients / optimizer state.

    Args:
        spike_factor: A finite tensor whose L2 norm exceeds this multiple
            of its running-average norm is flagged as a spike. Defaults to
            100.0.
        ema_decay: Decay of the running-average norm in ``[0, 1)``. Higher
            tracks more slowly. Defaults to 0.9.
        recover: If ``True``, sanitise flagged tensors in place (zero
            non-finite entries, renormalise spikes). Defaults to ``True``.

    Raises:
        ValueError: If ``spike_factor`` ≤ 1 or ``ema_decay`` ∉ ``[0, 1)``.
    """

    name = "gradient_guard"

    def __init__(
        self,
        *,
        spike_factor: float = 100.0,
        ema_decay: float = 0.9,
        recover: bool = True,
    ) -> None:
        if spike_factor <= 1.0:
            raise ValueError(f"spike_factor must be > 1, got {spike_factor}")
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError(f"ema_decay must be in [0, 1), got {ema_decay}")
        self.spike_factor = float(spike_factor)
        self.ema_decay = float(ema_decay)
        self.recover = bool(recover)
        self._ema: dict[str, float] = {}
        self._n_checks = 0
        self._n_detected = 0
        self._n_recovered = 0
        self._n_non_finite = 0
        self._n_spikes = 0

    def apply(self, model: nn.Module) -> nn.Module:
        # Gradients/optimizer state are not part of the module graph.
        return model

    def check_gradients(self, model: nn.Module) -> int:
        """Inspect every parameter's ``.grad`` after a backward pass.

        Returns:
            Number of gradient tensors flagged as corrupted.
        """
        named = ((name, p.grad) for name, p in model.named_parameters() if p.grad is not None)
        return self._scan(named, prefix="grad")

    def check_optimizer_state(self, optimizer: torch.optim.Optimizer) -> int:
        """Inspect optimizer state tensors (e.g. Adam ``exp_avg``/``exp_avg_sq``).

        Returns:
            Number of state tensors flagged as corrupted.
        """
        items: list[tuple[str, torch.Tensor]] = []
        for gi, group in enumerate(optimizer.param_groups):
            for pi, param in enumerate(group["params"]):
                state = optimizer.state.get(param, {})
                for key, value in state.items():
                    if isinstance(value, torch.Tensor) and value.is_floating_point():
                        items.append((f"opt.{gi}.{pi}.{key}", value))
        return self._scan(iter(items), prefix="opt")

    def _scan(
        self,
        named_tensors: Iterable[tuple[str, torch.Tensor]],
        *,
        prefix: str,
    ) -> int:
        flagged = 0
        for name, tensor in named_tensors:
            self._n_checks += 1
            key = f"{prefix}:{name}"
            is_bad = False

            finite_mask = torch.isfinite(tensor)
            has_non_finite = not bool(finite_mask.all().item())
            if has_non_finite:
                is_bad = True
                self._n_non_finite += 1
                if self.recover:
                    tensor[~finite_mask] = 0.0

            norm = float(tensor.detach().norm().item())
            baseline = self._ema.get(key)
            if baseline is not None and norm > self.spike_factor * max(baseline, _TINY):
                is_bad = True
                self._n_spikes += 1
                if self.recover and norm > _TINY:
                    threshold_norm = self.spike_factor * max(baseline, _TINY)
                    tensor.mul_(threshold_norm / norm)
                    norm = threshold_norm

            if is_bad:
                self._n_detected += 1
                flagged += 1
                if self.recover:
                    self._n_recovered += 1

            # Update the running average with the (possibly sanitised) norm so
            # a corruption spike never poisons the baseline.
            sane_norm = float(tensor.detach().norm().item())
            if baseline is None:
                self._ema[key] = sane_norm
            else:
                self._ema[key] = self.ema_decay * baseline + (1.0 - self.ema_decay) * sane_norm
        return flagged

    def collect_report(self) -> ProtectionReport:
        return ProtectionReport(
            strategy_name=self.name,
            n_corruptions_detected=self._n_detected,
            n_corruptions_recovered=self._n_recovered,
            n_checks_performed=self._n_checks,
            overhead_bytes=len(self._ema) * 8,
            extra={
                "n_non_finite": float(self._n_non_finite),
                "n_spikes": float(self._n_spikes),
                "n_tracked_tensors": float(len(self._ema)),
            },
        )

    def reset(self) -> None:
        self._n_checks = 0
        self._n_detected = 0
        self._n_recovered = 0
        self._n_non_finite = 0
        self._n_spikes = 0


__all__ = ["GradientGuard"]
