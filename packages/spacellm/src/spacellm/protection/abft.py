"""Algorithm-based fault tolerance as a composable protection strategy."""

from __future__ import annotations

from torch import nn

from spacellm.nn.abft import ABFTLinear
from spacellm.protection._module_paths import set_submodule, walk_named_submodules
from spacellm.protection.base import ProtectionReport, Strategy


class ABFTProtection(Strategy):
    """Wrap the largest ``top_k_percent`` of linear layers with ABFT.

    This is the low-memory counterpart to :class:`SelectiveTMR`: instead of
    storing three weight replicas (200 % memory overhead), each protected
    layer keeps two checksum vectors (``row`` of length ``out_features`` and
    ``col`` of length ``in_features``, well under 1 % of the weight) and
    verifies both the stored weight *and* the matmul result on every
    forward. A single corrupted weight element is detected and corrected;
    a transient compute fault is detected and replayed. See
    :class:`spacellm.nn.abft.ABFTLinear` for the math.

    Args:
        top_k_percent: Percentage of eligible ``nn.Linear`` modules to
            protect, in ``(0, 100]``. Defaults to 100, ABFT is cheap
            enough to apply broadly, unlike TMR.
        rtol: Relative tolerance forwarded to each ``ABFTLinear``.
        atol: Absolute tolerance forwarded to each ``ABFTLinear``.
        correct: If ``True`` (default) protected layers correct single
            weight flips and replay on transient mismatch; if ``False``
            they are detect-only.

    Raises:
        ValueError: If ``top_k_percent`` is outside ``(0, 100]``.
    """

    name = "abft"

    def __init__(
        self,
        top_k_percent: float = 100.0,
        *,
        rtol: float = 1e-3,
        atol: float = 1e-5,
        correct: bool = True,
    ) -> None:
        if not 0.0 < top_k_percent <= 100.0:
            raise ValueError(f"top_k_percent must be in (0, 100], got {top_k_percent}")
        self.top_k_percent = float(top_k_percent)
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.correct = bool(correct)
        self._wrapped_paths: list[str] = []
        self._overhead_bytes: int = 0
        self._modules: list[ABFTLinear] = []

    def apply(self, model: nn.Module) -> nn.Module:
        candidates: list[tuple[str, nn.Linear, int]] = []
        for path, module in walk_named_submodules(model):
            # ABFTLinear is an nn.Module, not an nn.Linear, so an already-
            # wrapped layer is never a candidate on re-application.
            if isinstance(module, nn.Linear):
                size = sum(p.numel() for p in module.parameters(recurse=False))
                candidates.append((path, module, size))

        if not candidates:
            return model

        candidates.sort(key=lambda item: -item[2])
        n_select = max(1, round(len(candidates) * self.top_k_percent / 100.0))

        for path, module, _ in candidates[:n_select]:
            wrapped = ABFTLinear(
                module,
                rtol=self.rtol,
                atol=self.atol,
                correct=self.correct,
            )
            set_submodule(model, path, wrapped)
            self._wrapped_paths.append(path)
            self._modules.append(wrapped)
            self._overhead_bytes += (
                wrapped._row_checksum.element_size() * wrapped._row_checksum.numel()
                + wrapped._col_checksum.element_size() * wrapped._col_checksum.numel()
            )
        return model

    def collect_report(self) -> ProtectionReport:
        return ProtectionReport(
            strategy_name=self.name,
            n_corruptions_detected=sum(m.n_detected for m in self._modules),
            n_corruptions_recovered=sum(m.n_corrected for m in self._modules),
            n_checks_performed=sum(m.n_checks for m in self._modules),
            overhead_bytes=self._overhead_bytes,
            extra={
                "n_wrapped_modules": float(len(self._wrapped_paths)),
                "top_k_percent": self.top_k_percent,
            },
        )

    def reset(self) -> None:
        for m in self._modules:
            m.n_checks = 0
            m.n_detected = 0
            m.n_corrected = 0

    @property
    def wrapped_paths(self) -> tuple[str, ...]:
        """Tuple of dotted paths wrapped with ABFT. Useful for tests."""
        return tuple(self._wrapped_paths)


__all__ = ["ABFTProtection"]
