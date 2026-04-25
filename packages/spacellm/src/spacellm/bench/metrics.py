"""Benchmark metrics."""

from __future__ import annotations

import torch


def silent_error_rate(
    output: torch.Tensor,
    reference: torch.Tensor,
    *,
    threshold: float = 1e-3,
) -> float:
    """Fraction of output elements that deviate from the reference past
    ``threshold`` in absolute value.

    A "silent error" is the failure mode SpaceLLM explicitly cares about:
    the model produces output that the user *trusts*, but it is in fact
    corrupted by a SEU. ``silent_error_rate`` is the simplest scalar
    that quantifies this, fraction-of-diverged-elements is the unique
    KPI radiation-hardened ML brings beyond the standard accuracy
    metric. (See the bench module.)

    NaNs in either tensor count as silent errors (they certainly diverge
    from a finite reference). This matters because exponent-bit flips
    routinely produce NaNs.

    Args:
        output: Model output under fault injection. Any shape.
        reference: Trusted output (typically the clean run). Same shape.
        threshold: Absolute deviation that counts as a silent error.

    Returns:
        Fraction in [0, 1]: 0.0 means perfectly clean, 1.0 means every
        element deviated past the threshold.

    Raises:
        ValueError: If shapes mismatch or ``output`` / ``reference`` is
            empty.
    """
    if output.shape != reference.shape:
        raise ValueError(
            f"output and reference must match in shape; got "
            f"{tuple(output.shape)} vs {tuple(reference.shape)}",
        )
    if output.numel() == 0:
        raise ValueError("output must not be empty")
    out = output.detach().to(torch.float32)
    ref = reference.detach().to(torch.float32)
    diff = (out - ref).abs()
    # NaNs in either tensor → counted as silent errors.
    deviated = torch.where(
        torch.isnan(diff) | (diff > threshold),
        torch.ones_like(diff),
        torch.zeros_like(diff),
    )
    return float(deviated.mean().item())


__all__ = ["silent_error_rate"]
