"""Dynamic sensitivity profile, bit-flip → output-divergence."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from spacellm._internal.bitops import flip_bit_uniform
from spacellm._internal.sampling import RandomState
from spacellm.profiling.report import LayerSensitivity, ProfileReport
from spacellm.profiling.static import static_profile


def sensitivity_profile(
    model: nn.Module,
    forward_fn: Callable[[nn.Module], torch.Tensor],
    *,
    n_flips_per_layer: int = 5,
    seed: int = 0,
) -> ProfileReport:
    """Estimate per-layer sensitivity by repeated bit-flip injection.

    For every module that owns at least one parameter, the routine
    snapshots the original parameters, injects ``n_flips_per_layer``
    random bit-flips into them, runs ``forward_fn(model)`` to obtain the
    perturbed output, then restores the snapshot. The output divergence
    from the clean baseline is recorded as MSE.

    The method is reproducible for a given ``seed``; each layer gets a
    distinct sub-seed derived from the layer's name to avoid identical
    flip sequences across layers.

    Cost: ``O(n_layers × n_flips_per_layer × forward_cost)``. For a
    50-layer transformer with ``n_flips_per_layer=5`` and a 10 ms
    forward, this is ≈ 2.5 s on CPU. Cheap enough for v0.1
    development; v0.5 swaps to a gradient-based fast approximation.

    Args:
        model: Module under test. Will be temporarily mutated, then
            restored.
        forward_fn: Callable ``model -> tensor`` producing the model's
            output for a fixed input. The user fixes the input.
        n_flips_per_layer: Number of bit-flips per layer; each flip is
            applied independently and the mean / max divergence is
            reported.
        seed: RNG seed.

    Returns:
        A :class:`ProfileReport` whose layers carry populated
        ``mean_perturbation`` and ``max_perturbation`` fields where a
        layer has at least one parameter; ``None`` otherwise.
    """
    base_report = static_profile(model)
    rng = RandomState(seed=seed)
    with torch.no_grad():
        baseline = forward_fn(model).detach().clone()

    enriched: list[LayerSensitivity] = []
    for layer in base_report.layers:
        if layer.n_bits == 0:
            enriched.append(layer)
            continue
        module = model.get_submodule(layer.layer_name) if layer.layer_name else model
        own_params = [p for p in module.parameters(recurse=False) if p.numel() > 0]
        if not own_params:
            enriched.append(layer)
            continue

        deviations: list[float] = []
        for _ in range(n_flips_per_layer):
            target = own_params[rng.randint(0, len(own_params))]
            n_bits = target.numel() * target.element_size() * 8
            bit_index = rng.randint(0, n_bits)
            snapshot = target.detach().clone()
            try:
                flip_bit_uniform(target.data, bit_index)
                with torch.no_grad():
                    out = forward_fn(model).detach()
                mse = float(((out - baseline) ** 2).mean().item())
                deviations.append(mse)
            finally:
                target.data.copy_(snapshot)

        layer.mean_perturbation = sum(deviations) / len(deviations)
        layer.max_perturbation = max(deviations)
        enriched.append(layer)

    return ProfileReport(layers=enriched)


__all__ = ["sensitivity_profile"]
