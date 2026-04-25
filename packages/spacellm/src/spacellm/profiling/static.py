"""Static profile, parameter count + bit area per module."""

from __future__ import annotations

from torch import nn

from spacellm.profiling.report import LayerSensitivity, ProfileReport


def static_profile(model: nn.Module, *, include_root: bool = False) -> ProfileReport:
    """Cheap pass, every leaf module's parameter count and bit-area.

    Direct (non-recursive) parameters only, that way the report rows
    align one-to-one with what a strategy might actually wrap, instead of
    double-counting nested children.

    Args:
        model: Module to profile.
        include_root: If ``True``, the root module (``""`` path) is included.

    Returns:
        A :class:`ProfileReport`.
    """
    layers: list[LayerSensitivity] = []
    for name, module in model.named_modules():
        if not include_root and name == "":
            continue
        own_params = list(module.parameters(recurse=False))
        n_params = sum(p.numel() for p in own_params)
        n_bits = sum(p.numel() * p.element_size() * 8 for p in own_params)
        layers.append(
            LayerSensitivity(
                layer_name=name,
                module_class=type(module).__name__,
                n_parameters=n_params,
                n_bits=n_bits,
            ),
        )
    return ProfileReport(layers=layers)


__all__ = ["static_profile"]
