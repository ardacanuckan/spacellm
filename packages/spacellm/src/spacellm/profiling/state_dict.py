"""Profile a flat state-dict, when there is no live nn.Module to walk.

The CLI accepts paths to saved checkpoints (``.safetensors``, ``.pt``,
``.pth``, ``.bin``, or HuggingFace model directories) without
requiring the ``transformers`` import path. The state-dict-only
profiler bridges that gap: each tensor becomes one row in a
:class:`~spacellm.profiling.report.ProfileReport` keyed by its dotted
parameter name.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from spacellm.profiling.report import LayerSensitivity, ProfileReport


def state_dict_profile(state_dict: Mapping[str, torch.Tensor]) -> ProfileReport:
    """Build a :class:`ProfileReport` from a state-dict.

    Each tensor is treated as one "layer", the dotted key becomes the
    ``layer_name``, and ``module_class`` records the dtype. ``n_bits``
    is computed from element-size × 8.

    Args:
        state_dict: Mapping of parameter name to tensor, the format
            ``torch.load`` and ``safetensors.torch.load_file`` both
            return.

    Returns:
        A :class:`ProfileReport` whose layers carry only the static
        fields (no perturbation data).
    """
    layers: list[LayerSensitivity] = []
    for name, tensor in state_dict.items():
        n = int(tensor.numel())
        layers.append(
            LayerSensitivity(
                layer_name=name,
                module_class=str(tensor.dtype).removeprefix("torch."),
                n_parameters=n,
                n_bits=n * tensor.element_size() * 8,
            ),
        )
    return ProfileReport(layers=layers)


__all__ = ["state_dict_profile"]
