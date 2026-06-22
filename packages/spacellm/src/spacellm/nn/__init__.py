"""PyTorch-style protected modules.

Mirrors the role of :mod:`torch.nn`: drop-in replacements for standard
modules that incorporate radiation-tolerance at the layer boundary. Used
internally by :mod:`spacellm.protection` strategies and exposed publicly so
power users can hand-pick replacements without going through a strategy.
"""

from __future__ import annotations

from spacellm.nn.abft import ABFTLinear
from spacellm.nn.layer_norm import TMRLayerNorm
from spacellm.nn.tmr import TMRLinear

__all__ = ["ABFTLinear", "TMRLayerNorm", "TMRLinear"]
