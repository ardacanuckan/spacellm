"""TMR-protected LayerNorm and RMSNorm."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812, torch convention


def _is_rmsnorm(module: nn.Module) -> bool:
    """Detect a PyTorch :class:`torch.nn.RMSNorm` (added in PyTorch 2.4)."""
    rms = getattr(nn, "RMSNorm", None)
    return rms is not None and isinstance(module, rms)


class TMRLayerNorm(nn.Module):
    """Drop-in replacement for :class:`nn.LayerNorm` and :class:`nn.RMSNorm` with 3× redundancy.

    Both LayerNorm and RMSNorm are tiny in parameter count yet
    disproportionately critical: a corrupted γ scales *every* activation
    flowing through the layer. TMR-ing them is one of the cheapest
    high-impact defences in the catalogue (FT-Transformer reports 13.9 %
    overhead and 97.2 % detection for normalisation-targeted protection;
    see the protection docs).

    Memory: 3× the original γ (and β for LayerNorm). Compute: one
    element-wise median per forward.

    Args:
        original: Either an :class:`nn.LayerNorm` or :class:`nn.RMSNorm`
            instance. The original's hyperparameters (``normalized_shape``,
            ``eps``, ``elementwise_affine``) are copied; its weight (and
            bias for LayerNorm) is cloned three times.
        trainable: If ``True``, replicas are trainable parameters. Defaults
            to ``False`` for inference safety.

    Raises:
        TypeError: If ``original`` is neither LayerNorm nor RMSNorm.
        ValueError: If the layer has no learnable affine parameters
            (``elementwise_affine=False``), there is nothing to TMR.
    """

    def __init__(self, original: nn.Module, *, trainable: bool = False) -> None:
        super().__init__()
        is_layernorm = isinstance(original, nn.LayerNorm)
        is_rmsnorm = _is_rmsnorm(original)
        if not (is_layernorm or is_rmsnorm):
            raise TypeError(
                f"TMRLayerNorm wraps nn.LayerNorm or nn.RMSNorm; got {type(original).__name__}",
            )
        weight = getattr(original, "weight", None)
        if weight is None:
            raise ValueError(
                "TMRLayerNorm requires a learnable affine, got a module with elementwise_affine=False",
            )
        normalized_shape = getattr(original, "normalized_shape", None)
        if normalized_shape is None:
            raise ValueError("Source module is missing normalized_shape")
        eps_attr = getattr(original, "eps", None)
        self._kind: str = "layernorm" if is_layernorm else "rmsnorm"
        self.normalized_shape: tuple[int, ...] = tuple(normalized_shape)
        # ``nn.RMSNorm`` admits ``eps=None`` (uses ``finfo(dtype).eps`` at
        # forward time); ``nn.LayerNorm`` always carries a float eps.
        self.eps: float | None = float(eps_attr) if eps_attr is not None else None

        weight_data = weight.detach().clone()
        self.weight_a = nn.Parameter(weight_data.clone(), requires_grad=trainable)
        self.weight_b = nn.Parameter(weight_data.clone(), requires_grad=trainable)
        self.weight_c = nn.Parameter(weight_data.clone(), requires_grad=trainable)

        bias = getattr(original, "bias", None) if is_layernorm else None
        self._has_bias: bool = bias is not None
        if bias is not None:
            bias_data = bias.detach().clone()
            self.bias_a = nn.Parameter(bias_data.clone(), requires_grad=trainable)
            self.bias_b = nn.Parameter(bias_data.clone(), requires_grad=trainable)
            self.bias_c = nn.Parameter(bias_data.clone(), requires_grad=trainable)
        else:
            self.register_parameter("bias_a", None)
            self.register_parameter("bias_b", None)
            self.register_parameter("bias_c", None)

    @staticmethod
    def _vote(*tensors: Tensor) -> Tensor:
        return torch.stack(tensors).median(dim=0).values

    @property
    def weight(self) -> Tensor:
        return self._vote(self.weight_a, self.weight_b, self.weight_c)

    @property
    def bias(self) -> Tensor | None:
        if not self._has_bias:
            return None
        assert self.bias_a is not None
        assert self.bias_b is not None
        assert self.bias_c is not None
        return self._vote(self.bias_a, self.bias_b, self.bias_c)

    def forward(self, x: Tensor) -> Tensor:
        if self._kind == "layernorm":
            # LayerNorm requires a concrete float eps; we default to 1e-5 if
            # the source somehow lacked one.
            eps = self.eps if self.eps is not None else 1e-5
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, eps)
        # RMSNorm, F.rms_norm accepts eps=None and picks finfo(dtype).eps
        return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)

    def extra_repr(self) -> str:
        return (
            f"kind={self._kind}, normalized_shape={self.normalized_shape}, "
            f"eps={self.eps}, bias={self._has_bias}, replicas=3"
        )


__all__ = ["TMRLayerNorm"]
