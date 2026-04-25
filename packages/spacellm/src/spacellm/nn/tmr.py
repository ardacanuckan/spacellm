"""Triple-modular-redundancy linear layer."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812, torch convention


class TMRLinear(nn.Module):
    """Drop-in replacement for :class:`torch.nn.Linear` with 3× weight redundancy.

    Three weight (and bias) replicas are stored independently. On every
    forward, an element-wise median is computed across the three; any
    single bit-flip in one replica is masked by the other two.

    Cost
    ----
    * Memory: 3× the original weight + 3× the original bias.
    * Compute: one median over a stack of three plus the linear op.
    * Latency on CPU is roughly 1.2× the unprotected layer for typical LLM
      sizes; on GPU the median is vectorised cheaply.

    Trainability
    ------------
    By default the replicas are *frozen* (``requires_grad=False``). The
    intent of v0.1 is inference protection. Training-aware TMR, where
    gradients flow into all three replicas and re-vote on the optimizer
    step, lands in v0.7 with the rest of ``spacellm.training``.

    Args:
        original: The :class:`torch.nn.Linear` to wrap. Its weight and
            bias are cloned, not moved.
        trainable: If ``True``, the replicas are trainable parameters.
            Defaults to ``False`` for inference safety.
    """

    def __init__(self, original: nn.Linear, *, trainable: bool = False) -> None:
        super().__init__()
        self.in_features = original.in_features
        self.out_features = original.out_features
        self._has_bias = original.bias is not None
        weight = original.weight.detach().clone()
        self.weight_a = nn.Parameter(weight.clone(), requires_grad=trainable)
        self.weight_b = nn.Parameter(weight.clone(), requires_grad=trainable)
        self.weight_c = nn.Parameter(weight.clone(), requires_grad=trainable)
        if self._has_bias:
            assert original.bias is not None
            bias = original.bias.detach().clone()
            self.bias_a = nn.Parameter(bias.clone(), requires_grad=trainable)
            self.bias_b = nn.Parameter(bias.clone(), requires_grad=trainable)
            self.bias_c = nn.Parameter(bias.clone(), requires_grad=trainable)
        else:
            self.register_parameter("bias_a", None)
            self.register_parameter("bias_b", None)
            self.register_parameter("bias_c", None)

    @staticmethod
    def _vote(*tensors: Tensor) -> Tensor:
        return torch.stack(tensors).median(dim=0).values

    def voted_weight(self) -> Tensor:
        return self._vote(self.weight_a, self.weight_b, self.weight_c)

    def voted_bias(self) -> Tensor | None:
        if not self._has_bias:
            return None
        assert self.bias_a is not None
        assert self.bias_b is not None
        assert self.bias_c is not None
        return self._vote(self.bias_a, self.bias_b, self.bias_c)

    # Public ``weight`` / ``bias`` aliases so TMRLinear is a true drop-in for
    # nn.Linear inside containers that read these attributes directly (e.g.
    # nn.MultiheadAttention reads ``out_proj.weight`` without going through
    # ``forward``). The properties always return the voted view.
    @property
    def weight(self) -> Tensor:
        return self.voted_weight()

    @property
    def bias(self) -> Tensor | None:
        return self.voted_bias()

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.voted_weight(), self.voted_bias())

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self._has_bias}, replicas=3"
        )


__all__ = ["TMRLinear"]
