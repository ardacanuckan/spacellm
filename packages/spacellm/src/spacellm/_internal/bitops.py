"""Bit-level operations on PyTorch tensors.

Strategy
--------
PyTorch's ``Tensor.view(int_dtype)`` reinterprets the underlying memory as the
matching-size signed integer type without a copy. We then XOR a single bit and
write back. This is dtype-agnostic, works for FP32/FP16/BF16/FP64 *and* the
integer dtypes, because every supported scalar type has a same-byte-width
signed-integer view in torch (``int8``/``int16``/``int32``/``int64``).

The only fiddly part is two's-complement: we expose the *unsigned* bit pattern
to the rest of the codebase (it's what people mean when they say "bit pattern")
but PyTorch's signed-int slot wants signed values.

In-place semantics
------------------
``flip_bit`` modifies the input tensor's storage. The caller is responsible for
making a snapshot if they want to compare before/after as PyTorch tensors.
``before`` and ``after`` returned from ``flip_bit`` are unsigned integer values
of the affected scalar element only.

The operation is GPU-safe (``view`` works on CUDA tensors) but requires a
contiguous tensor, non-contiguous tensors raise ``RuntimeError`` from PyTorch
itself, which we surface unchanged.
"""

from __future__ import annotations

import torch

# Map element size in bytes → matching-width signed int dtype.
_INT_FOR_SIZE: dict[int, torch.dtype] = {
    1: torch.int8,
    2: torch.int16,
    4: torch.int32,
    8: torch.int64,
}


def _to_signed(unsigned: int, bits: int) -> int:
    """Reinterpret an unsigned ``bits``-wide value as two's-complement signed."""
    threshold = 1 << (bits - 1)
    return unsigned - (1 << bits) if unsigned >= threshold else unsigned


def _to_unsigned(signed: int, bits: int) -> int:
    """Reinterpret a signed two's-complement ``bits``-wide value as unsigned."""
    return signed + (1 << bits) if signed < 0 else signed


def flip_bit(
    tensor: torch.Tensor,
    flat_index: int,
    bit_position: int,
) -> tuple[int, int]:
    """Flip one bit in ``tensor`` in place.

    Args:
        tensor: Contiguous tensor of any element size in {1, 2, 4, 8} bytes.
        flat_index: Element index when the tensor is flattened (row-major).
        bit_position: Bit position inside the element, 0 ≤ p < element_size·8.

    Returns:
        ``(before, after)``, unsigned integer bit patterns of the modified
        element before and after the flip.

    Raises:
        IndexError: If ``flat_index`` or ``bit_position`` is out of range.
        TypeError: If the element size is not supported (e.g. complex tensors).
    """
    if tensor.is_complex():
        raise TypeError(
            f"Complex tensors are not supported (got {tensor.dtype}); a single-bit "
            "flip would be ambiguous between real and imaginary components.",
        )
    bytes_per_element = tensor.element_size()
    bits_per_element = bytes_per_element * 8
    int_dtype = _INT_FOR_SIZE.get(bytes_per_element)
    if int_dtype is None:
        raise TypeError(
            f"Unsupported element size {bytes_per_element} for dtype {tensor.dtype}",
        )
    if not 0 <= flat_index < tensor.numel():
        raise IndexError(
            f"flat_index {flat_index} out of range [0, {tensor.numel()})",
        )
    if not 0 <= bit_position < bits_per_element:
        raise IndexError(
            f"bit_position {bit_position} out of range [0, {bits_per_element})",
        )

    int_view = tensor.view(int_dtype).reshape(-1)
    before_signed = int(int_view[flat_index].item())
    before_unsigned = _to_unsigned(before_signed, bits_per_element)
    after_unsigned = before_unsigned ^ (1 << bit_position)
    after_signed = _to_signed(after_unsigned, bits_per_element)
    int_view[flat_index] = after_signed
    return before_unsigned, after_unsigned


def flip_bit_uniform(
    tensor: torch.Tensor,
    bit_index: int,
) -> tuple[int, int, int]:
    """Flip the bit at absolute ``bit_index`` over the entire flattened tensor.

    Equivalent to ``flip_bit(tensor, bit_index // bpe, bit_index % bpe)`` where
    ``bpe`` is the bit-count per element.

    Returns:
        ``(before, after, bit_position)``, bit patterns and the local position
        within the affected element, useful for telemetry.
    """
    bits_per_element = tensor.element_size() * 8
    flat_index, bit_position = divmod(bit_index, bits_per_element)
    before, after = flip_bit(tensor, flat_index, bit_position)
    return before, after, bit_position


__all__ = ["flip_bit", "flip_bit_uniform"]
