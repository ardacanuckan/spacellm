"""Module-tree path manipulation utilities (private)."""

from __future__ import annotations

from typing import cast

from torch import nn


def _step(parent: nn.Module, key: str) -> nn.Module:
    """Descend one level through a module tree.

    Integer keys index into :class:`torch.nn.ModuleList` /
    :class:`torch.nn.Sequential`; string keys go through ``getattr``.
    """
    if key.isdigit():
        container = cast("nn.ModuleList", parent)
        return container[int(key)]
    return cast("nn.Module", getattr(parent, key))


def set_submodule(root: nn.Module, dotted_path: str, replacement: nn.Module) -> None:
    """Replace a submodule identified by a dotted path.

    Handles both attribute access (``encoder.layer_norm``) and integer
    indexing (``encoder.layers.0``), the latter is needed because
    :class:`torch.nn.ModuleList` exposes its children via integer keys.

    Args:
        root: The root module to navigate from.
        dotted_path: Dot-separated path, as emitted by
            :meth:`torch.nn.Module.named_modules`.
        replacement: The module to insert at ``dotted_path``.

    Raises:
        ValueError: If ``dotted_path`` is empty.
    """
    if not dotted_path:
        raise ValueError("Cannot replace the root module via set_submodule()")
    parts = dotted_path.split(".")
    parent: nn.Module = root
    for part in parts[:-1]:
        parent = _step(parent, part)
    last = parts[-1]
    if last.isdigit():
        container = cast("nn.ModuleList", parent)
        container[int(last)] = replacement
    else:
        setattr(parent, last, replacement)


def walk_named_submodules(
    root: nn.Module,
    *,
    skip_root: bool = True,
) -> list[tuple[str, nn.Module]]:
    """List ``(path, module)`` pairs.

    Snapshot-style: returns a list rather than a generator so callers can
    safely mutate the tree during iteration.
    """
    pairs = list(root.named_modules())
    return pairs[1:] if skip_root and pairs and pairs[0][0] == "" else pairs


__all__ = ["set_submodule", "walk_named_submodules"]
