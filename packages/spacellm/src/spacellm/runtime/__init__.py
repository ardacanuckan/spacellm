"""High-level runtime entry points.

The public face of the framework. Most users only ever touch
:func:`harden` and the returned :class:`HardenedModel`; everything else in
``spacellm`` is composable but not strictly required.
"""

from __future__ import annotations

from spacellm.runtime.harden import HardenedModel, harden

__all__ = ["HardenedModel", "harden"]
