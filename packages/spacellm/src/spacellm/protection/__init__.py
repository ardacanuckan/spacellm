"""Hardening strategies, the radiation-tolerance half of the framework.

Every strategy implements :class:`~spacellm.protection.base.Strategy`. They
are composed by the runtime via :func:`spacellm.harden`:

.. code-block:: python

    import spacellm as sl

    hardened = sl.harden(
        model,
        strategies=[
            sl.protection.SelectiveTMR(top_k_percent=5),
            sl.protection.AttentionChecksum(),
            sl.protection.KVCacheParity(),
        ],
    )

In v0.1 we ship three orthogonal strategies that target the three highest-
priority sensitivity hotspots identified in
the protection module docs, the linear projections on the long
tail of weights (TMR), the attention QKV/output weights (checksum), and the
KV cache during long-context decode (parity).
"""

from __future__ import annotations

from spacellm.protection.attention import AttentionChecksum
from spacellm.protection.base import ProtectionReport, Strategy
from spacellm.protection.embedding import EmbeddingChecksum
from spacellm.protection.kv_cache import KVCacheParity
from spacellm.protection.layer_norm import LayerNormTMR
from spacellm.protection.tmr import SelectiveTMR

__all__ = [
    "AttentionChecksum",
    "EmbeddingChecksum",
    "KVCacheParity",
    "LayerNormTMR",
    "ProtectionReport",
    "SelectiveTMR",
    "Strategy",
]
