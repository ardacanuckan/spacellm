"""Multi-cell upset (MCU / MBU) environment wrapper.

Modern submicron silicon (≤ 65 nm, especially ≤ 28 nm) shows the bulk of
heavy-ion strikes producing not single bit-flips but spatially-correlated
clusters of 2–8 adjacent flipped cells. This module wraps any base
:class:`~spacellm.environments.base.Environment` and turns each base event
into a cluster sampled from a configurable size distribution.

Why this matters
----------------
* In INT8 weight tensors a 2-cell MCU within the same byte exceeds
  Hamming-1 ECC.
* For TMR-protected layers, MCU events that hit two of the three replicas
  in the same bit position defeat the median vote. With independent bits
  this is vanishingly rare; with spatially-correlated MCU it is the
  dominant failure mode at modern process nodes.
* Independent-bit assumptions therefore *under*-estimate SEE rates by
  2–10× on Jetson Orin / Coral-class hardware (a 65 nm triple-well
  study reports 71–86 % of upset events as multi-cell, with stronger
  effect at 14 nm and below).

Caveats
-------
* "Adjacent" here means *flat-index adjacency in the tensor*, which is a
  practical approximation of physical-cell adjacency. Real silicon
  layouts route adjacent cells to non-contiguous logical addresses; a
  full physical-layout model is out of scope for v0.4.
* Single-event latch-up (SEL) and SEFI are *not* injected by this
  wrapper, for those, gate the simulation on
  ``device.sel_threshold_mev_cm2_per_mg`` and ``tid_failure_krad_si``
  separately.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch

from spacellm._internal.bitops import flip_bit_uniform
from spacellm._internal.sampling import RandomState
from spacellm.environments.base import Environment
from spacellm.types import FaultEvent, RadiationStep


@dataclass(frozen=True, slots=True)
class MCUDistribution:
    """Discrete cluster-size distribution for multi-cell upsets.

    ``probabilities[i]`` is the probability that a single heavy-ion strike
    produces a cluster of size ``i + 1`` (i.e. ``probabilities[0]`` is the
    SBU probability, ``probabilities[1]`` is the 2-cell MCU probability,
    and so on).

    Raises:
        ValueError: If any probability is negative or the sum is not 1.0
            within ``rtol=1e-3``.
    """

    probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.probabilities:
            raise ValueError("probabilities must be non-empty")
        if any(p < 0 for p in self.probabilities):
            raise ValueError("probabilities must be non-negative")
        total = sum(self.probabilities)
        if not np.isclose(total, 1.0, rtol=1e-3):
            raise ValueError(f"probabilities must sum to 1 (got {total})")

    @property
    def max_size(self) -> int:
        return len(self.probabilities)

    @property
    def expected_size(self) -> float:
        """E[cluster size] under this distribution."""
        return float(sum((i + 1) * p for i, p in enumerate(self.probabilities)))

    def sample_size(self, rng: RandomState) -> int:
        """Draw a cluster size 1, 2, …, ``max_size`` from this distribution."""
        u = rng.random()
        cum = 0.0
        for i, p in enumerate(self.probabilities, start=1):
            cum += p
            if u < cum:
                return i
        return self.max_size

    @classmethod
    def for_process_node(cls, process_node_nm: float) -> MCUDistribution:
        """Default cluster-size distribution chosen by lithography node.

        Coarse process-node bands derived from the 65 nm triple-well
        observation . Submicron tail values are
        chosen to monotonically increase MCU dominance with shrinking
        node.
        """
        if process_node_nm >= 90.0:
            return cls((1.0,))
        if process_node_nm >= 28.0:
            return cls((0.55, 0.30, 0.10, 0.05))
        if process_node_nm >= 14.0:
            return cls((0.30, 0.35, 0.25, 0.10))
        return cls((0.20, 0.30, 0.30, 0.15, 0.05))


class MCUEnvironment(Environment):
    """Wrap a base environment and turn each strike into a correlated cluster.

    For every :class:`FaultEvent` produced by the wrapped environment,
    this wrapper samples a cluster size ``k`` from ``distribution`` and
    applies ``k − 1`` *additional* bit-flips at adjacent flat-bit indices
    in the same tensor. The original flip from the base call is preserved
    untouched; the added flips appear in the returned event list with
    the same ``cause`` and ``t`` as the seed event.

    Args:
        base: Underlying environment (typically
            :class:`~spacellm.environments.statistical.StatisticalEnvironment`
            or
            :class:`~spacellm.environments.physics_lite.PhysicsLiteEnvironment`).
        distribution: Cluster-size distribution to sample from.
        seed: RNG seed for cluster-size sampling. Independent of the
            base environment's RNG.
    """

    def __init__(
        self,
        base: Environment,
        distribution: MCUDistribution,
        *,
        seed: int | None = None,
    ) -> None:
        self.base = base
        self.distribution = distribution
        self._rng = RandomState(seed=seed)

    @property
    def t(self) -> float:
        return self.base.t

    def step(self, dt: float) -> RadiationStep:
        return self.base.step(dt)

    def sample_faults(
        self,
        tensors: Mapping[str, torch.Tensor],
        dt: float,
    ) -> list[FaultEvent]:
        base_events = self.base.sample_faults(tensors, dt)
        all_events: list[FaultEvent] = list(base_events)
        for event in base_events:
            cluster_size = self.distribution.sample_size(self._rng)
            if cluster_size <= 1:
                continue
            tensor = tensors.get(event.tensor_id)
            if tensor is None or tensor.numel() == 0:
                continue
            n_bits = tensor.numel() * tensor.element_size() * 8
            for offset in range(1, cluster_size):
                # Prefer +offset; if the target overflows the tensor, mirror to -offset.
                target = event.bit_index + offset
                if target >= n_bits:
                    target = event.bit_index - offset
                if target < 0 or target >= n_bits or target == event.bit_index:
                    continue
                before, after, bit_position = flip_bit_uniform(tensor, target)
                all_events.append(
                    FaultEvent(
                        t=event.t,
                        tensor_id=event.tensor_id,
                        bit_index=target,
                        bit_position=bit_position,
                        before=before,
                        after=after,
                        cause=event.cause,
                    ),
                )
        return all_events

    def reset(self) -> None:
        self.base.reset()


__all__ = ["MCUDistribution", "MCUEnvironment"]
