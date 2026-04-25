"""Core data types shared by every module.

These are the stable contracts every other module agrees on. Changes here are
breaking changes; we treat them as such even pre-1.0 to surface integration
issues early.

Why frozen dataclasses with ``slots``?
- *frozen*: events and environment steps must be safe to hash / log / send to
  another process without worrying about silent mutation.
- *slots*: makes the per-instance memory footprint small, which matters when a
  single inference can produce millions of ``FaultEvent``s.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4


class FaultCause(StrEnum):
    """Source attribution for a synthetic or measured bit-flip event.

    String-valued so the enum round-trips cleanly through JSON / SQLite.
    """

    SYNTHETIC = "synthetic"
    SEU_PROTON = "seu_proton"
    SEU_HEAVY_ION = "seu_heavy_ion"
    SEU_NEUTRON = "seu_neutron"
    SEU_ALPHA = "seu_alpha"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class FaultEvent:
    """One bit-flip in one tensor.

    Attributes:
        t: Seconds since the simulation / mission start.
        tensor_id: Stable name of the affected tensor, conventionally a
            ``"."``-separated dotted path such as
            ``"model.layers.0.attention.q_proj.weight"``.
        bit_index: Absolute bit index in the flat tensor, i.e.,
            ``flat_index * bits_per_element + bit_position``.
        bit_position: Bit position within the element scalar
            (``0 .. bits_per_element-1``).
        before: Unsigned-integer bit pattern of the affected element prior to
            the flip.
        after: Unsigned-integer bit pattern after the flip.
        cause: Attribution label, useful for benchmarking which particle
            populations dominate a workload's failure profile.
    """

    t: float
    tensor_id: str
    bit_index: int
    bit_position: int
    before: int
    after: int
    cause: FaultCause = FaultCause.SYNTHETIC


@dataclass(frozen=True, slots=True)
class RadiationStep:
    """Flux + dose snapshot at one moment in mission time.

    Attributes:
        t: Seconds since mission start.
        proton_flux: Particles · cm⁻² · s⁻¹ above 10 MeV (proxy for SEU-relevant
            proton population).
        heavy_ion_flux: Particles · cm⁻² · s⁻¹ with LET above 1 MeV·cm²·mg⁻¹.
        seu_rate_per_bit_per_s: Device-specific upset rate derived from the
            flux × cross-section convolution. Statistical environments may
            populate this directly; physics-based environments compute it.
        tid_rate_gy_per_s: Total ionizing dose rate (Gy(Si) · s⁻¹).
        inside_saa: ``True`` if the producing trajectory is inside the South
            Atlantic Anomaly at this step.
        source: Free-form label for the producing environment, e.g.
            ``"statistical"``, ``"physics_lite:LEO_ISS"``, ``"spenvis_import"``.
    """

    t: float
    proton_flux: float
    heavy_ion_flux: float
    seu_rate_per_bit_per_s: float
    tid_rate_gy_per_s: float
    inside_saa: bool = False
    source: str = "synthetic"


@dataclass(frozen=True, slots=True)
class DeviceModel:
    """Target silicon characterisation used by physics-based environments.

    The four-parameter Weibull cross-section is the de-facto standard fit for
    heavy-ion SEU response and is what physics-lite (v0.3) consumes. See
    ``packages/docs/docs/physics-primer.md`` §2 and
    the physics primer for source citations.

    Attributes:
        name: Human-readable identifier, e.g. ``"jetson_orin_nano"``.
        process_node_nm: Lithography node, e.g. 7.0 for TSMC N7.
        sensitive_volume_um3: SRAM cell sensitive-volume estimate.
        saturation_cross_section_cm2: Weibull asymptotic cross-section σ_sat.
        onset_let_mev_cm2_per_mg: Weibull onset LET threshold L₀.
        width: Weibull width parameter W.
        shape: Weibull shape parameter s.
        shielding_g_cm2: Areal shielding mass between the chip and the
            environment, in g·cm⁻².
        verified: ``True`` only when every Weibull / SEL / TID parameter
            below has been pulled directly from a primary-source beam-test
            paper (NEPP / NSREC / vendor PDF). ``False`` for plausibility
            estimates from process-node analogues; profiles in the
            ``spacellm.environments.devices`` catalogue use this flag to
            mark which numbers are mission-credible.
        sel_threshold_mev_cm2_per_mg: LET above which a heavy-ion strike
            can trigger single-event latch-up. ``None`` if the device is
            SEL-immune within the tested LET range.
        tid_failure_krad_si: Total-ionising-dose failure threshold in
            krad(Si). ``None`` if not characterised.
        mcu_cluster_distribution: Discrete probability of cluster size 1, 2,
            3, … for a single heavy-ion strike. Index 0 is single-bit
            upset (SBU). ``None`` defers to a default derived from
            ``process_node_nm`` by ``spacellm.environments.mcu``.
        citations: Tuple of bibliographic strings backing every numeric
            field above.
    """

    name: str
    process_node_nm: float
    sensitive_volume_um3: float
    saturation_cross_section_cm2: float
    onset_let_mev_cm2_per_mg: float
    width: float = 30.0
    shape: float = 1.5
    shielding_g_cm2: float = 0.0
    verified: bool = False
    sel_threshold_mev_cm2_per_mg: float | None = None
    tid_failure_krad_si: float | None = None
    mcu_cluster_distribution: tuple[float, ...] | None = None
    citations: tuple[str, ...] = ()


@dataclass(slots=True)
class RunRecord:
    """One full simulation, training, or evaluation run.

    Mutable so the run loop can update ``ended_at``, ``metrics``, and counters
    without allocating a new record. Persisted by ``spacellm.observability``.

    Attributes:
        run_id: 32-char hex UUID4, primary key in the run database.
        name: Human-friendly label.
        model_name: Model identifier (HF Hub repo, local path, or custom name).
        model_sha: Git commit / HF revision hash for reproducibility.
        environment_spec: Free-form dict capturing the environment configuration.
        protection_spec: Free-form dict capturing the protection configuration.
        started_at: UTC start timestamp.
        ended_at: UTC end timestamp; ``None`` while the run is in progress.
        metrics: Final numeric metrics keyed by name.
        artifact_uris: Pointers to large artifacts stored outside the DB
            (Parquet, checkpoints, plots).
        n_faults_injected: Bit-flips applied to model state during this run.
        n_faults_recovered: Bit-flips a protection strategy detected and
            corrected.
        notes: Free-form text.
    """

    run_id: str = field(default_factory=lambda: uuid4().hex)
    name: str = ""
    model_name: str = ""
    model_sha: str = ""
    environment_spec: dict[str, Any] = field(default_factory=dict)
    protection_spec: dict[str, Any] = field(default_factory=dict)
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    ended_at: datetime | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    artifact_uris: list[str] = field(default_factory=list)
    n_faults_injected: int = 0
    n_faults_recovered: int = 0
    notes: str = ""


__all__ = [
    "DeviceModel",
    "FaultCause",
    "FaultEvent",
    "RadiationStep",
    "RunRecord",
]
