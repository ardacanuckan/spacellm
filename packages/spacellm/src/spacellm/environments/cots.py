"""COTS accelerator device profiles, process-node analogue *estimates*.

Why this is a separate module from :mod:`spacellm.environments.devices`
----------------------------------------------------------------------
``devices.py`` holds a strict, primary-source-only catalogue: every number
traces to a fetched beam-test PDF and carries ``verified=True``. The
orbital-data-center reality, however, is that operators run **COTS GPUs and
NPUs** (Jetson Orin, Coral Edge TPU, datacentre GPUs), and public,
fittable heavy-ion Weibull tables for those exact parts are scarce or
NDA-bound.

Rather than fabricate "verified" numbers, the profiles here are explicit
**process-node analogue estimates**: order-of-magnitude SRAM SEU
parameters typical of the lithography node, every one flagged
``verified=False``. They exist so a mission planner can get a first
SEU-rate bracket on COTS silicon in one line, and the docstrings shout the
correct next step:

    Replace these with your own beam-test data via
    :func:`spacellm.environments.physics.device_from_measurements`.

That calibration path, not these estimates, is the load-bearing
contribution for COTS hardware.
"""

from __future__ import annotations

from spacellm.types import DeviceModel

_COTS_DISCLAIMER = (
    "PROCESS-NODE ANALOGUE ESTIMATE, NOT a beam-test fit. Weibull params are "
    "order-of-magnitude SRAM values typical of the stated lithography node "
    "(per-bit σ_sat ~1e-8..1e-9 cm², low onset LET characteristic of "
    "unhardened COTS); ranges consistent with the open SEU literature "
    "(e.g. Quinn et al. COTS SRAM/FPGA surveys; the 65 nm triple-well MCU "
    "study cited in spacellm.environments.mcu). verified=False. Calibrate "
    "with spacellm.environments.physics.device_from_measurements on your own "
    "heavy-ion data before any mission analysis."
)


def _cots(
    *,
    name: str,
    process_node_nm: float,
    sigma_sat: float,
    onset: float,
    width: float,
    shape: float,
    sel_threshold: float | None,
) -> DeviceModel:
    return DeviceModel(
        name=name,
        process_node_nm=process_node_nm,
        sensitive_volume_um3=0.02,  # small modern-node SRAM cell
        saturation_cross_section_cm2=sigma_sat,
        onset_let_mev_cm2_per_mg=onset,
        width=width,
        shape=shape,
        verified=False,
        sel_threshold_mev_cm2_per_mg=sel_threshold,
        citations=(_COTS_DISCLAIMER,),
    )


JETSON_ORIN_ESTIMATE = _cots(
    name="nvidia_jetson_orin_estimate",
    process_node_nm=8.0,  # Samsung 8 nm
    sigma_sat=1.0e-8,
    onset=0.8,
    width=18.0,
    shape=1.4,
    sel_threshold=20.0,
)
"""NVIDIA Jetson Orin (Samsung 8 nm), analogue estimate. ``verified=False``."""


CORAL_EDGE_TPU_ESTIMATE = _cots(
    name="google_coral_edge_tpu_estimate",
    process_node_nm=14.0,
    sigma_sat=7.0e-9,
    onset=1.0,
    width=16.0,
    shape=1.4,
    sel_threshold=30.0,
)
"""Google Coral Edge TPU (~14 nm class), analogue estimate. ``verified=False``."""


DATACENTRE_GPU_5NM_ESTIMATE = _cots(
    name="datacentre_gpu_5nm_estimate",
    process_node_nm=5.0,  # TSMC 4N/5 nm class (H100-era)
    sigma_sat=4.0e-9,
    onset=0.5,
    width=14.0,
    shape=1.3,
    sel_threshold=15.0,
)
"""Datacentre GPU on a TSMC 4N/5 nm-class node (H100-era), analogue estimate.
``verified=False``, the SRAM/register-file analogue an orbital-data-centre
operator would bracket before a real beam campaign."""


COTS_DEVICES: dict[str, DeviceModel] = {
    JETSON_ORIN_ESTIMATE.name: JETSON_ORIN_ESTIMATE,
    CORAL_EDGE_TPU_ESTIMATE.name: CORAL_EDGE_TPU_ESTIMATE,
    DATACENTRE_GPU_5NM_ESTIMATE.name: DATACENTRE_GPU_5NM_ESTIMATE,
}
"""Stable name → estimate lookup. Every entry is ``verified=False``."""


__all__ = [
    "CORAL_EDGE_TPU_ESTIMATE",
    "COTS_DEVICES",
    "DATACENTRE_GPU_5NM_ESTIMATE",
    "JETSON_ORIN_ESTIMATE",
]
