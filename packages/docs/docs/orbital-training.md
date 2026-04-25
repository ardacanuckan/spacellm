# Orbital LLM training with SpaceLLM

This page is the runbook for the persona SpaceLLM is built for: an
operator of an orbital data center (Starcloud-class LEO solar compute,
satellite-bus inference) running transformer training or inference on
hardware that takes single-event upsets while the job is in flight.

If your team operates compute on the ground, you will not find the
unique value of SpaceLLM here, you would be better served by a
hardware-failure resilience layer like `torchrun` checkpointing or
DeepSpeed elasticity. SpaceLLM solves the *new* problem that arises
when the substrate itself is randomly mutating mid-step.

---

## What changes when you train in orbit

| Property | Ground GPU cluster | Orbital data center |
|---|---|---|
| SEU rate per bit | ~10⁻²⁴ /s (terrestrial neutrons) | **10⁻¹² – 10⁻¹⁰ /s** depending on orbit |
| Multi-cell upsets | not relevant | **70-86 % of strikes flip ≥ 2 adjacent cells** at 14 nm |
| Per-bit mean time to failure on a 7B FP16 model | ~10²² s | **~10⁴ s = a few hours** in LEO ISS |
| Visibility into corruption | crash → SIGBUS → core dump | **silent**: loss looks monotone, gradient is poisoned |

Source rates: see the [physics primer](physics-primer.md) and the
LEO / GEO / Mars-transit rate table in the project
[`README.md`](https://github.com/ardacanuckan/spacellm#mission-realistic-seu-rate-per-orbit-rt-polarfire-lsram).

The framework's job is to convert that bottom row from "silent" to
"counted, masked when possible, and persisted in the run database for
post-flight reconciliation."

---

## End-to-end pipeline

Five phases, each backed by a real module:

### 1. Pre-flight rate prediction

Before you commit GPU-hours to an in-orbit training job, get a
calibrated upset budget for the orbit and silicon you are flying.

```python
from spacellm.environments import (
    PhysicsLiteEnvironment, MCUEnvironment, LEO_ISS_NOMINAL,
)
from spacellm.environments.devices import POLARFIRE_LSRAM
from spacellm.environments.mcu import default_mcu_distribution

env = MCUEnvironment(
    PhysicsLiteEnvironment(POLARFIRE_LSRAM, LEO_ISS_NOMINAL, seed=0),
    distribution=default_mcu_distribution(process_node_nm=14.0),
)
print("SEU/bit/s:", env.base.seu_rate_per_bit_per_s())
```

The number you get back is *primary-source-traceable*, every Weibull
parameter cites the page of the PDF it came from. See
[`packages/docs/docs/beam-test-data.md`](beam-test-data.md).

### 2. Hardening the model

Wrap the same `nn.Module` you would normally pass to your trainer.
Strategies compose; pick the budget you want to spend.

```python
from spacellm import harden, protection

hardened = harden(model, strategies=[
    protection.SelectiveTMR(top_k_percent=5),    # ~3× memory for top-5% layers
    protection.AttentionChecksum(),              # ~0.001 % overhead, detection-only
    protection.LayerNormTMR(),                   # ~3× memory for all norms
    protection.EmbeddingChecksum(),
    protection.KVCacheParity(),                  # decoder-only: protects KV cache
], environment=env)
```

`hardened.model` is still an `nn.Module`. Pass it to
`torch.optim.AdamW`, `accelerate`, or your trainer of choice.

### 3. Training-time protection

In a normal training step, the protection strategies do their work
inline (TMR median vote, parity checks). When the environment fires
a fault, the bench harness records the event in the run DB; if
protection masks it, downstream code never sees it. If it does not,
the silent-error rate ticks up and you can decide whether to roll the
step back.

### 4. Persisted observability

Every run goes through `spacellm.observability.Run`, a SQLite
context manager that records start/end timestamps, metrics, and
detection events. The schema is the same on the ground rehearsal as
on the spacecraft, so post-flight analysts can compare ground-side
expectations to orbital telemetry on a join.

```python
from spacellm.observability import Run

with Run(name="leo_finetune_step_42") as run:
    loss = train_step(hardened.model, batch)
    run.log_metric("loss", loss.item())
    run.log_metric("silent_error_rate", hardened.silent_error_rate())
```

### 5. Post-flight reconciliation

After the spacecraft downlinks its run-DB rows, compare them to the
predictions you produced in phase 1. If on-orbit upsets per
mission-day exceed prediction by more than the validation envelope,
the gap becomes an explicit research item, not an unmodelled mystery.

The validation harness already supports this on the ground side:
`spacellm.validation.validate_against_measurements(device, data)`
returns RMSE in log10 σ space, R², and a pass/fail flag against an
ECSS-Q-ST-60-15C-style envelope.

---

## What this does not solve

The framework is narrow. Things you still have to do
yourself:

- **Power and thermal budgeting.** SpaceLLM tells you when bits will
  flip; it does not tell you when your radiator pumps will fail.
- **Comms downlink scheduling.** The run-DB is local; you decide
  when to ship it to the ground.
- **Beam testing your specific GPU.** Until the silicon you are
  flying has a Weibull fit, the rate prediction is upper-bounded by
  whatever DeviceModel you pick as a proxy. SpaceLLM's catalogue
  ships RT PolarFire, for COTS GPU silicon you need to either fit
  a Weibull from your own beam-test data or pick a published proxy
  with explicit caveats logged in the DeviceModel citations.

The honest guarantee SpaceLLM gives you is: *given a calibrated
DeviceModel and a calibrated OrbitProfile, the bench will report
silent-error rate within a documented envelope.* Everything upstream
of that, the calibration itself, is on the integrator.
