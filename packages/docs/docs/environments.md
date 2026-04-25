# Environments

A radiation environment is a fault source. SpaceLLM ships two environment tiers and a wrapper for multi-cell-upset clustering.

## StatisticalEnvironment

Bit upsets modeled as a homogeneous Poisson process with a configurable rate per bit per second. No physics; the fastest and most reproducible environment, appropriate for unit tests, regression suites, and protection-strategy A/B comparisons.

```python
import spacellm as sl

env = sl.environments.StatisticalEnvironment(
    rate_per_bit_per_s=1e-9,
    seed=0,
)

step = env.step(dt=1.0)  # RadiationStep(t=1.0, source="statistical", ...)
events = env.sample_faults({"w": tensor}, dt=1.0)
```

## PhysicsLiteEnvironment

A device-orbit pair that produces a calibrated SEU rate per bit per second. The Weibull cross-section is convolved with the orbital differential LET flux through the integrated rectangular-parallelepiped (IRPP) integral.

```python
import spacellm as sl
from spacellm.environments import LEO_ISS_NOMINAL, PhysicsLiteEnvironment
from spacellm.environments.devices import POLARFIRE_LSRAM

env = PhysicsLiteEnvironment(POLARFIRE_LSRAM, LEO_ISS_NOMINAL, seed=0)
print(f"SEU rate: {env.mission_seu_rate_per_bit_per_s:.2e} per bit per second")
```

Three orbit profiles ship today, each with primary-source citations: `LEO_ISS_NOMINAL` (Narici 2015), `GEO_QUIET` (NASA SP-2008-565), and `MARS_TRANSIT` (Zeitlin 2013). Six RT PolarFire device profiles ship today: `POLARFIRE_USRAM`, `POLARFIRE_LSRAM`, and four DFF data patterns. Every Weibull parameter is traced to a page and table of the source PDF in the device's `citations` field.

## MCUEnvironment

Wraps any base environment and amplifies each seed event into a multi-cell-upset cluster of `k` adjacent bit-flips. For 14-nm-class hardware, between 70% and 86% of strikes produce `k >= 2`.

```python
from spacellm.environments import MCUEnvironment
from spacellm.environments.mcu import default_mcu_distribution

env = MCUEnvironment(
    PhysicsLiteEnvironment(POLARFIRE_LSRAM, LEO_ISS_NOMINAL, seed=0),
    distribution=default_mcu_distribution(process_node_nm=14.0),
)
```

MCU clustering matters because TMR with three replicas survives a single flip per protected element but is defeated by two flips on the same bit position across two replicas, which is the failure mode MCU drives.

## Reproducibility

`StatisticalEnvironment` is deterministic for a given seed. `PhysicsLiteEnvironment` is deterministic given seed plus the orbit and device profiles. `MCUEnvironment` inherits the seed of its base.

Seed PyTorch's RNG separately if you also want deterministic model output. Environments seed *fault* RNG; users seed *model* RNG. Confusing the two is the most common cause of irreproducible fault-injection results.
