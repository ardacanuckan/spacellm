# Quickstart

## Install

```bash
pip install spacellm           # core
pip install spacellm[hf]       # + Hugging Face stack
pip install spacellm[all]      # + profiling + visualization extras
```

The package targets Python 3.11 and newer. PyTorch 2.4+ is a runtime dependency.

## Harden a model

```python
import torch
import spacellm as sl
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

hardened = sl.harden(
    model,
    strategies=[
        sl.protection.SelectiveTMR(top_k_percent=5),
        sl.protection.AttentionChecksum(),
        sl.protection.KVCacheParity(),
    ],
)

# `hardened` is a transparent handle. Forward calls delegate to the model.
out = hardened.generate(input_ids, max_new_tokens=128)

# Telemetry: a ProtectionReport per strategy.
for report in hardened.collect_reports():
    print(report)
```

Three strategies in v0.1, each independent and composable:

* **`SelectiveTMR`**, wraps the largest top-k% of `nn.Linear` modules with three weight replicas, voted element-wise on every forward.
* **`AttentionChecksum`**, registers a forward pre-hook on every attention module that compares a Frobenius-norm fingerprint of its parameters against a trusted reference; mismatches surface as detected corruptions.
* **`KVCacheParity`**, XOR row-parity over guarded KV tensors. Detect-only in v0.1.

## Inject a fault environment

```python
env = sl.environments.StatisticalEnvironment(rate_per_bit_per_s=1e-6, seed=0)

# Time advances explicitly.
step = env.step(dt=60.0)
print(step)  # RadiationStep(t=60.0, ...)

# Sample bit-flip events for any tensor population.
weights = {name: p.data for name, p in model.named_parameters()}
events = env.sample_faults(weights, dt=60.0)
print(f"{len(events)} bit-flips in 60 s")
```

Statistical (Level-1) environments are deterministic for a given seed, irreproducible fault injection is a useless benchmark. Physics-based (Level-2) environments using AE-9 / AP-9 + Weibull cross-sections land in v0.3.

## Track runs

```python
from spacellm.observability import Run, RunDB

with RunDB("runs.db") as db, Run(name="ablation-1", db=db) as run:
    run.log_metric("perplexity", 7.42)
    for event in events:
        run.log_fault(event)
```

Then inspect from the CLI:

```bash
spacellm runs                 # list recent runs
spacellm show <run-id-prefix> # details for one run
```

## Run the demo

The repository ships a full end-to-end example that runs in under a second on a CPU laptop:

```bash
git clone https://github.com/spacellm/spacellm
cd spacellm
make install
uv run python examples/01_quickstart.py
```

Expected output highlights:

* AttentionChecksum detects 4 corruptions after 1710 injected bit-flips.
* Post-injection forward produces NaN, confirming the fault is real and the detection has signal.
