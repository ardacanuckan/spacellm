# SpaceLLM

Open-source Python framework for **radiation-tolerant transformer training and inference on orbital compute**.

Apache-2.0. Built for operators of orbital data centers (Starcloud-class LEO solar compute, satellite-bus inference) and satellite ML engineers running transformers on hardware that takes single-event upsets while the job is in flight.

## What it is

SpaceLLM wraps any [Hugging Face Transformers](https://github.com/huggingface/transformers) model with three layers no other tool combines under one umbrella:

1. **A calibrated radiation environment.** Weibull cross-section, IRPP integration, MCU clustering. Per-bit SEU rates derive from publicly fetched primary-source beam-test data.
2. **Five composable protection strategies.** `SelectiveTMR`, `AttentionChecksum`, `KVCacheParity`, `LayerNormTMR`, `EmbeddingChecksum`. Compose freely via a single `harden()` call.
3. **A bench harness and validation routine.** `silent_error_rate` is the unique radhard KPI; `validate_against_measurements` scores DeviceModel predictions against published beam-test data within an explicit ECSS-Q-ST-60-15C envelope.

## Why it exists

A single bit-flip in an LLM weight tensor can be catastrophic. Recent published evidence:

| Attack | Model | Damage | Source |
|---|---|---|---|
| SBFA  | Qwen2.5-7B | MMLU 71% to 0% with **one** bit-flip | [arXiv:2509.21843](https://arxiv.org/abs/2509.21843) |
| AttentionBreaker | Any LLM | catastrophic collapse with **3 of 10¹⁰** flips | [arXiv:2411.13757](https://arxiv.org/abs/2411.13757) |
| KV-cache drift | Long-context QA | SQuAD F1 77.4 to 7.2 | [arXiv:2510.17098](https://arxiv.org/abs/2510.17098) |

In space, those flips arrive whether you defend against them or not. Protons, heavy ions, and neutrons drive single-event upset rates that depend on orbit, shielding, and silicon. SpaceLLM is the runtime layer that catches, masks, and budgets those upsets while PyTorch keeps running underneath.

## Where to start

<div class="grid cards" markdown>

-   :material-rocket-launch: **[Quickstart](quickstart.md)**

    Install, harden a model, observe protection telemetry. Under 60 seconds.

-   :material-source-branch: **[Architecture](architecture.md)**

    Module layout, stack choices, what is and is not shipped today.

-   :material-shield-half-full: **[Protection strategies](protection.md)**

    Five strategies in depth, with composition rules.

-   :material-chart-line: **[Orbital LLM training runbook](orbital-training.md)**

    Five-phase pipeline for the persona this framework targets.

</div>

## Status

`0.3.0.dev0`. Pre-1.0; API may still change. 217 / 217 tests passing, ruff clean, mypy `--strict` clean across the source tree.
