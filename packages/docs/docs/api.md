# API reference

The full public API at v0.1.0-dev. Everything here is exported from
`spacellm` or one of its submodules; private (`_`-prefixed) names are
implementation detail.

## Top level

| Name | Kind | Notes |
|---|---|---|
| `spacellm.__version__` | `str` | Single source of truth, duplicated in `pyproject.toml`, CI verifies match. |
| `spacellm.harden(model, *, strategies, environment=None)` | function | Applies strategies, returns a `HardenedModel`. |
| `spacellm.HardenedModel` | dataclass | Wraps `model`, `strategies`, optional `environment`. Callable; delegates forward. |
| `spacellm.RadiationStep` | dataclass | One environment snapshot. |
| `spacellm.FaultEvent` | dataclass | One bit-flip event. |
| `spacellm.FaultCause` | StrEnum | Source attribution. |
| `spacellm.RunRecord` | dataclass | One persisted run. |
| `spacellm.DeviceModel` | dataclass | Silicon characterisation for physics-based environments. |

## `spacellm.environments`

| Name | Kind | Notes |
|---|---|---|
| `Environment` | abstract | Base class for all environments. |
| `StatisticalEnvironment` | class | Level-1 Poisson injector. |

## `spacellm.protection`

| Name | Kind | Notes |
|---|---|---|
| `Strategy` | abstract | Base class for all hardening strategies. |
| `ProtectionReport` | dataclass | Telemetry snapshot. |
| `SelectiveTMR` | class | Triple-modular redundancy on top-k% of `nn.Linear`. |
| `AttentionChecksum` | class | Frobenius-norm fingerprint detection on attention modules. |
| `KVCacheParity` | class | XOR row-parity primitive for KV-cache tensors. |

## `spacellm.nn`

| Name | Kind | Notes |
|---|---|---|
| `TMRLinear` | class | Drop-in `nn.Linear` replacement with three replicas + median vote. |

## `spacellm.runtime`

| Name | Kind | Notes |
|---|---|---|
| `harden` | function | (Re-exported as `spacellm.harden`.) |
| `HardenedModel` | dataclass | (Re-exported as `spacellm.HardenedModel`.) |

## `spacellm.profiling`

| Name | Kind | Notes |
|---|---|---|
| `LayerSensitivity` | dataclass | Per-module sensitivity record. |
| `ProfileReport` | dataclass | Bundle of `LayerSensitivity` + serialization. |
| `static_profile(model, *, include_root=False)` | function | Cheap parameter-count + bit-area pass. |
| `sensitivity_profile(model, forward_fn, *, n_flips_per_layer=5, seed=0)` | function | Dynamic per-layer perturbation profile. |

## `spacellm.observability`

| Name | Kind | Notes |
|---|---|---|
| `RunDB` | class | SQLite-backed run database. Context-manager friendly. |
| `Run` | class | Context-managed run lifecycle. |

## CLI

The CLI lives in `spacellm.cli`; the entry point is the Typer app `spacellm.cli.app`.

* `spacellm version`
* `spacellm runs [--db PATH] [-n LIMIT]`
* `spacellm show RUN_ID [--db PATH]`
* `spacellm profile` (placeholder, v0.2)

## Stability

`__all__` declarations are the contract. Anything reachable via `import spacellm.*` but not in an `__all__` may move without notice during 0.x. The 1.0 release freezes the API.
