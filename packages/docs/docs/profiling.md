# Profiling

Two profiling modes ship in v0.1:

## Static profile

A cheap pass that records every leaf module's parameter count and bit-area. Useful as a sanity check and as the default ranking source for `SelectiveTMR`.

```python
from spacellm.profiling import static_profile

report = static_profile(model)
for layer in report.top_k(5, by="n_parameters"):
    print(layer.layer_name, layer.module_class, layer.n_parameters)

report.save_json("static_profile.json")
```

Direct (non-recursive) parameters only, so the report aligns one-to-one with what a strategy might wrap.

## Sensitivity profile

A dynamic mode that runs your forward function under bit-flip injection. For every module that owns at least one parameter, the routine snapshots the originals, injects `n_flips_per_layer` random bit-flips, runs the forward, and records `(mean, max)` MSE between clean and post-flip outputs. The model is restored exactly.

```python
import torch
from spacellm.profiling import sensitivity_profile

inputs = torch.randint(0, 200, (1, 16))

def forward(model):
    return model(inputs)

report = sensitivity_profile(model, forward, n_flips_per_layer=5, seed=0)

for layer in report.top_k(10, by="max_perturbation"):
    print(layer.layer_name, layer.max_perturbation)
```

* **Cost.** Roughly `n_layers × n_flips_per_layer × forward_cost`. For a 50-layer transformer with 5 flips and a 10 ms forward this is ≈ 2.5 s on CPU.
* **Restoration guarantee.** Tested, see `tests/test_profiling.py::test_sensitivity_profile_restores_weights`.

## Output schema

Both modes produce a `ProfileReport`:

```python
@dataclass
class LayerSensitivity:
    layer_name: str
    module_class: str
    n_parameters: int
    n_bits: int
    mean_perturbation: float | None  # None for static profiles
    max_perturbation: float | None
```

`ProfileReport.to_dict()` returns a JSON-serialisable structure; `save_json()` writes it.

## How `SelectiveTMR` will use this in v0.5

In v0.1 the strategy ranks by parameter count. From v0.5 the recommended workflow becomes:

```python
profile = sensitivity_profile(model, forward_fn)
strategy = sl.protection.SelectiveTMR.from_profile(profile, top_k_percent=5)
sl.harden(model, strategies=[strategy])
```

The `from_profile` ranking will use `max_perturbation` as the criterion, i.e., protect what *demonstrably* matters under fault injection rather than what's biggest by accident.
