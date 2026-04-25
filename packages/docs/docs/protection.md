# Protection strategies

Every strategy implements the small `Strategy` abstract base in `spacellm.protection.base`. The runtime composes them with `sl.harden()`. Two strategies (`SelectiveTMR`, `LayerNormTMR`) are *structural*: they replace existing modules with multi-replica protected versions. Three (`AttentionChecksum`, `EmbeddingChecksum`, `KVCacheParity`) are *detection-only*: they install hooks that fingerprint or parity-protect the relevant tensors and surface mismatches as detection events.

## SelectiveTMR

Triple-modular redundancy on the largest top-`k`% of `nn.Linear` modules.

```python
sl.protection.SelectiveTMR(top_k_percent=5.0)
```

Each protected module is replaced with `spacellm.nn.TMRLinear`, which holds three independent weight (and bias) replicas and votes element-wise on every forward via `torch.median`. A single bit-flip in any one replica is masked by the other two.

* **Cost.** Memory: 3× the weights of the protected layer. Compute: one median over a stack of three plus the matmul.
* **Sensitivity ranking.** Ranks modules by parameter count by default; an externally-provided sensitivity profile from `spacellm.profiling.sensitivity_profile()` can be passed in.
* **Trainability.** Replicas are frozen by default (`requires_grad=False`).

## AttentionChecksum

Detection-only Frobenius-norm fingerprint per attention parameter.

```python
sl.protection.AttentionChecksum(rtol=1e-5, atol=1e-8)
```

`apply()` walks the module tree, identifies attention modules (`nn.MultiheadAttention` and class names containing `attn` or `attention`), and registers a forward pre-hook that recomputes a per-parameter Frobenius-norm fingerprint and compares it against the reference recorded at hardening time. Mismatches are counted as `n_corruptions_detected`.

* **Why Frobenius norm?** A bit-flip in an exponent bit shifts magnitude by a power of two, which dominates the norm. Per [PrisonBreak](https://arxiv.org/abs/2401.13594), 94% of high-impact attack bits live in the top three exponent bits, so this detector has high recall on the dangerous flips while staying cheap.
* **Recovery.** None; this is a detection-only strategy. Compose with `SelectiveTMR` to gain redundancy.

## LayerNormTMR

TMR for `nn.LayerNorm` and `nn.RMSNorm` modules.

```python
sl.protection.LayerNormTMR(include_rmsnorm=True)
```

`apply()` walks the module tree, identifies normalisation layers, and replaces each with `spacellm.nn.TMRLayerNorm`. The wrapped module computes the median across three γ and β replicas before applying the standard normalisation. Normalisation parameters are tiny in count but disproportionately critical: a corrupted gain element scales every activation through the layer.

* **Caveat.** Custom RMSNorm classes shipped by individual model families (`Qwen2RMSNorm`, `LlamaRMSNorm`) do not currently match `isinstance(module, nn.RMSNorm)` and so are not wrapped. The `include_rmsnorm` flag covers the standard `nn.RMSNorm`.

## EmbeddingChecksum

Detection-only Frobenius fingerprint on `nn.Embedding` modules.

```python
sl.protection.EmbeddingChecksum()
```

The same detection mechanism as `AttentionChecksum`, targeting token and position embeddings. Kept as a separate strategy because token embeddings have a different sensitivity profile from attention weights, and an operator may wish to deploy one without the other.

## KVCacheParity

XOR row-parity over decoder cache tensors.

```python
parity = sl.protection.KVCacheParity()
parity.guard("layer0", key, value)
# ... time passes, perhaps a flip ...
mismatches = parity.verify("layer0", key, value)  # 0, 1, or 2
```

`apply()` is a no-op because the KV cache lives outside the `nn.Module` graph; HuggingFace's decoder cache is a per-layer tuple managed by user code. The strategy provides `guard()` and `verify()` primitives the user wires into the cache write/read path.

* **Caveat.** SEU-driven KV-cache drift in long-context decode has no published baseline defense; this strategy plants the flag with a parity primitive. Reed-Solomon recovery is on the roadmap, not shipped.

## Composing strategies

`sl.harden()` applies strategies in the order given. Strategies that need to wrap modules (e.g. `SelectiveTMR`, `LayerNormTMR`) run before strategies that read the wrapped form (e.g. `AttentionChecksum` reads parameters that may already be `TMRLinear` properties). The recommended baseline:

```python
hardened = sl.harden(
    model,
    strategies=[
        sl.protection.SelectiveTMR(top_k_percent=5),
        sl.protection.AttentionChecksum(),
        sl.protection.LayerNormTMR(),
        sl.protection.EmbeddingChecksum(),
        sl.protection.KVCacheParity(),
    ],
)
```

## Telemetry

Every strategy returns a `ProtectionReport`:

```python
for r in hardened.collect_reports():
    print(r.strategy_name, r.n_corruptions_detected, r.overhead_bytes)
```

`reset_telemetry()` clears counters without unwinding the structural changes.
