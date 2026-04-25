import { CodeBlock } from "@/components/code-block";
import { SiteShell } from "@/components/site-shell";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Architecture",
  description:
    "Module decomposition, the five-phase orbital-LLM-training pipeline, and the math that backs every panel of the bench dashboard.",
};

const LAYOUT = `spacellm/
├── types.py             RadiationStep · FaultEvent · RunRecord · DeviceModel
├── runtime/             sl.harden() · HardenedModel · ProtectionReport aggregator
├── environments/
│   ├── physics.py       weibull_cross_section · irpp_seu_rate_per_cell · petersen_fom
│   ├── orbits.py        OrbitProfile + LEO_ISS_NOMINAL · GEO_QUIET · MARS_TRANSIT
│   ├── devices.py       6 verified RT PolarFire DeviceModel entries (primary-source)
│   ├── physics_lite.py  PhysicsLiteEnvironment(device, orbit), calibrated SEU sampler
│   ├── mcu.py           MCUDistribution + MCUEnvironment cluster amplifier
│   └── statistical.py   StatisticalEnvironment for tests + tutorials
├── protection/
│   ├── tmr.py           SelectiveTMR, top-k Linear modules → 3-replica median vote
│   ├── attention.py     AttentionChecksum, Frobenius fingerprint per parameter
│   ├── kv_cache.py      KVCacheParity, XOR row-parity over decoder cache rows
│   ├── layer_norm.py    LayerNormTMR, TMR for nn.LayerNorm / nn.RMSNorm
│   └── embedding.py     EmbeddingChecksum, fingerprint guard on token + position embeds
├── nn/                  TMRLinear · TMRLayerNorm, composable protected modules
├── _internal/bitops.py  flip_bit primitive (XOR via Tensor.view(int_dtype) reinterpret)
├── profiling/           static + sensitivity-driven layer reports (HTML / JSON)
├── observability/       SQLite-backed RunDB + Run context manager + report renderer
├── bench/               silent_error_rate · bench_protection · BenchResult
├── validation/          WeibullValidationData + validate_against_measurements
└── cli.py               spacellm version | runs | show | profile`;

const PIPELINE = [
  {
    title: "Pre-flight rate prediction",
    body: "PhysicsLiteEnvironment(device, orbit) produces a calibrated SEU rate per bit per second. Every Weibull parameter cites the page of the PDF it came from, no synthesised numbers.",
    code: "rate = env.base.seu_rate_per_bit_per_s()",
  },
  {
    title: "Hardening the model",
    body: "harden() composes a stack of strategies and an environment, then returns a transparent handle. Existing trainers and inference loops keep working.",
    code: "hardened = sl.harden(model, strategies=[...], environment=env)",
  },
  {
    title: "Inject / observe / measure",
    body: "bench_protection runs the canonical loop: sample faults, step the environment, forward, measure. The unique KPI is silent error rate, the fraction of output elements that drift past a threshold.",
    code: "result = sl.bench.bench_protection(hardened.model, ...)",
  },
  {
    title: "Persisted observability",
    body: "Run is a SQLite-backed context manager. The schema is identical for ground rehearsal and on-orbit telemetry, so post-flight analysts can join them.",
    code: 'with Run(name="leo_finetune_step_42") as run: ...',
  },
  {
    title: "Validation against beam-test data",
    body: "validate_against_measurements scores a DeviceModel's Weibull against publicly fetched primary-source (LET, σ) tables, returning RMSE in log10 σ space, R², and an ECSS-style pass flag.",
    code: "validate_against_measurements(POLARFIRE_LSRAM, data, ecss_factor=10)",
  },
];

const FORMULAS = [
  {
    label: "Weibull cross-section",
    expr: "σ(L) = σ_sat · (1 − exp(−((L − L₀) / W)^s))   for L > L₀",
    body: "Four-parameter fit (σ_sat, L₀, W, s). The standard de-facto representation of heavy-ion SEU response.",
  },
  {
    label: "IRPP integral",
    expr: "rate_per_cell = ∫ σ(L) · (dF/dL) dL",
    body: "Convolves the device cross-section with the orbital LET spectrum to produce a mission-realistic SEU rate per cell per second.",
  },
  {
    label: "Bit-flip injection",
    expr: "int_view ← T.view(int_dtype);  int_view[i] ←  int_view[i] XOR (1 << b)",
    body: "Tensor.view(int_dtype) reinterprets memory as the matching-size signed int without copying, dtype-agnostic, works on FP32 / FP16 / BF16 / INT8…",
  },
  {
    label: "TMR median vote",
    expr: "weight = median(W_a, W_b, W_c)   element-wise",
    body: "A single bit-flip in W_a is masked because W_b and W_c agree at that element and the median picks them.",
  },
];

export default function ArchitecturePage() {
  return (
    <SiteShell>
      <section className="border-b">
        <div className="mx-auto max-w-6xl px-6 py-16">
          <p className="font-mono text-xs uppercase tracking-widest text-(--color-muted-foreground)">
            Architecture
          </p>
          <h1 className="mt-3 text-4xl font-medium tracking-tight md:text-5xl">
            One namespace, every layer cited.
          </h1>
          <p className="mt-4 max-w-3xl text-(--color-muted-foreground)">
            SpaceLLM ships as a single Python package. The simulator is deterministic physics, not a
            learned model, and every device parameter cites the primary source it was pulled from.
            Where the validation harness exposes a gap, the README documents the envelope rather
            than hiding it.
          </p>
        </div>
      </section>

      <section className="border-b">
        <div className="mx-auto max-w-6xl px-6 py-12">
          <h2 className="text-2xl font-medium tracking-tight">Package layout</h2>
          <CodeBlock className="mt-6" language="text" code={LAYOUT} />
        </div>
      </section>

      <section className="border-b bg-(--color-muted)/20">
        <div className="mx-auto max-w-6xl px-6 py-16">
          <h2 className="text-2xl font-medium tracking-tight">
            The five-phase orbital-LLM-training pipeline
          </h2>
          <p className="mt-3 max-w-2xl text-(--color-muted-foreground)">
            How a Starcloud-class operator uses the framework end-to-end, from a calibrated
            pre-flight rate budget to a post-flight reconciliation between predicted and observed
            SEUs.
          </p>
          <ol className="mt-10 space-y-6">
            {PIPELINE.map((step, index) => (
              <li key={step.title} className="rounded-lg border bg-(--color-background) p-6">
                <header className="flex items-baseline gap-3">
                  <span className="font-mono text-sm text-(--color-primary)">
                    {String(index + 1).padStart(2, "0")}
                  </span>
                  <h3 className="text-lg font-medium tracking-tight">{step.title}</h3>
                </header>
                <p className="mt-2 text-sm leading-relaxed text-(--color-muted-foreground)">
                  {step.body}
                </p>
                <pre className="mt-3 overflow-x-auto rounded bg-(--color-muted)/60 px-3 py-2 font-mono text-xs">
                  <code>{step.code}</code>
                </pre>
              </li>
            ))}
          </ol>
        </div>
      </section>

      <section>
        <div className="mx-auto max-w-6xl px-6 py-16">
          <h2 className="text-2xl font-medium tracking-tight">The four equations that matter</h2>
          <p className="mt-3 max-w-2xl text-(--color-muted-foreground)">
            Each of these is implemented in a single function inside
            <code className="ml-1 font-mono">spacellm.environments.physics</code>,
            <code className="ml-1 font-mono">spacellm._internal.bitops</code>, or
            <code className="ml-1 font-mono">spacellm.nn.tmr</code>. No black box.
          </p>
          <div className="mt-10 grid gap-6 md:grid-cols-2">
            {FORMULAS.map((formula) => (
              <article key={formula.label} className="rounded-lg border p-6">
                <p className="font-mono text-xs uppercase tracking-widest text-(--color-muted-foreground)">
                  {formula.label}
                </p>
                <pre className="mt-3 overflow-x-auto rounded bg-(--color-muted)/60 px-3 py-2 font-mono text-sm">
                  <code>{formula.expr}</code>
                </pre>
                <p className="mt-3 text-sm leading-relaxed text-(--color-muted-foreground)">
                  {formula.body}
                </p>
              </article>
            ))}
          </div>
        </div>
      </section>
    </SiteShell>
  );
}
