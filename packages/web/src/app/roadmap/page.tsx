import { SiteShell } from "@/components/site-shell";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Roadmap",
  description:
    "What's shipped today in SpaceLLM v0.3, what's near-term on the path to v1.0, and what is out of scope.",
};

const SHIPPED = [
  {
    label: "Calibrated radiation environment",
    body: "Weibull cross-section + IRPP integration + MCU clustering. Reference orbits LEO_ISS, GEO_QUIET, MARS_TRANSIT shipped with cited primary-source LET spectra.",
  },
  {
    label: "Verified device catalogue",
    body: "Six RT PolarFire family entries (USRAM, LSRAM, four DFF data patterns), every Weibull parameter and SEL onset taken from the Microchip 2020 radiation report.",
  },
  {
    label: "Five composable protection strategies",
    body: "SelectiveTMR, AttentionChecksum, KVCacheParity, LayerNormTMR, EmbeddingChecksum, wrap any HuggingFace transformer in one harden() call.",
  },
  {
    label: "Validation harness",
    body: "validate_against_measurements scores DeviceModel Weibulls against publicly fetched primary-source (LET, σ) tables with an ECSS-Q-ST-60-15C-style envelope.",
  },
  {
    label: "Bench + observability",
    body: "silent_error_rate / bench_protection / BenchResult; SQLite-backed RunDB; self-contained HTML reports; CLI (spacellm version | runs | show | profile).",
  },
  {
    label: "Empirical evaluation",
    body: "benchmarks/qwen_eval.py produces three multi-panel dashboards (bit-flip attack surface, protection efficacy & cost, mission overview), rerun nightly in CI.",
  },
];

const NEAR_TERM = [
  {
    label: "Independent validation data",
    body: "A second device family beyond RT PolarFire, published Xilinx Versal beam-test data or the Li et al. 2017 transformer fault-injection set, so the simulator is not validated against only its own primary source.",
  },
  {
    label: "Per-strategy ablation in CI",
    body: "Extend the bench workflow to score each protection strategy independently, not only the composed stacks. Numbers go straight into the bench artifact.",
  },
  {
    label: "Real GPU bench at production model size",
    body: "Reproduce the silent-error-rate curve on H100 with a 7 B / 13 B model. Report p50 / p99 latency hit, throughput overhead, and a comparison against FT-Transformer (PPoPP'25) and ATTNChecker.",
  },
  {
    label: "Custom RMSNorm coverage",
    body: "Pattern-match Qwen2RMSNorm and similar custom norm classes so LayerNormTMR is not silently a no-op on common HuggingFace architectures.",
  },
  {
    label: "Methodology preprint",
    body: "An arXiv writeup of the simulator methodology + bench results + the validation envelope, converts the framework from 'GitHub repo' to 'X et al. 2026 methodology'.",
  },
];

const NON_GOALS = [
  "Substitute for a heavy-ion beam test. SpaceLLM predicts SEU behaviour given a Weibull-fitted device; the Weibull itself still has to come from accelerator time.",
  "SEL / SEFI / TID hardware mitigation. These are board / package / process-node concerns; SpaceLLM tracks them as orbit-budgeted bookkeeping, not as something software can mask.",
  "Formal verification. The bench reports a measured silent-error rate over the simulated trajectory; it does not prove the absence of silent corruption under arbitrary fault traces.",
  "Radiation-hardened compute itself. SpaceLLM runs on commodity PyTorch, it assumes your hardware boots and executes; the simulation is about what happens to the data the hardware is computing on.",
  "Cumulative-effect modelling. TID drift over weeks of mission time and displacement-damage degradation are out of scope for the v0.x line.",
];

export default function RoadmapPage() {
  return (
    <SiteShell>
      <section className="border-b">
        <div className="mx-auto max-w-6xl px-6 py-16">
          <p className="font-mono text-xs uppercase tracking-widest text-(--color-muted-foreground)">
            Roadmap
          </p>
          <h1 className="mt-3 text-4xl font-medium tracking-tight md:text-5xl">
            What ships, what&rsquo;s next, what we will not pretend to do.
          </h1>
          <p className="mt-4 max-w-3xl text-(--color-muted-foreground)">
            The current line is v0.3.0-dev. The plan below covers what runs in CI today, what we are
            working on toward v1.0, and what is out of scope. Calendar dates are not promises,
            milestones ship when validation passes, not when a press cycle wants them to.
          </p>
        </div>
      </section>

      <section className="border-b">
        <div className="mx-auto max-w-6xl px-6 py-16">
          <h2 className="text-2xl font-medium tracking-tight">Shipped, v0.3</h2>
          <p className="mt-3 max-w-2xl text-(--color-muted-foreground)">
            All of the following run on every push of the GitHub Actions Python matrix (3.11 · 3.12
            · 3.13). 217 / 217 pytest cases passing, ruff and mypy --strict clean.
          </p>
          <div className="mt-8 grid gap-6 md:grid-cols-2">
            {SHIPPED.map((item) => (
              <article key={item.label} className="rounded-lg border p-6">
                <header className="flex items-baseline gap-2">
                  <span className="text-(--color-radiation)">●</span>
                  <h3 className="text-base font-medium">{item.label}</h3>
                </header>
                <p className="mt-2 text-sm leading-relaxed text-(--color-muted-foreground)">
                  {item.body}
                </p>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section className="border-b bg-(--color-muted)/20">
        <div className="mx-auto max-w-6xl px-6 py-16">
          <h2 className="text-2xl font-medium tracking-tight">Near-term, toward v1.0</h2>
          <p className="mt-3 max-w-2xl text-(--color-muted-foreground)">
            Concrete next moves, ordered by their ability to bridge the gap between &ldquo;GitHub
            repo with a clean test suite&rdquo; and &ldquo;framework an orbital-data-center engineer
            would adopt without hesitation.&rdquo;
          </p>
          <ol className="mt-8 space-y-4">
            {NEAR_TERM.map((item, index) => (
              <li
                key={item.label}
                className="flex gap-4 rounded-lg border bg-(--color-background) p-6"
              >
                <span className="font-mono text-sm text-(--color-primary)">
                  {String(index + 1).padStart(2, "0")}
                </span>
                <div>
                  <h3 className="text-base font-medium">{item.label}</h3>
                  <p className="mt-2 text-sm leading-relaxed text-(--color-muted-foreground)">
                    {item.body}
                  </p>
                </div>
              </li>
            ))}
          </ol>
        </div>
      </section>

      <section>
        <div className="mx-auto max-w-6xl px-6 py-16">
          <h2 className="text-2xl font-medium tracking-tight">Non-goals</h2>
          <p className="mt-3 max-w-2xl text-(--color-muted-foreground)">
            The framework is narrow. To stay honest, the things SpaceLLM does
            <em className="px-1">not</em>
            claim to deliver are listed explicitly:
          </p>
          <ul className="mt-8 space-y-3">
            {NON_GOALS.map((item) => (
              <li
                key={item}
                className="flex gap-3 rounded border-l-2 border-(--color-muted-foreground)/40 bg-(--color-muted)/30 p-4 text-sm leading-relaxed"
              >
                <span className="text-(--color-muted-foreground)">×</span>
                <span>{item}</span>
              </li>
            ))}
          </ul>
        </div>
      </section>
    </SiteShell>
  );
}
