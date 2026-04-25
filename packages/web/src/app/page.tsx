import { CodeBlock } from "@/components/code-block";
import { SiteShell } from "@/components/site-shell";
import {
  ArrowRightIcon,
  GaugeIcon,
  GitBranchIcon,
  LayersIcon,
  ShieldCheckIcon,
  SignalIcon,
  ZapIcon,
} from "lucide-react";
import Link from "next/link";

const QUICKSTART = `import spacellm as sl
from transformers import AutoModelForCausalLM
from spacellm.environments import (
    PhysicsLiteEnvironment, MCUEnvironment, LEO_ISS_NOMINAL,
)
from spacellm.environments.devices import POLARFIRE_LSRAM
from spacellm.environments.mcu import default_mcu_distribution

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct"
).eval()

env = MCUEnvironment(
    PhysicsLiteEnvironment(POLARFIRE_LSRAM, LEO_ISS_NOMINAL, seed=0),
    distribution=default_mcu_distribution(process_node_nm=14.0),
)

hardened = sl.harden(
    model,
    strategies=[
        sl.protection.SelectiveTMR(top_k_percent=5),
        sl.protection.AttentionChecksum(),
        sl.protection.LayerNormTMR(),
        sl.protection.EmbeddingChecksum(),
        sl.protection.KVCacheParity(),
    ],
    environment=env,
)
print(hardened.report())`;

const FEATURES = [
  {
    icon: ShieldCheckIcon,
    title: "Five composable protection strategies",
    body: "SelectiveTMR (median-vote redundancy), AttentionChecksum (Frobenius fingerprint), KVCacheParity (XOR row-parity), LayerNormTMR, EmbeddingChecksum. Wrap any HuggingFace transformer in one harden() call.",
  },
  {
    icon: GaugeIcon,
    title: "Calibrated environment, not assumed",
    body: "Weibull cross-section, IRPP integration, MCU clustering, predict SEU/bit/s for a chosen (device, orbit) and validate against publicly fetched primary-source beam-test data within an explicit ECSS-style envelope.",
  },
  {
    icon: SignalIcon,
    title: "Built for orbital LLM training",
    body: "Made for Starcloud-class on-orbit data centers and satellite ML engineers. Detects and budgets the silent corruption that turns a 6-hour training step into four hours of wasted solar power.",
  },
];

const SHIPPED = [
  {
    icon: LayersIcon,
    label: "Framework modules",
    body: "environments · protection · runtime · profiling · observability · bench · validation · cli",
  },
  {
    icon: ZapIcon,
    label: "Verified device profiles",
    body: "6 RT PolarFire family entries, every Weibull parameter cited to a fetched primary source",
  },
  {
    icon: GitBranchIcon,
    label: "Engineering status",
    body: "217 / 217 tests · ruff clean · mypy --strict (43 source files) · MkDocs strict · CI matrix Python 3.11 · 3.12 · 3.13",
  },
];

export default function Home() {
  return (
    <SiteShell>
      <Hero />
      <ThreatBanner />
      <Quickstart />
      <Features />
      <Shipped />
      <Cta />
    </SiteShell>
  );
}

function Hero() {
  return (
    <section className="border-b">
      <div className="mx-auto max-w-6xl px-6 py-20">
        <div className="mb-6 inline-flex items-center gap-2 rounded-full border px-3 py-1 font-mono text-xs text-(--color-muted-foreground)">
          <span className="size-1.5 animate-pulse rounded-full bg-(--color-radiation)" />
          radiation-tolerant runtime · v0.3.0-dev
        </div>
        <h1 className="max-w-4xl text-balance font-sans text-5xl font-medium tracking-tight md:text-6xl">
          The Python framework for{" "}
          <span className="text-(--color-primary)">on-orbit LLM training and inference.</span>
        </h1>
        <p className="mt-6 max-w-2xl text-pretty text-lg text-(--color-muted-foreground)">
          Operators of orbital data centers, Starcloud-class LEO solar compute, satellite-bus
          inference, train transformers on hardware that takes single-event upsets while the job is
          in flight. SpaceLLM is the runtime layer that catches, masks, and budgets that silent
          corruption while your CUDA stack keeps running.
        </p>
        <div className="mt-10 flex flex-wrap items-center gap-3">
          <Link
            href="/architecture"
            className="inline-flex items-center gap-2 rounded-md bg-(--color-foreground) px-4 py-2 text-sm font-medium text-(--color-background) transition-opacity hover:opacity-90"
          >
            See the architecture
            <ArrowRightIcon className="size-4" />
          </Link>
          <Link
            href="/playground"
            className="inline-flex items-center gap-2 rounded-md border px-4 py-2 text-sm font-medium transition-colors hover:bg-(--color-muted)"
          >
            Interactive playground
          </Link>
          <a
            href="https://github.com/ardacanuckan/spacellm"
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-2 rounded-md border px-4 py-2 text-sm font-medium transition-colors hover:bg-(--color-muted)"
          >
            View on GitHub
          </a>
        </div>
      </div>
    </section>
  );
}

function ThreatBanner() {
  return (
    <section className="border-b bg-(--color-muted)/30">
      <div className="mx-auto grid max-w-6xl gap-8 px-6 py-14 md:grid-cols-3">
        <Stat
          value="1 bit"
          unit="flip"
          caption="…destroys MMLU on Qwen2.5-7B (71% → 0%), SBFA, arXiv:2509.21843"
        />
        <Stat
          value="3 bits"
          unit="of 10¹⁰"
          caption="…collapse any LLM under AttentionBreaker, arXiv:2411.13757"
        />
        <Stat
          value="77.4 → 7.2"
          unit="F1"
          caption="…on SQuAD when KV-cache rows drift in long-context decode, arXiv:2510.17098"
        />
      </div>
    </section>
  );
}

function Stat({ value, unit, caption }: { value: string; unit: string; caption: string }) {
  return (
    <div>
      <div className="flex items-baseline gap-2 font-mono">
        <span className="text-3xl font-medium tracking-tight">{value}</span>
        <span className="text-sm text-(--color-muted-foreground)">{unit}</span>
      </div>
      <p className="mt-2 text-sm leading-relaxed text-(--color-muted-foreground)">{caption}</p>
    </div>
  );
}

function Quickstart() {
  return (
    <section className="border-b">
      <div className="mx-auto max-w-6xl px-6 py-16 md:grid md:grid-cols-2 md:gap-12">
        <div>
          <h2 className="text-3xl font-medium tracking-tight">
            Drop-in protection on any HuggingFace transformer.
          </h2>
          <p className="mt-4 text-(--color-muted-foreground)">
            <code className="rounded bg-(--color-muted) px-1.5 py-0.5 font-mono text-sm">
              spacellm.harden()
            </code>{" "}
            wraps a <code className="font-mono">torch.nn.Module</code> with a stack of protection
            strategies and an optional radiation environment, then returns a transparent handle.
            Forward and backward passes still go through PyTorch, SpaceLLM only adds the layer
            underneath that knows about bit-flips.
          </p>
          <p className="mt-4 text-(--color-muted-foreground)">
            Built on PyTorch 2 + Hugging Face Transformers. Apache-2.0. The full bench (
            <code className="font-mono">benchmarks/qwen_eval.py</code>) runs on a developer-laptop
            CPU in roughly three minutes.
          </p>
          <div className="mt-6">
            <Link
              href="/architecture"
              className="inline-flex items-center gap-1 text-sm font-medium text-(--color-primary) hover:underline"
            >
              How it works under the hood
              <ArrowRightIcon className="size-4" />
            </Link>
          </div>
        </div>
        <div className="mt-10 md:mt-0">
          <CodeBlock language="python" code={QUICKSTART} />
        </div>
      </div>
    </section>
  );
}

function Features() {
  return (
    <section className="border-b">
      <div className="mx-auto max-w-6xl px-6 py-16">
        <h2 className="text-3xl font-medium tracking-tight">What the framework gives you</h2>
        <p className="mt-3 max-w-2xl text-(--color-muted-foreground)">
          Three layers, each composable, each calibrated against a fetched primary source. PyTorch
          handles the math; SpaceLLM handles what happens when the bits underneath mutate mid-step.
        </p>
        <div className="mt-10 grid gap-6 md:grid-cols-3">
          {FEATURES.map((feature) => (
            <article
              key={feature.title}
              className="rounded-lg border p-6 transition-colors hover:bg-(--color-muted)/40"
            >
              <feature.icon className="size-6 text-(--color-primary)" />
              <h3 className="mt-4 text-lg font-medium">{feature.title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-(--color-muted-foreground)">
                {feature.body}
              </p>
            </article>
          ))}
        </div>
      </div>
    </section>
  );
}

function Shipped() {
  return (
    <section className="border-b bg-(--color-muted)/20">
      <div className="mx-auto max-w-6xl px-6 py-16">
        <h2 className="text-3xl font-medium tracking-tight">What ships in v0.3</h2>
        <p className="mt-3 max-w-2xl text-(--color-muted-foreground)">
          Inference + statistical and physics-lite environments + the full validation harness. Not a
          slide deck, every claim below is reproduced by the CI matrix on every push.
        </p>
        <div className="mt-10 grid gap-6 md:grid-cols-3">
          {SHIPPED.map((item) => (
            <div key={item.label} className="rounded-lg border bg-(--color-background) p-6">
              <item.icon className="size-6 text-(--color-radiation)" />
              <h3 className="mt-4 font-mono text-xs uppercase tracking-widest text-(--color-muted-foreground)">
                {item.label}
              </h3>
              <p className="mt-2 text-sm leading-relaxed">{item.body}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function Cta() {
  return (
    <section>
      <div className="mx-auto max-w-6xl px-6 py-20 text-center">
        <h2 className="text-3xl font-medium tracking-tight md:text-4xl">
          Read the empirical results.
        </h2>
        <p className="mx-auto mt-4 max-w-2xl text-(--color-muted-foreground)">
          The interactive playground renders bit-flip attack-surface, protection efficacy, and
          mission-realistic SEU rates from the same benchmark data the nightly CI workflow uploads
          as artifacts.
        </p>
        <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
          <Link
            href="/playground"
            className="inline-flex items-center gap-2 rounded-md bg-(--color-foreground) px-4 py-2 text-sm font-medium text-(--color-background) transition-opacity hover:opacity-90"
          >
            Open the playground
            <ArrowRightIcon className="size-4" />
          </Link>
          <Link
            href="/roadmap"
            className="inline-flex items-center gap-2 rounded-md border px-4 py-2 text-sm font-medium transition-colors hover:bg-(--color-muted)"
          >
            What ships next
          </Link>
        </div>
      </div>
    </section>
  );
}
