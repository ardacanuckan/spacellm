import { CodeBlock } from "@/components/code-block";
import { SiteShell } from "@/components/site-shell";
import type { Metadata } from "next";
import { QWEN_DEMO } from "./data";
import {
  BitAttackSurfaceChart,
  OrbitSurvivalCalculator,
  ProtectionEfficacyChart,
} from "./playground-charts";

export const metadata: Metadata = {
  title: "Playground",
  description:
    "Interactive playground for SpaceLLM: explore the bit-flip attack surface, the protection-efficacy curve, and the model-survival timer for a chosen orbit and model size.",
};

const REPRODUCE_CMD = `# regenerate the chart data behind this page
uv run python benchmarks/qwen_eval.py
# → docs/assets/{bit_position_attack_surface,qwen_protection_efficacy,orbit_seu_rates}.png
# (the JSON copy of the same numbers lives in
#  packages/web/src/app/playground/data.ts)`;

export default function PlaygroundPage() {
  return (
    <SiteShell>
      <section className="border-b">
        <div className="mx-auto max-w-6xl px-6 py-16">
          <p className="font-mono text-xs uppercase tracking-widest text-(--color-muted-foreground)">
            Playground
          </p>
          <h1 className="mt-3 text-4xl font-medium tracking-tight md:text-5xl">
            Interactive evidence.
          </h1>
          <p className="mt-4 max-w-2xl text-(--color-muted-foreground)">
            Every chart on this page is rendered from real benchmark data captured on a
            developer-laptop CPU running{" "}
            <code className="rounded bg-(--color-muted) px-1.5 py-0.5 font-mono text-sm">
              benchmarks/qwen_eval.py
            </code>{" "}
            against{" "}
            <code className="rounded bg-(--color-muted) px-1.5 py-0.5 font-mono text-sm">
              {QWEN_DEMO.modelId}
            </code>
            . Hover anything; tweak the orbit picker; the numbers all trace back to the bench
            script.
          </p>
        </div>
      </section>

      <section className="border-b">
        <div className="mx-auto max-w-6xl px-6 py-12">
          <h2 className="text-2xl font-medium tracking-tight">
            1 · Which bit positions actually break the model?
          </h2>
          <p className="mt-3 max-w-3xl text-(--color-muted-foreground)">
            One bit flipped on element 0 of <code>lm_head.weight</code>, no protection. Hover any
            bar to see the exact bit + region.
          </p>
          <div className="mt-8">
            <BitAttackSurfaceChart />
          </div>
          <p className="mt-6 max-w-3xl text-sm text-(--color-muted-foreground)">
            Only <strong>bit 30</strong> (the second-most-significant bit of the FP32 exponent)
            broke the model on this prompt. Mantissa flips just nudge magnitude; bit 30 shifts the
            value by ~2¹²⁸ and turns one logit into ±∞, locking the greedy decoder onto a single
            token.
          </p>
        </div>
      </section>

      <section className="border-b">
        <div className="mx-auto max-w-6xl px-6 py-12">
          <h2 className="text-2xl font-medium tracking-tight">
            2 · Does protection actually hold under fault injection?
          </h2>
          <p className="mt-3 max-w-3xl text-(--color-muted-foreground)">
            Three protection levels, five fault counts, two seeds averaged. Hover for the exact
            match-rate at each step.
          </p>
          <div className="mt-8">
            <ProtectionEfficacyChart />
          </div>
          <p className="mt-6 max-w-3xl text-sm text-(--color-muted-foreground)">
            The unprotected curve breaks at <strong>5 random faults</strong>; light TMR holds until{" "}
            <strong>25</strong>; the heavier stack still reproduces the baseline at 25. At 100-fault
            density every defence saturates, that is the regime SpaceBench v0 will benchmark in
            detail.
          </p>
        </div>
      </section>

      <section className="border-b">
        <div className="mx-auto max-w-6xl px-6 py-12">
          <h2 className="text-2xl font-medium tracking-tight">3 · Model survival calculator</h2>
          <p className="mt-3 max-w-3xl text-(--color-muted-foreground)">
            Pick an orbit and a model size; the page computes how often a SEU is expected to land on{" "}
            <em>any</em> bit of the model in real time.
          </p>
          <div className="mt-8">
            <OrbitSurvivalCalculator />
          </div>
        </div>
      </section>

      <section className="border-b">
        <div className="mx-auto max-w-6xl px-6 py-12">
          <h2 className="text-2xl font-medium tracking-tight">
            4 · The same demo, captured end-to-end
          </h2>
          <p className="mt-3 max-w-3xl text-(--color-muted-foreground)">
            Real captured output from{" "}
            <code className="rounded bg-(--color-muted) px-1.5 py-0.5 font-mono text-sm">
              examples/04_qwen_protected.py
            </code>
            . Same prompt, three model copies; identical bit-flip; only the hardened copy survives.
          </p>
          <p className="mt-6 font-mono text-xs uppercase tracking-widest text-(--color-muted-foreground)">
            Prompt
          </p>
          <p className="mt-2 italic">{QWEN_DEMO.prompt}</p>

          <div className="mt-8 grid gap-4 md:grid-cols-3">
            <DemoCard
              title="Clean baseline"
              tone="ok"
              text={QWEN_DEMO.baseline}
              caption="No fault, no protection. The reference."
            />
            <DemoCard
              title="Fault, no protection"
              tone="bad"
              text={QWEN_DEMO.unprotected}
              caption="One bit flipped on lm_head[0].weight. Model collapses to repeated punctuation."
            />
            <DemoCard
              title="Fault + SelectiveTMR(5%)"
              tone="ok"
              text={QWEN_DEMO.protected}
              caption="Same flip, but only one of three replicas; the median vote restores the clean weight."
            />
          </div>
        </div>
      </section>

      <section>
        <div className="mx-auto max-w-6xl px-6 py-12">
          <h2 className="text-2xl font-medium tracking-tight">5 · Reproduce locally</h2>
          <p className="mt-3 max-w-3xl text-(--color-muted-foreground)">
            All numbers above come from the bench script. Re-run it to refresh the PNGs and the JSON
            behind this page.
          </p>
          <div className="mt-6">
            <CodeBlock code={REPRODUCE_CMD} language="bash" />
          </div>
          <p className="mt-4 text-sm text-(--color-muted-foreground)">
            ~3 minutes on a developer-laptop CPU. The first run downloads Qwen2.5-0.5B-Instruct (≈ 1
            GB) from the HuggingFace Hub.
          </p>
        </div>
      </section>
    </SiteShell>
  );
}

function DemoCard({
  title,
  tone,
  text,
  caption,
}: {
  title: string;
  tone: "ok" | "bad";
  text: string;
  caption: string;
}) {
  const accent =
    tone === "ok"
      ? "border-emerald-200 bg-emerald-50 text-emerald-900 dark:border-emerald-900 dark:bg-emerald-950/30 dark:text-emerald-200"
      : "border-red-200 bg-red-50 text-red-900 dark:border-red-900 dark:bg-red-950/30 dark:text-red-200";
  return (
    <article className="rounded-lg border p-4">
      <header
        className={`-mx-4 -mt-4 mb-3 rounded-t-lg border-b px-4 py-2 text-sm font-medium ${accent}`}
      >
        {title}
      </header>
      <p className="text-sm leading-relaxed">{text}</p>
      <p className="mt-3 text-xs text-(--color-muted-foreground)">{caption}</p>
    </article>
  );
}
