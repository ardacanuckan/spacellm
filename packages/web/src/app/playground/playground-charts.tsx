"use client";

import { useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  BIT_ATTACK_SURFACE,
  FAULT_COUNTS,
  ORBITS,
  type OrbitName,
  PROTECTION_COLORS,
  PROTECTION_EFFICACY,
  PROTECTION_LABELS,
  type ProtectionLevel,
} from "./data";

const PROTECTION_LEVELS: ProtectionLevel[] = ["no_protection", "selective_tmr", "heavy_stack"];

const FAULT_INDEX = (n: number) => Math.max(0, FAULT_COUNTS.indexOf(n));

const efficacyData: Array<{
  faults: number;
  no_protection: number;
  selective_tmr: number;
  heavy_stack: number;
}> = FAULT_COUNTS.map((n, idx) => ({
  faults: n,
  no_protection: PROTECTION_EFFICACY.no_protection[idx] ?? 0,
  selective_tmr: PROTECTION_EFFICACY.selective_tmr[idx] ?? 0,
  heavy_stack: PROTECTION_EFFICACY.heavy_stack[idx] ?? 0,
}));

const bitData = BIT_ATTACK_SURFACE.map(({ bit, broken }) => ({
  bit,
  broken: broken ? 1 : 0,
  region: bit === 31 ? "sign" : bit >= 23 ? "exponent" : "mantissa",
}));

const REGION_COLOR: Record<string, string> = {
  sign: "#a855f7",
  exponent: "#dc2626",
  mantissa: "#94a3b8",
};

export function ProtectionEfficacyChart() {
  const [hoverFaults, setHoverFaults] = useState<number | null>(null);

  return (
    <div>
      <ResponsiveContainer width="100%" height={320}>
        <LineChart
          data={efficacyData}
          onMouseMove={(e) => {
            if (e?.activePayload?.[0]) {
              setHoverFaults(e.activePayload[0].payload.faults);
            }
          }}
          onMouseLeave={() => setHoverFaults(null)}
        >
          <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
          <XAxis
            dataKey="faults"
            label={{ value: "bit-flips injected", position: "insideBottom", offset: -5 }}
          />
          <YAxis
            domain={[0, 1]}
            ticks={[0, 0.25, 0.5, 0.75, 1]}
            label={{
              value: "match rate",
              angle: -90,
              position: "insideLeft",
            }}
          />
          <Tooltip
            formatter={(value: number, name: string) => [
              value.toFixed(2),
              PROTECTION_LABELS[name as ProtectionLevel] ?? name,
            ]}
            contentStyle={{
              fontFamily: "ui-monospace, monospace",
              fontSize: 12,
              borderRadius: 8,
              border: "1px solid var(--color-border)",
            }}
          />
          {PROTECTION_LEVELS.map((level) => (
            <Line
              key={level}
              type="monotone"
              dataKey={level}
              name={level}
              stroke={PROTECTION_COLORS[level]}
              strokeWidth={2.5}
              dot={{ r: 5 }}
              activeDot={{ r: 7 }}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>

      <div className="mt-4 flex flex-wrap gap-3 text-xs">
        {PROTECTION_LEVELS.map((level) => (
          <span key={level} className="inline-flex items-center gap-1.5">
            <span
              className="size-3 rounded-full"
              style={{ background: PROTECTION_COLORS[level] }}
            />
            {PROTECTION_LABELS[level]}
          </span>
        ))}
      </div>

      {hoverFaults !== null ? (
        <div className="mt-4 rounded border bg-(--color-muted) p-3 font-mono text-xs">
          <strong>{hoverFaults}</strong> bit-flips:{" "}
          {PROTECTION_LEVELS.map((level) => {
            const row = efficacyData[FAULT_INDEX(hoverFaults)];
            const value = row ? row[level].toFixed(2) : ",";
            return (
              <span key={level} className="ml-2">
                {PROTECTION_LABELS[level]}{" "}
                <strong style={{ color: PROTECTION_COLORS[level] }}>{value}</strong>
              </span>
            );
          })}
        </div>
      ) : null}
    </div>
  );
}

export function BitAttackSurfaceChart() {
  return (
    <div>
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={bitData}>
          <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
          <XAxis dataKey="bit" tick={{ fontSize: 10 }} />
          <YAxis
            domain={[0, 1.05]}
            ticks={[0, 1]}
            tickFormatter={(v) => (v === 0 ? "clean" : "broken")}
            tick={{ fontSize: 11 }}
          />
          <Tooltip
            formatter={(_value, _name, item) => [
              item.payload.broken ? "BROKEN" : "OK",
              `bit ${item.payload.bit} (${item.payload.region})`,
            ]}
            contentStyle={{
              fontFamily: "ui-monospace, monospace",
              fontSize: 12,
              borderRadius: 8,
              border: "1px solid var(--color-border)",
            }}
          />
          <Bar dataKey="broken" radius={[2, 2, 0, 0]}>
            {bitData.map((d) => (
              <Cell key={d.bit} fill={REGION_COLOR[d.region]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      <div className="mt-4 flex flex-wrap gap-4 text-xs">
        {[
          { label: "mantissa (bits 0–22)", color: REGION_COLOR.mantissa },
          { label: "exponent (bits 23–30)", color: REGION_COLOR.exponent },
          { label: "sign (bit 31)", color: REGION_COLOR.sign },
        ].map(({ label, color }) => (
          <span key={label} className="inline-flex items-center gap-1.5">
            <span className="size-3 rounded" style={{ background: color }} />
            {label}
          </span>
        ))}
      </div>
    </div>
  );
}

export function OrbitSurvivalCalculator() {
  const [paramsM, setParamsM] = useState(500);
  const [orbitName, setOrbitName] = useState<OrbitName>("LEO_ISS_NOMINAL");

  const orbit = ORBITS.find((o) => o.name === orbitName) ?? ORBITS[0];
  if (!orbit) {
    return null;
  }
  const totalBits = paramsM * 1_000_000 * 32; // FP32
  const seuPerSecond = totalBits * orbit.seuPerBitPerS;
  const secondsPerSeu = seuPerSecond > 0 ? 1 / seuPerSecond : Number.POSITIVE_INFINITY;

  const formatTime = (s: number): string => {
    if (s < 1) return `${(s * 1000).toFixed(1)} ms`;
    if (s < 60) return `${s.toFixed(2)} s`;
    if (s < 3600) return `${(s / 60).toFixed(2)} min`;
    if (s < 86400) return `${(s / 3600).toFixed(2)} h`;
    return `${(s / 86400).toFixed(2)} d`;
  };

  return (
    <div className="rounded-lg border p-6">
      <div className="mb-4 grid gap-4 md:grid-cols-2">
        <label className="block">
          <span className="font-mono text-xs uppercase tracking-widest text-(--color-muted-foreground)">
            Model size
          </span>
          <div className="mt-2 flex items-center gap-3">
            <input
              type="range"
              min={50}
              max={7000}
              step={50}
              value={paramsM}
              onChange={(e) => setParamsM(Number(e.target.value))}
              className="flex-1"
            />
            <span className="w-20 font-mono text-sm">{paramsM} M</span>
          </div>
        </label>

        <label className="block">
          <span className="font-mono text-xs uppercase tracking-widest text-(--color-muted-foreground)">
            Orbit
          </span>
          <select
            value={orbitName}
            onChange={(e) => setOrbitName(e.target.value as OrbitName)}
            className="mt-2 w-full rounded border bg-(--color-background) px-3 py-2 font-mono text-sm"
          >
            {ORBITS.map((o) => (
              <option key={o.name} value={o.name}>
                {o.label}, {o.citation}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <Metric label="model bit-area" value={`${(totalBits / 1e9).toFixed(1)} G bits`} />
        <Metric label="expected SEUs / sec" value={seuPerSecond.toFixed(3)} />
        <Metric label="time between SEUs" value={formatTime(secondsPerSeu)} accent />
      </div>

      <p className="mt-4 text-sm text-(--color-muted-foreground)">
        At {orbit.label} on a <strong>{paramsM} M</strong>-parameter FP32 model the radiation
        environment delivers about <strong>{seuPerSecond.toFixed(2)} SEUs every second</strong>.
        Without a runtime defence the inference path will not stay coherent for more than a few
        seconds.
      </p>
    </div>
  );
}

function Metric({
  label,
  value,
  accent = false,
}: {
  label: string;
  value: string;
  accent?: boolean;
}) {
  return (
    <div className={`rounded p-4 ${accent ? "bg-(--color-radiation)/10" : "bg-(--color-muted)"}`}>
      <div className="font-mono text-xs uppercase tracking-widest text-(--color-muted-foreground)">
        {label}
      </div>
      <div className="mt-1 font-mono text-2xl">{value}</div>
    </div>
  );
}
