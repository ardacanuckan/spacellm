// Empirical results captured from `benchmarks/qwen_eval.py` on
// Qwen/Qwen2.5-0.5B-Instruct. Re-run the benchmark to refresh.

export type ProtectionLevel = "no_protection" | "selective_tmr" | "heavy_stack";

export const PROTECTION_LABELS: Record<ProtectionLevel, string> = {
  no_protection: "No protection",
  selective_tmr: "SelectiveTMR(5%)",
  heavy_stack: "TMR + Attn + LN",
};

export const PROTECTION_COLORS: Record<ProtectionLevel, string> = {
  no_protection: "#dc2626",
  selective_tmr: "#2563eb",
  heavy_stack: "#16a34a",
};

export const FAULT_COUNTS: number[] = [0, 1, 5, 25, 100];

export const PROTECTION_EFFICACY: Record<ProtectionLevel, number[]> = {
  no_protection: [1.0, 1.0, 0.5, 0.0, 0.0],
  selective_tmr: [1.0, 1.0, 1.0, 0.5, 0.0],
  heavy_stack: [1.0, 1.0, 1.0, 1.0, 0.0],
};

export const BIT_ATTACK_SURFACE: { bit: number; broken: boolean }[] = Array.from(
  { length: 32 },
  (_, bit) => ({ bit, broken: bit === 30 }),
);

export type OrbitName = "LEO_ISS_NOMINAL" | "GEO_QUIET" | "MARS_TRANSIT";

export const ORBITS: {
  name: OrbitName;
  label: string;
  altitudeKm: number;
  inclinationDeg: number;
  seuPerBitPerS: number;
  tidGyPerS: number;
  citation: string;
  color: string;
}[] = [
  {
    name: "LEO_ISS_NOMINAL",
    label: "LEO ISS",
    altitudeKm: 400,
    inclinationDeg: 51.6,
    seuPerBitPerS: 3.767e-12,
    tidGyPerS: 4.59e-10,
    citation: "Narici 2015",
    color: "#4682b4",
  },
  {
    name: "GEO_QUIET",
    label: "GEO quiet",
    altitudeKm: 35786,
    inclinationDeg: 0,
    seuPerBitPerS: 2.31e-11,
    tidGyPerS: 5.79e-10,
    citation: "NASA SP-2008-565",
    color: "#daa520",
  },
  {
    name: "MARS_TRANSIT",
    label: "Mars transit",
    altitudeKm: 0,
    inclinationDeg: 0,
    seuPerBitPerS: 5.186e-11,
    tidGyPerS: 5.3e-9,
    citation: "Zeitlin 2013 (MSL/RAD)",
    color: "#cd5c5c",
  },
];

export const QWEN_DEMO = {
  modelId: "Qwen/Qwen2.5-0.5B-Instruct",
  prompt: "In one sentence, why is radiation dangerous for AI hardware in space?",
  baseline:
    "Radiation from the sun and other celestial bodies can damage or destroy electronic components on spacecraft, posing significant risks to mission safety and data integrity.",
  unprotected: "Radiation!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",
  protected:
    "Radiation from the sun and other celestial bodies can damage or destroy electronic components on spacecraft, posing significant risks to mission safety and data integrity.",
};
