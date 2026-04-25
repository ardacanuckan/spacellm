"""Empirical evaluation of SpaceLLM on Qwen2.5-0.5B-Instruct.

Produces three multi-panel figures used by the project README:

* ``docs/assets/bit_position_attack_surface.png``, two stacked panels
  on a shared bit-position x-axis. The upper panel is the *empirical*
  result (each FP32 bit flipped on element 0 of ``lm_head.weight``,
  did the greedy probe still match the baseline?). The lower panel
  is the *theoretical* relative magnitude shift |Δw / w| for a
  representative weight, which is what makes the upper panel make
  physical sense, exponent flips shift the value by a power of two.

* ``docs/assets/qwen_protection_efficacy.png``, a 2×2 dashboard.
  Top row is the run-time *empirical* dimension (match-rate and
  silent-error-rate as the fault count climbs, three protection
  levels). Bottom row is the deploy-time *static* dimension
  (per-strategy memory overhead and per-strategy parameter coverage)
  so a reader can map "what does each strategy cost me" alongside
  "what does each strategy buy me".

* ``docs/assets/orbit_seu_rates.png``, a 2×2 mission-overview
  dashboard. SEU/bit/s heatmap across the full RT PolarFire family
  × three reference orbits; the LET differential-flux spectra that
  drive those rates; SEU events per mission day on a 7B FP16 model;
  and TID dose budget per mission day with the RT PolarFire 300
  krad(SiO₂) datasheet limit overlaid.

Speed-optimised: each empirical data point is a 4-token greedy probe
rather than a full generation, so the whole benchmark finishes in
roughly 90 s on a developer-laptop CPU. The static / analytical
panels add no extra inference cost.

Run with::

    uv run python benchmarks/qwen_eval.py
"""

from __future__ import annotations

import copy
import gc
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm

import spacellm as sl
from spacellm._internal.bitops import flip_bit
from spacellm.environments import (
    GEO_QUIET,
    LEO_ISS_NOMINAL,
    MARS_TRANSIT,
    PhysicsLiteEnvironment,
)
from spacellm.environments.devices import (
    POLARFIRE_DFF_ALL_ONE,
    POLARFIRE_DFF_ALL_ZERO,
    POLARFIRE_DFF_CB_SLOW,
    POLARFIRE_DFF_CHECKERBOARD,
    POLARFIRE_LSRAM,
    POLARFIRE_USRAM,
)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover
    print("This benchmark needs the [hf] extras: pip install 'spacellm[hf]'")
    sys.exit(1)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "In one sentence, why is radiation dangerous for AI hardware in space?"
PROBE_TOKENS = 4

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "docs" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "font.family": ["DejaVu Sans"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "legend.frameon": False,
        "savefig.bbox": "tight",
    },
)

PALETTE = {
    "none": "#dc2626",
    "tmr": "#2563eb",
    "heavy": "#16a34a",
    "exponent": "#dc2626",
    "mantissa": "#94a3b8",
    "sign": "#a855f7",
    "leo": "#4682b4",
    "geo": "#daa520",
    "mars": "#cd5c5c",
    "warn": "#f59e0b",
    "limit": "#7c2d12",
}

ORBITS = (LEO_ISS_NOMINAL, GEO_QUIET, MARS_TRANSIT)
ORBIT_COLORS = {
    "LEO_ISS_NOMINAL": PALETTE["leo"],
    "GEO_QUIET": PALETTE["geo"],
    "MARS_TRANSIT": PALETTE["mars"],
}

DEVICES = (
    ("USRAM", POLARFIRE_USRAM),
    ("LSRAM", POLARFIRE_LSRAM),
    ("DFF\nall-1", POLARFIRE_DFF_ALL_ONE),
    ("DFF\nall-0", POLARFIRE_DFF_ALL_ZERO),
    ("DFF\nckbd", POLARFIRE_DFF_CHECKERBOARD),
    ("DFF\nckbd-slow", POLARFIRE_DFF_CB_SLOW),
)

# Reference workload for translating "rate per bit" into something a
# reader can feel: a 7B-parameter model held in FP16 (= 14 G bytes =
# 1.12 × 10¹¹ bits of model weight). Quoted as "7 B FP16".
MODEL_WEIGHT_BITS = 7e9 * 16


def _prompt_text(tok: AutoTokenizer) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _probe(model, tok, prompt_text: str) -> tuple[tuple[int, ...], torch.Tensor]:
    """Greedy probe + last-token logits for the silent-error-rate metric."""
    inputs = tok(prompt_text, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=PROBE_TOKENS,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )
    new_ids = out.sequences[0][inputs.input_ids.shape[1] :].tolist()
    scores = out.scores[0][0].detach().float()
    return tuple(new_ids), scores


def chart_bit_position_attack_surface(base, tok) -> None:
    print("\n[chart 1] bit-position attack surface (32 short probes)…", flush=True)
    prompt_text = _prompt_text(tok)
    baseline_tokens, _ = _probe(base, tok, prompt_text)
    print(f"  baseline tokens: {baseline_tokens}", flush=True)

    matches: list[bool] = []
    for bit_pos in range(32):
        model = copy.deepcopy(base).eval()
        flip_bit(model.lm_head.weight.data, 0, bit_pos)
        try:
            probe_tokens, _ = _probe(model, tok, prompt_text)
        except Exception:
            probe_tokens = (-1,)
        ok = probe_tokens == baseline_tokens
        matches.append(ok)
        del model
        gc.collect()
        print(f"  bit {bit_pos:2d}: {'OK' if ok else 'BROKEN'}", flush=True)

    matches_arr = np.array(matches, dtype=float)

    # FP32 IEEE-754 layout: bits 0..22 mantissa, 23..30 exponent, 31 sign.
    # For a representative weight w₀, flipping bit b of the IEEE encoding
    # produces a Δw whose magnitude depends on which field was hit. We
    # compute |Δw / w₀| for w₀ = 0.05 (typical lm_head scale) so the
    # x-axis lines up with the empirical bar.
    w0 = np.float32(0.05)
    w0_int = w0.view(np.uint32).item()
    rel_shift = np.zeros(32)
    for b in range(32):
        flipped = np.uint32(w0_int ^ (1 << b))
        w_flipped = np.frombuffer(flipped.tobytes(), dtype=np.float32)[0]
        if not np.isfinite(w_flipped):
            rel_shift[b] = 1e18  # Inf / NaN flips, clamped for log plot
        else:
            rel_shift[b] = abs(float(w_flipped) - float(w0)) / abs(float(w0))

    colors = [
        PALETTE["sign"]
        if b == 31
        else PALETTE["exponent"]
        if 23 <= b <= 30
        else PALETTE["mantissa"]
        for b in range(32)
    ]

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.6), sharex=True, height_ratios=[1.2, 1.0])
    ax_top, ax_bot = axes

    ax_top.bar(
        range(32), matches_arr * 0 + 1 - matches_arr, color=colors, edgecolor="white", linewidth=0.4
    )
    ax_top.set_ylim(-0.05, 1.15)
    ax_top.set_yticks([0.0, 1.0])
    ax_top.set_yticklabels(["clean", "broken"])
    ax_top.set_ylabel("Empirical: did the\nprobe match baseline?")
    ax_top.set_title(
        "Bit-flip attack surface, Qwen2.5-0.5B, single bit-flip on lm_head.weight[0] (no protection)",
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=PALETTE["mantissa"], label="mantissa (bits 0–22)"),
        plt.Rectangle((0, 0), 1, 1, color=PALETTE["exponent"], label="exponent (bits 23–30)"),
        plt.Rectangle((0, 0), 1, 1, color=PALETTE["sign"], label="sign (bit 31)"),
    ]
    ax_top.legend(handles=handles, loc="upper left", fontsize=9)

    ax_bot.bar(range(32), rel_shift, color=colors, edgecolor="white", linewidth=0.4)
    ax_bot.set_yscale("log")
    ax_bot.set_ylim(1e-8, 1e20)
    ax_bot.set_xlabel("Bit position in the FP32 element (0 = LSB, 31 = sign)")
    ax_bot.set_ylabel("Theoretical: |Δw / w₀|\nfor w₀ = 0.05 (log scale)")
    ax_bot.set_xticks(range(32))
    ax_bot.set_xticklabels(range(32), fontsize=8)
    ax_bot.axhline(
        1.0, color=PALETTE["limit"], linestyle="--", linewidth=1, alpha=0.7, label="100 % shift"
    )
    ax_bot.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    out_path = ASSETS / "bit_position_attack_surface.png"
    plt.savefig(out_path)
    plt.close()
    print(f"  → {out_path.relative_to(ROOT)}", flush=True)


def _inject_random_flips(model, n_flips: int, seed: int) -> int:
    rng = np.random.default_rng(seed)
    params = [(name, p.data) for name, p in model.named_parameters() if p.numel() > 0]
    sizes = np.array([p.numel() for _, p in params])
    cum = np.cumsum(sizes)
    total = int(cum[-1])
    applied = 0
    for _ in range(n_flips):
        flat = int(rng.integers(0, total))
        idx = int(np.searchsorted(cum, flat, side="right"))
        local = flat - (int(cum[idx - 1]) if idx > 0 else 0)
        bits_per_elem = params[idx][1].element_size() * 8
        if rng.random() < 0.8:
            high = bits_per_elem - 1
            low = max(high - 8, 0)
            bit_pos = int(rng.integers(low, high))
        else:
            bit_pos = int(rng.integers(0, bits_per_elem))
        try:
            flip_bit(params[idx][1], local, bit_pos)
            applied += 1
        except Exception:
            pass
    return applied


def _model_param_bytes(model) -> int:
    return sum(p.element_size() * p.numel() for p in model.parameters())


_MODULE_COUNT_KEYS = (
    "n_wrapped_modules",  # SelectiveTMR, LayerNormTMR
    "n_protected_modules",  # AttentionChecksum, EmbeddingChecksum
    "n_guarded_pairs",  # KVCacheParity
)


def _strategy_static_profile(base, label: str, factory) -> tuple[float, int]:
    """Return (memory_overhead_bytes, modules_under_protection_count).

    Each strategy reports its scope under a different `extra` key ,
    structural strategies (TMR, LayerNormTMR) expose
    ``n_wrapped_modules``, detection strategies (Attention/Embedding)
    expose ``n_protected_modules``, and KVCacheParity exposes
    ``n_guarded_pairs``. We pick whichever the strategy populated so
    the bar chart shows a comparable scope per strategy.
    """
    base_bytes = _model_param_bytes(base)
    model = copy.deepcopy(base).eval()
    strat = factory()
    sl.harden(model, strategies=[strat])
    delta_bytes = _model_param_bytes(model) - base_bytes
    rep = strat.collect_report()
    n_modules = 0
    for key in _MODULE_COUNT_KEYS:
        if key in rep.extra:
            n_modules = int(rep.extra[key])
            break
    del model
    gc.collect()
    print(
        f"  static profile {label:24s} +{delta_bytes / 1e6:7.2f} MB  n_modules={n_modules}",
        flush=True,
    )
    return float(delta_bytes), n_modules


def chart_protection_efficacy(base, tok) -> None:
    print("\n[chart 2] protection efficacy dashboard…", flush=True)
    prompt_text = _prompt_text(tok)
    baseline_tokens, baseline_logits = _probe(base, tok, prompt_text)

    fault_counts = [0, 1, 5, 25, 100]
    levels = [
        ("no protection", "none", []),
        ("SelectiveTMR(5%)", "tmr", [sl.protection.SelectiveTMR(top_k_percent=5.0)]),
        (
            "TMR + Attn + LN",
            "heavy",
            [
                sl.protection.SelectiveTMR(top_k_percent=10.0),
                sl.protection.AttentionChecksum(),
                sl.protection.LayerNormTMR(),
            ],
        ),
    ]
    seeds = (1, 2)
    match_rate: dict[str, list[float]] = {label: [] for label, _, _ in levels}
    silent_err: dict[str, list[float]] = {label: [] for label, _, _ in levels}

    for label, _key, strats in levels:
        for n in fault_counts:
            ok_count = 0
            divergences: list[float] = []
            for seed in seeds:
                model = copy.deepcopy(base).eval()
                if strats:
                    fresh = [copy.deepcopy(s) for s in strats]
                    sl.harden(model, strategies=fresh)
                _inject_random_flips(model, n, seed=seed)
                try:
                    probe_tokens, probe_logits = _probe(model, tok, prompt_text)
                except Exception:
                    probe_tokens = (-1,)
                    probe_logits = baseline_logits.clone()
                if probe_tokens == baseline_tokens:
                    ok_count += 1
                # Silent-error rate: fraction of vocab logits whose
                # divergence from baseline exceeds an absolute threshold.
                # This is the smooth analogue of the discrete match check.
                divergence = (probe_logits - baseline_logits).abs()
                if torch.isnan(divergence).any() or torch.isinf(divergence).any():
                    ser = 1.0
                else:
                    ser = float((divergence > 1.0).float().mean().item())
                divergences.append(ser)
                del model
                gc.collect()
            match_rate[label].append(ok_count / len(seeds))
            silent_err[label].append(float(np.mean(divergences)))
            print(
                f"  {label:18s} | {n:4d} flips | match={match_rate[label][-1]:.2f}  "
                f"silent_err={silent_err[label][-1]:.3f}",
                flush=True,
            )

    # Static / deploy-time profiles (memory + coverage) per individual strategy.
    static_strats = [
        ("SelectiveTMR\n(5 %)", lambda: sl.protection.SelectiveTMR(top_k_percent=5.0)),
        ("AttentionChecksum", sl.protection.AttentionChecksum),
        ("LayerNormTMR", sl.protection.LayerNormTMR),
        ("EmbeddingChecksum", sl.protection.EmbeddingChecksum),
        ("KVCacheParity", sl.protection.KVCacheParity),
    ]
    static_results: list[tuple[str, float, int]] = []
    for name, factory in static_strats:
        try:
            mem, n_modules = _strategy_static_profile(base, name.replace("\n", " "), factory)
        except Exception as exc:
            print(f"  static profile {name}: FAILED ({exc})", flush=True)
            mem, n_modules = 0.0, 0
        static_results.append((name, mem, n_modules))

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8))
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    for label, key, _ in levels:
        ax_a.plot(
            fault_counts,
            match_rate[label],
            marker="o",
            linewidth=2.2,
            markersize=8,
            color=PALETTE[key],
            label=label,
        )
    ax_a.set_xscale("symlog", linthresh=1)
    ax_a.set_xticks(fault_counts)
    ax_a.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax_a.set_xlabel("Injected bit-flips (random, exponent-biased)")
    ax_a.set_ylabel("P(probe matches baseline)")
    ax_a.set_ylim(-0.05, 1.05)
    ax_a.set_title("(a) Empirical efficacy, output integrity vs fault count")
    ax_a.legend(loc="lower left", fontsize=9)

    for label, key, _ in levels:
        ax_b.plot(
            fault_counts,
            silent_err[label],
            marker="s",
            linewidth=2.2,
            markersize=7,
            color=PALETTE[key],
            label=label,
        )
    ax_b.set_xscale("symlog", linthresh=1)
    ax_b.set_xticks(fault_counts)
    ax_b.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax_b.set_xlabel("Injected bit-flips")
    ax_b.set_ylabel("Silent-error rate (logit divergence > 1)")
    ax_b.set_ylim(-0.05, 1.05)
    ax_b.set_title("(b) Silent-error rate, smooth-degradation view")
    ax_b.legend(loc="upper left", fontsize=9)

    names = [r[0] for r in static_results]
    mem_mb = [r[1] / 1e6 for r in static_results]
    n_modules_per_strat = [r[2] for r in static_results]
    bar_colors = [PALETTE["tmr"], PALETTE["heavy"], "#0ea5e9", "#a855f7", "#f97316"]

    flat_names = [n.replace("\n", " ") for n in names]
    y_pos = np.arange(len(flat_names))

    bars_c = ax_c.barh(y_pos, mem_mb, color=bar_colors, edgecolor="white", linewidth=0.5)
    for bar, value in zip(bars_c, mem_mb, strict=True):
        ax_c.text(
            bar.get_width() + max(mem_mb) * 0.02,
            bar.get_y() + bar.get_height() / 2.0,
            f"{value:.1f} MB" if value > 0 else "0",
            ha="left",
            va="center",
            fontsize=9,
        )
    ax_c.set_yticks(y_pos)
    ax_c.set_yticklabels(flat_names, fontsize=9)
    ax_c.invert_yaxis()
    ax_c.set_xlabel("Added parameter bytes (MB)")
    ax_c.set_xlim(0, max(mem_mb) * 1.2 if max(mem_mb) > 0 else 1)
    ax_c.set_title("(c) Per-strategy memory overhead on Qwen2.5-0.5B")

    bars_d = ax_d.barh(
        y_pos,
        n_modules_per_strat,
        color=bar_colors,
        edgecolor="white",
        linewidth=0.5,
    )
    for bar, value in zip(bars_d, n_modules_per_strat, strict=True):
        ax_d.text(
            bar.get_width() + max(n_modules_per_strat) * 0.02 + 0.5,
            bar.get_y() + bar.get_height() / 2.0,
            str(value),
            ha="left",
            va="center",
            fontsize=9,
        )
    ax_d.set_yticks(y_pos)
    ax_d.set_yticklabels(flat_names, fontsize=9)
    ax_d.invert_yaxis()
    ax_d.set_xlabel("Modules under protection (count)")
    ax_d.set_xlim(0, max(max(n_modules_per_strat, default=10) * 1.20, 10))
    ax_d.set_title("(d) Per-strategy scope, modules wrapped or watched")
    ax_d.text(
        0.99,
        -0.20,
        "LayerNormTMR matches nn.LayerNorm; Qwen2.5 ships Qwen2RMSNorm (custom).\n"
        "KVCacheParity activates after the first forward call (cache must exist first).",
        transform=ax_d.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#475569",
        style="italic",
    )

    fig.suptitle(
        "SpaceLLM protection efficacy & cost, Qwen2.5-0.5B-Instruct (FP32)",
        fontsize=13,
        fontweight="bold",
        y=1.00,
    )
    plt.tight_layout()
    out_path = ASSETS / "qwen_protection_efficacy.png"
    plt.savefig(out_path)
    plt.close()
    print(f"  → {out_path.relative_to(ROOT)}", flush=True)


def chart_orbit_rates() -> None:
    print("\n[chart 3] mission overview dashboard…", flush=True)

    # Panel A: SEU/bit/s heatmap [device × orbit].
    rate_matrix = np.zeros((len(DEVICES), len(ORBITS)))
    for i, (_, dev) in enumerate(DEVICES):
        for j, orbit in enumerate(ORBITS):
            env = PhysicsLiteEnvironment(dev, orbit)
            rate_matrix[i, j] = float(env.mission_seu_rate_per_bit_per_s)
    print(f"  rate matrix range: [{rate_matrix.min():.2e}, {rate_matrix.max():.2e}]", flush=True)

    # Panel B: differential flux dF/dL per orbit (LET spectrum).
    # Panel C: events/day on the 7B FP16 reference workload.
    seconds_per_day = 86400.0
    events_per_day = {
        orbit.name: float(
            PhysicsLiteEnvironment(POLARFIRE_LSRAM, orbit).mission_seu_rate_per_bit_per_s
        )
        * MODEL_WEIGHT_BITS
        * seconds_per_day
        for orbit in ORBITS
    }
    # Panel D: TID dose per mission day (Gy(Si) → krad(SiO₂) ≈ × 100 / 1e3).
    tid_krad_per_day = {
        orbit.name: orbit.tid_rate_gy_per_s * seconds_per_day * 100.0 / 1000.0 for orbit in ORBITS
    }

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    # --- (a) heatmap
    orbit_labels = [o.name.replace("_NOMINAL", "").replace("_", " ") for o in ORBITS]
    device_labels = [d[0] for d in DEVICES]
    norm = LogNorm(vmin=max(rate_matrix.min(), 1e-30), vmax=rate_matrix.max())
    im = ax_a.imshow(rate_matrix, cmap="YlOrRd", norm=norm, aspect="auto")
    ax_a.set_xticks(range(len(orbit_labels)))
    ax_a.set_xticklabels(orbit_labels, fontsize=9)
    ax_a.set_yticks(range(len(device_labels)))
    ax_a.set_yticklabels(device_labels, fontsize=9)
    for i in range(rate_matrix.shape[0]):
        for j in range(rate_matrix.shape[1]):
            ax_a.text(
                j,
                i,
                f"{rate_matrix[i, j]:.1e}",
                ha="center",
                va="center",
                fontsize=8,
                color="black" if rate_matrix[i, j] / rate_matrix.max() < 0.6 else "white",
            )
    ax_a.set_title("(a) SEU rate per bit per second\nRT PolarFire family × reference orbits")
    fig.colorbar(im, ax=ax_a, label="SEU · bit⁻¹ · s⁻¹ (log)", shrink=0.8)

    # --- (b) LET spectrum
    for orbit in ORBITS:
        ax_b.plot(
            orbit.let_grid_mev_cm2_per_mg,
            orbit.flux_per_let_per_s,
            linewidth=2.4,
            color=ORBIT_COLORS[orbit.name],
            label=orbit.name.replace("_NOMINAL", ""),
        )
    ax_b.axvspan(
        POLARFIRE_LSRAM.onset_let_mev_cm2_per_mg,
        max(orbit.let_grid_mev_cm2_per_mg.max() for orbit in ORBITS),
        color=PALETTE["warn"],
        alpha=0.10,
        label="LSRAM susceptible band\n(L > L₀)",
    )
    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_xlabel("Linear energy transfer L (MeV·cm²·mg⁻¹)")
    ax_b.set_ylabel("Differential flux dF/dL\n(particles · cm⁻² · s⁻¹ · (MeV·cm²/mg)⁻¹)")
    ax_b.set_title("(b) Orbital LET spectrum, what the IRPP integral consumes")
    ax_b.legend(loc="lower left", fontsize=9)

    # --- (c) events per mission day (RT PolarFire LSRAM × 7B FP16 model)
    names = list(events_per_day.keys())
    vals = list(events_per_day.values())
    bars_c = ax_c.bar(
        [n.replace("_NOMINAL", "").replace("_", " ") for n in names],
        vals,
        color=[ORBIT_COLORS[n] for n in names],
        edgecolor="white",
        linewidth=0.5,
    )
    for bar, value in zip(bars_c, vals, strict=True):
        ax_c.text(
            bar.get_x() + bar.get_width() / 2.0,
            value,
            f"{value:.2e}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax_c.set_yscale("log")
    ax_c.set_ylabel("SEU events on a 7 B FP16 model per mission day")
    ax_c.set_title("(c) What '10⁻¹² /bit/s' actually means\nfor a 7 B FP16 model in flight")

    # --- (d) TID dose per mission day vs RT PolarFire 300 krad limit
    krad_vals = list(tid_krad_per_day.values())
    bars_d = ax_d.bar(
        [n.replace("_NOMINAL", "").replace("_", " ") for n in names],
        krad_vals,
        color=[ORBIT_COLORS[n] for n in names],
        edgecolor="white",
        linewidth=0.5,
    )
    for bar, value in zip(bars_d, krad_vals, strict=True):
        ax_d.text(
            bar.get_x() + bar.get_width() / 2.0,
            value,
            f"{value:.2e}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax_d.set_yscale("log")
    ax_d.axhline(
        300.0,
        color=PALETTE["limit"],
        linestyle="--",
        linewidth=1.5,
        label="RT PolarFire datasheet limit (300 krad SiO₂)",
    )
    ax_d.set_ylabel("TID dose per mission day (krad SiO₂)")
    ax_d.set_title("(d) TID budget, orbits vs the datasheet ceiling")
    ax_d.legend(loc="upper left", fontsize=9)

    fig.suptitle(
        "Mission overview, RT PolarFire family × reference orbital environments",
        fontsize=13,
        fontweight="bold",
        y=1.00,
    )
    plt.tight_layout()
    out_path = ASSETS / "orbit_seu_rates.png"
    plt.savefig(out_path)
    plt.close()
    print(f"  → {out_path.relative_to(ROOT)}", flush=True)


def main() -> None:
    t_start = time.perf_counter()
    print(f"Loading {MODEL_ID}…", flush=True)
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).eval()
    print(f"  loaded in {time.perf_counter() - t0:.1f} s", flush=True)

    chart_bit_position_attack_surface(base, tok)
    chart_protection_efficacy(base, tok)
    chart_orbit_rates()

    print(
        f"\nAll charts saved to docs/assets/ (total {time.perf_counter() - t_start:.0f}s).",
        flush=True,
    )


if __name__ == "__main__":
    main()
