"""Self-contained HTML rendering for runs and profile reports.

No external CSS, no JavaScript. Reports are valid, semantic HTML5 with
inline styles, suitable for emailing, archiving, or hosting on a static
page. Designed to be reasonably accessible (semantic landmarks,
keyboard-navigable tables) and printable.
"""

from __future__ import annotations

import html
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spacellm.profiling.report import ProfileReport
    from spacellm.types import FaultEvent, RunRecord


_CSS = """
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', Roboto, sans-serif;
  color: #18181b;
  background: #fafafa;
  margin: 0;
  padding: 2rem 1rem;
  line-height: 1.55;
}
.report {
  max-width: 1200px;
  margin: 0 auto;
  background: white;
  border-radius: 10px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.04);
  overflow: hidden;
}
.report > header {
  padding: 2rem;
  border-bottom: 1px solid #ececec;
  background: linear-gradient(180deg, #fafafa, #fff);
}
.report > header h1 {
  font-size: 1.6rem;
  margin: 0 0 0.25rem 0;
  font-weight: 600;
  letter-spacing: -0.01em;
}
.subtitle {
  color: #71717a;
  font-size: 0.95rem;
  margin: 0;
}
dl.meta {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 0.5rem 2rem;
  margin: 1.25rem 0 0 0;
  font-size: 0.85rem;
}
dl.meta dt {
  color: #71717a;
  text-transform: uppercase;
  font-size: 0.7rem;
  letter-spacing: 0.06em;
}
dl.meta dd {
  margin: 0;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  color: #18181b;
}
section {
  padding: 1.5rem 2rem;
  border-bottom: 1px solid #ececec;
}
section:last-of-type { border-bottom: none; }
section h2 {
  font-size: 0.85rem;
  margin: 0 0 1rem 0;
  color: #52525b;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.85rem;
}
th, td {
  padding: 0.45rem 0.65rem;
  text-align: left;
  border-bottom: 1px solid #f4f4f5;
}
th {
  font-weight: 600;
  color: #3f3f46;
  background: #fafafa;
  position: sticky;
  top: 0;
  z-index: 1;
}
tbody tr:hover { background: #fafafa; }
.num { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; text-align: right; }
code {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  background: #f4f4f5;
  border-radius: 3px;
  padding: 0.05em 0.4em;
  font-size: 0.85em;
}
.tag {
  display: inline-block;
  padding: 0.05rem 0.5rem;
  border-radius: 999px;
  font-size: 0.7rem;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  background: #e0e7ff;
  color: #3730a3;
  border: 1px solid #c7d2fe;
}
.tag-warn { background: #fef3c7; color: #92400e; border-color: #fde68a; }
.tag-bad  { background: #fee2e2; color: #991b1b; border-color: #fecaca; }
.metric-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 0.75rem;
}
.metric {
  padding: 0.85rem 1rem;
  background: #fafafa;
  border-radius: 6px;
  border: 1px solid #ececec;
}
.metric-label {
  font-size: 0.7rem;
  color: #71717a;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}
.metric-value {
  font-size: 1.4rem;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  margin-top: 0.25rem;
  color: #18181b;
}
footer {
  padding: 1rem 2rem;
  font-size: 0.75rem;
  color: #a1a1aa;
  text-align: center;
}
@media print {
  body { background: white; padding: 0; }
  .report { box-shadow: none; border-radius: 0; }
  th { position: static; }
}
"""

_DOCUMENT = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{css}</style>
</head>
<body>
<main class="report">
<header>
<h1>{title}</h1>
<p class="subtitle">{subtitle}</p>
{meta}
</header>
{body}
<footer>SpaceLLM · generated {timestamp}</footer>
</main>
</body>
</html>
"""


def _esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def _meta(items: list[tuple[str, str]]) -> str:
    pairs = "".join(f"<dt>{_esc(k)}</dt><dd>{_esc(v)}</dd>" for k, v in items)
    return f'<dl class="meta">{pairs}</dl>'


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def render_profile_html(report: ProfileReport, *, title: str = "SpaceLLM Profile") -> str:
    """Render a :class:`~spacellm.profiling.report.ProfileReport` as HTML."""
    layers = sorted(report.layers, key=lambda layer: -layer.n_parameters)
    has_dynamic = any(layer.mean_perturbation is not None for layer in layers)

    headers = ["#", "Layer", "Class", "Params", "Bits"]
    if has_dynamic:
        headers += ["Mean Δ", "Max Δ"]

    rows: list[str] = []
    for i, layer in enumerate(layers, 1):
        cells = [
            f'<td class="num">{i}</td>',
            f"<td><code>{_esc(layer.layer_name or '<root>')}</code></td>",
            f'<td><span class="tag">{_esc(layer.module_class)}</span></td>',
            f'<td class="num">{layer.n_parameters:,}</td>',
            f'<td class="num">{layer.n_bits:,}</td>',
        ]
        if has_dynamic:
            cells.append(
                f'<td class="num">{layer.mean_perturbation:.4g}</td>'
                if layer.mean_perturbation is not None
                else "<td>,</td>",
            )
            cells.append(
                f'<td class="num">{layer.max_perturbation:.4g}</td>'
                if layer.max_perturbation is not None
                else "<td>,</td>",
            )
        rows.append(f"<tr>{''.join(cells)}</tr>")

    head_html = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    table = f"<table><thead><tr>{head_html}</tr></thead><tbody>{''.join(rows)}</tbody></table>"

    body = f"<section><h2>Layers ({len(layers)})</h2>{table}</section>"

    return _DOCUMENT.format(
        title=_esc(title),
        css=_CSS,
        subtitle=_esc(f"{len(layers)} modules · {'dynamic' if has_dynamic else 'static'} mode"),
        meta=_meta(
            [
                ("layers", str(len(layers))),
                ("mode", "dynamic" if has_dynamic else "static"),
                (
                    "total_params",
                    f"{sum(layer.n_parameters for layer in layers):,}",
                ),
            ]
        ),
        body=body,
        timestamp=_esc(_now_iso()),
    )


def render_run_html(
    record: RunRecord,
    events: list[FaultEvent],
    *,
    title: str | None = None,
    event_cap: int = 500,
) -> str:
    """Render a :class:`~spacellm.types.RunRecord` (with fault events) as HTML."""
    resolved_title = title or f"Run {record.run_id[:8]}"

    duration: str = ","
    if record.ended_at is not None:
        duration = f"{(record.ended_at - record.started_at).total_seconds():.2f}s"

    sections: list[str] = []

    if record.metrics:
        cards = "".join(
            f'<div class="metric"><div class="metric-label">{_esc(k)}</div>'
            f'<div class="metric-value">{v:.4g}</div></div>'
            for k, v in record.metrics.items()
        )
        sections.append(
            f'<section><h2>Metrics</h2><div class="metric-grid">{cards}</div></section>',
        )

    sections.append(
        '<section><h2>Faults</h2><div class="metric-grid">'
        f'<div class="metric"><div class="metric-label">Injected</div>'
        f'<div class="metric-value">{record.n_faults_injected:,}</div></div>'
        f'<div class="metric"><div class="metric-label">Recovered</div>'
        f'<div class="metric-value">{record.n_faults_recovered:,}</div></div>'
        "</div></section>",
    )

    if events:
        shown = events[:event_cap]
        rows = "".join(
            "<tr>"
            f'<td class="num">{e.t:.4f}</td>'
            f"<td><code>{_esc(e.tensor_id)}</code></td>"
            f'<td class="num">{e.bit_index}</td>'
            f'<td class="num">{e.bit_position}</td>'
            f'<td class="num">0x{e.before:x}</td>'
            f'<td class="num">0x{e.after:x}</td>'
            f'<td><span class="tag">{_esc(e.cause.value)}</span></td>'
            "</tr>"
            for e in shown
        )
        truncated = (
            f"<p class='subtitle' style='margin-top:0.75rem'>"
            f"Showing first {event_cap:,} of {len(events):,} events.</p>"
            if len(events) > event_cap
            else ""
        )
        sections.append(
            "<section><h2>Events</h2>"
            "<table><thead><tr>"
            "<th>t (s)</th><th>tensor</th><th>bit_index</th><th>bit_pos</th>"
            "<th>before</th><th>after</th><th>cause</th>"
            "</tr></thead>"
            f"<tbody>{rows}</tbody></table>"
            f"{truncated}"
            "</section>",
        )

    return _DOCUMENT.format(
        title=_esc(resolved_title),
        css=_CSS,
        subtitle=_esc(record.name or ","),
        meta=_meta(
            [
                ("id", record.run_id),
                ("model", record.model_name or ","),
                ("started", record.started_at.isoformat(timespec="seconds")),
                ("duration", duration),
            ],
        ),
        body="".join(sections),
        timestamp=_esc(_now_iso()),
    )


__all__ = ["render_profile_html", "render_run_html"]
