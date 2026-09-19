"""Self-contained HTML validation reports (inline CSS only, no JavaScript).

The rendered page is deterministic: equity curves are drawn as static inline
SVG paths and every piece of textual data is HTML-escaped, so the output can
be archived or diffed without a browser.
"""

from __future__ import annotations

import html
import math
from typing import Any

from quantlab.reports.json import (
    _DSR_HARD,
    _DSR_WEAK,
    _PBO_HARD,
    _PBO_MODERATE,
    _PSR_MARGINAL,
    _read,
    classify_validation,
)
from quantlab.validation.metrics import all_metrics

_WIDTH = 800
_HEIGHT = 300
_PAD = 32

_CSS = """\
body{font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;margin:2rem auto;max-width:860px;color:#222;background:#fff;line-height:1.5}
header{margin-bottom:1.5rem}
h1{font-size:1.6rem;margin:0 0 .25rem}
.subtitle{color:#666;margin:0 0 1rem}
dl.meta{display:flex;gap:1rem;flex-wrap:wrap;margin:0}
dt{font-weight:600}
dd{margin-left:.25rem}
.badge{display:inline-block;padding:.15rem .6rem;border-radius:999px;font-weight:700;font-size:.85rem}
section{margin-bottom:1.5rem}
table{border-collapse:collapse;width:100%;margin-top:.5rem}
th,td{border:1px solid #ddd;padding:.35rem .6rem;text-align:left;font-size:.9rem}
th{background:#f4f4f4;width:40%}
svg{display:block;border:1px solid #eee;max-width:100%}
ul{margin:.5rem 0}
footer{margin-top:2rem;color:#888;font-size:.8rem}
"""


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _fmt_num(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and not math.isfinite(value):
        return "inf" if value > 0 else "-inf"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.4f}"
    return _esc(value)


def _svg_equity_chart(equity: list[float], baseline: float) -> str:
    """Draw the equity curve as a static SVG path with an area fill."""
    if not equity:
        return ""
    lo = min(min(equity), baseline)
    hi = max(max(equity), baseline)
    span = hi - lo
    if span <= 0:
        if hi == 0:
            hi = 1.0
        span = max(abs(hi), 1e-9)
    inner_w = _WIDTH - 2 * _PAD
    inner_h = _HEIGHT - 2 * _PAD

    def y(value: float) -> float:
        return _HEIGHT - _PAD - (value - lo) / span * inner_h

    n = len(equity)
    step = inner_w / max(n - 1, 1)
    line_cmds = []
    for i, value in enumerate(equity):
        x = _PAD + (i * step if n > 1 else inner_w / 2)
        line_cmds.append(f"{x:.2f} {y(value):.2f}")
    y_baseline = y(baseline)
    x0 = _PAD
    xn = _PAD + inner_w
    line_d = "M " + " L ".join(line_cmds)
    area_d = f"M {x0:.2f} {y_baseline:.2f} L " + " L ".join(line_cmds) + f" L {xn:.2f} {y_baseline:.2f} Z"
    return (
        f'<svg viewBox="0 0 {_WIDTH} {_HEIGHT}" width="{_WIDTH}" height="{_HEIGHT}" '
        'xmlns="http://www.w3.org/2000/svg" role="img" aria-label="equity curve">'
        f'<path d="{area_d}" fill="rgba(38,108,190,0.15)" stroke="none"/>'
        f'<path d="{line_d}" fill="none" stroke="rgba(18,60,120,0.9)" stroke-width="2" stroke-linejoin="round"/>'
        f'<line x1="{_PAD}" y1="{y_baseline:.2f}" x2="{_WIDTH - _PAD}" y2="{y_baseline:.2f}" '
        'stroke="rgba(120,120,120,0.8)" stroke-width="1" stroke-dasharray="4,4"/>'
        f'<text x="{_PAD}" y="{y_baseline - 6:.2f}" font-size="12" fill="rgba(60,60,60,0.9)">initial equity</text>'
        "</svg>"
    )


def _metrics_table(returns) -> str:
    rows = "".join(
        f"<tr><th>{_esc(name)}</th><td>{_fmt_num(value)}</td></tr>"
        for name, value in all_metrics(list(returns)).items()
    )
    return f"<table>{rows}</table>"


def _metric_row(label: str, value: Any, note: str) -> str:
    return (
        f"<tr><th>{_esc(label)}</th><td>{_fmt_num(value)}</td>"
        f"<td>{_esc(note)}</td></tr>"
    )


def _validation_section(validation: dict | None) -> str:
    if not validation:
        return "<p>No validation battery was run for this experiment.</p>"
    rows = []
    psr = _read(validation, "psr")
    if psr is not None:
        note = (
            "Evidence of skill over the benchmark (>= 0.5)."
            if psr >= _PSR_MARGINAL
            else "Below the 0.5 hurdle: does not yet clear the benchmark."
        )
        rows.append(_metric_row("Probabilistic Sharpe Ratio (benchmark 0)", psr, note))
    dsr = _read(validation, "dsr")
    if dsr is not None:
        if dsr >= _DSR_WEAK:
            note = "Survives the deflation hurdle (>= 0.5)."
        elif dsr >= _DSR_HARD:
            note = "Weak deflated Sharpe: selection bias is not ruled out."
        else:
            note = "Fails the deflation hurdle (< 0.25): corrected SR likely not positive."
        rows.append(_metric_row("Deflated Sharpe Ratio", dsr, note))
    pbo = _read(validation, "pbo")
    if pbo is not None:
        if pbo <= _PBO_MODERATE:
            note = "Low overfit risk (<= 0.2)."
        elif pbo <= _PBO_HARD:
            note = "Moderate overfit risk (0.2-0.5)."
        else:
            note = "Elevated overfit risk (> 0.5): IS winner tends to invert OOS."
        rows.append(_metric_row("Probability of Backtest Overfitting", pbo, note))
    bootstrap = validation.get("bootstrap")
    ci = bootstrap.get("ci95") if isinstance(bootstrap, dict) else validation.get("ci95")
    if isinstance(ci, dict):
        lo, hi, mean = ci.get("lo"), ci.get("hi"), ci.get("mean")
        note = (
            "Confidence interval excludes zero."
            if isinstance(lo, (int, float)) and isinstance(hi, (int, float)) and not (lo <= 0 <= hi)
            else "Confidence interval spans zero."
        )
        label = f"Bootstrap Sharpe 95% CI ({_esc(ci.get('method', 'iid'))})"
        if all(isinstance(v, (int, float)) for v in (lo, hi, mean)):
            value = f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"
        else:
            value = _esc(f"{mean} [{lo}, {hi}]")
        rows.append(f"<tr><th>{_esc(label)}</th><td>{value}</td><td>{_esc(note)}</td></tr>")
    if not rows:
        return "<p>Validation record present but no scalar statistics to render.</p>"
    return f"<table>{''.join(rows)}</table>"


def _concerns_section(concerns: list[str]) -> str:
    items = "".join(f"<li>{_esc(concern)}</li>" for concern in concerns)
    body = f"<ul>{items}</ul>" if items else "<p>None recorded.</p>"
    return f"<section><h2>Concerns</h2>{body}</section>"


def render_html_report(
    result,
    validation: dict | None = None,
    experiment_id: str = "",
    title: str = "Quant Lab Validation Report",
) -> str:
    """Render a fully self-contained HTML validation report (no JS)."""
    status, concerns = classify_validation(validation)
    bg = {
        "REJECTED": "#f6dcdc",
        "INSUFFICIENT_EVIDENCE": "#faf0d7",
        "SURVIVED_BATTERY": "#ddf0e3",
    }.get(status, "#eee")
    fg = {
        "REJECTED": "#8c1414",
        "INSUFFICIENT_EVIDENCE": "#96640a",
        "SURVIVED_BATTERY": "#146e2d",
    }.get(status, "#444")
    equity = [float(value) for value in result.equity()]
    config = getattr(result, "config", None)
    baseline = float(getattr(config, "initial_cash", equity[0] if equity else 0.0))
    header = (
        "<header>"
        "<h1>QUANT LAB VALIDATION REPORT</h1>"
        f'<p class="subtitle">{_esc(title)}</p>'
        '<dl class="meta">'
        f"<dt>Experiment ID</dt><dd>{_esc(experiment_id)}</dd>"
        f'<dt>Status</dt><dd><span class="badge" style="background:{bg};color:{fg}">{_esc(status)}</span></dd>'
        "</dl>"
        "</header>"
    )
    metrics = f"<section><h2>Performance Metrics</h2>{_metrics_table(result.returns)}</section>"
    validation_section = f"<section><h2>Validation Battery</h2>{_validation_section(validation)}</section>"
    equity_section = f"<section><h2>Equity Curve</h2>{_svg_equity_chart(equity, baseline)}</section>"
    concerns_html = _concerns_section(concerns)
    footer = "<footer>Generated with Quant Lab reports. Falsify, do not prove.</footer>"
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        f"<title>{_esc(title)}</title><style>{_CSS}</style></head><body>"
        f"{header}{metrics}{validation_section}{equity_section}{concerns_html}{footer}"
        "</body></html>"
    )


def render_graveyard_html(entries: list[dict]) -> str:
    """Render a simple table of failed experiments for the graveyard page."""
    def cell(value: Any) -> str:
        return _esc("-" if value is None else value)

    rows = "".join(
        "<tr>"
        f"<td>{cell(entry.get('id', entry.get('experiment_id', '-')))}</td>"
        f"<td>{cell(entry.get('status', '-'))}</td>"
        f"<td>{cell(entry.get('primary_failure', entry.get('failure_reason', '-')))}</td>"
        f"<td>{cell(entry.get('raw_sharpe', '-'))}</td>"
        f"<td>{cell(entry.get('net_sharpe', '-'))}</td>"
        "</tr>"
        for entry in entries
    )
    thead = (
        "<thead><tr><th>id</th><th>status</th><th>primary_failure</th>"
        "<th>raw_sharpe</th><th>net_sharpe</th></tr></thead>"
    )
    body = (
        "<h1>QUANT LAB EXPERIMENT GRAVEYARD</h1>"
        f"<table>{thead}<tbody>{rows}</tbody></table>"
    )
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<title>Quant Lab Graveyard</title>"
        f"<style>{_CSS}</style></head><body>{body}</body></html>"
    )
