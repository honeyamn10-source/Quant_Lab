"""Report renderers: JSON payloads and self-contained HTML pages."""

from __future__ import annotations

from quantlab.reports.html import render_graveyard_html, render_html_report
from quantlab.reports.json import render_json_report, render_validation_report

__all__ = [
    "render_json_report",
    "render_validation_report",
    "render_html_report",
    "render_graveyard_html",
]
