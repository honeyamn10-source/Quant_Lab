"""FastAPI research API.

Serves read-only views of the experiment ledger, dataset audits and reports.
Intended for loopback / trusted networks; add authentication and host binding
before exposing publicly (see SECURITY.md). The heavy scientific modules stay
CLI-first; the dashboard is a reporting surface, not the science.
"""

from __future__ import annotations

from typing import Any

from quantlab.experiments.registry import ExperimentLedger

ledger = ExperimentLedger()


def _lightweight() -> bool:
    try:
        import fastapi  # type: ignore[import-not-found]  # noqa: F401

        return True
    except ImportError:
        return False


app: Any = None
if _lightweight():
    from fastapi import FastAPI, HTTPException  # type: ignore[import-not-found]
    from fastapi.responses import HTMLResponse, JSONResponse  # type: ignore[import-not-found]

    app = FastAPI(title="Quant Lab Research API", version="0.1.0", description=__doc__)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "ledger_entries": ledger.count()}

    @app.get("/experiments")
    def experiments() -> list[dict]:
        return ledger.all()

    @app.get("/experiments/{experiment_id}")
    def experiment(experiment_id: str) -> dict:
        try:
            return ledger.get(experiment_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/experiments/{experiment_id}/report")
    def experiment_report(experiment_id: str) -> HTMLResponse:
        try:
            record = ledger.get(experiment_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        from quantlab.reports.html import render_html_report

        payload = record.get("results") or {}
        results = payload.get("results", payload)
        html = render_html_report(
            type(
                "R",
                (),
                {
                    "returns": results.get("returns", []),
                    "equity_curve": results.get("equity_curve", []),
                    "signals": results.get("signals", []),
                    "fills": results.get("fills", []),
                    "orders": results.get("orders", []),
                    "rejects": results.get("rejects", []),
                    "config": results.get("config", {}),
                },
            )(),
            validation=payload.get("validation", {}),
            experiment_id=experiment_id,
        )
        return HTMLResponse(content=html)

    @app.get("/experiments/{experiment_id}/results")
    def experiment_results(experiment_id: str) -> JSONResponse:
        try:
            record = ledger.get(experiment_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse(content=(record.get("results") or {}))

    @app.get("/graveyard")
    def graveyard() -> list[dict]:
        from quantlab.graveyard.recorder import Graveyard

        return Graveyard().list_entries()

    @app.get("/overview")
    def overview() -> dict:
        """Dashboard summary stub: counts + status breakdown."""
        entries = ledger.all()
        statuses: dict[str, int] = {}
        for e in entries:
            statuses[e.get("status", "created")] = statuses.get(e.get("status", "created"), 0) + 1
        return {
            "n_experiments": len(entries),
            "status_breakdown": statuses,
            "strategies": [e.get("strategy_name") for e in entries[-10:]],
        }
else:  # pragma: no cover
    app = None

    def get_app():
        raise RuntimeError("FastAPI is not installed; install with: pip install 'quantlab[web]'")


def dashboard_app(is_dashboard: Any = None) -> Any:
    """Resolve the FastAPI app, raising a helpful error when FastAPI is missing."""
    if app is None:
        raise RuntimeError("fastapi not installed; run: pip install 'quantlab[web]'")
    return app
