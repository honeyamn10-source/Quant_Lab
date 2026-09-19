"""Professional CLI for Quant Lab.

Surface:
    quantlab data audit <file.json>
    quantlab data generate-synthetic [--out DIR] [--process NAME] [--seed N]
    quantlab run-experiment <config.yaml> [--report-dir DIR]
    quantlab backtest <config.yaml>
    quantlab validate <results.json>
    quantlab stress <experiment-id>
    quantlab regimes <file.json> [--symbol SYM]
    quantlab volatility <file.json>
    quantlab report <experiment-id>
    quantlab graveyard list
    quantlab reproduce <experiment-id>
    quantlab sizing --simulate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import quantlab
from quantlab.backtest.engine import BacktestConfig, BacktestEngine
from quantlab.data.schemas import Bar, BarSeries
from quantlab.data.validation import audit_bars
from quantlab.experiments.config import load_run_config
from quantlab.experiments.registry import ExperimentLedger, default_ledger, scaffold_default
from quantlab.strategies.benchmark import create_strategy, list_strategies
from quantlab.validation import metrics
from quantlab.validation.bootstrap import bootstrap_sharpe
from quantlab.validation.dsr import deflated_sharpe_ratio
from quantlab.validation.psr import probabilistic_sharpe_ratio


# ---------------------------------------------------------------- data I/O --
def _save_series_json(series: BarSeries, path: Path) -> None:
    doc = {
        "symbol": series.symbol,
        "interval_seconds": series.interval_seconds,
        "bars": [
            {
                "ts": b.ts,
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume,
                "available_time": b.available_time,
            }
            for b in series.bars
        ],
        "source": "synthetic",
    }
    path.write_text(json.dumps(doc), encoding="utf-8")


def _load_series_json(path: Path) -> BarSeries:
    doc = json.loads(path.read_text(encoding="utf-8"))
    bars = [
        Bar(
            symbol=doc["symbol"],
            ts=b["ts"],
            interval_seconds=b.get("interval_seconds", doc.get("interval_seconds", 86400)),
            open=b["open"],
            high=b["high"],
            low=b["low"],
            close=b["close"],
            volume=b.get("volume", 0.0),
            available_time=b.get("available_time"),
            source=doc.get("source", "file"),
        )
        for b in doc["bars"]
    ]
    return BarSeries(doc["symbol"], bars)


def _series_from_rows_file(path: Path, symbol: str) -> BarSeries:
    doc = json.loads(path.read_text(encoding="utf-8"))
    bars = [
        Bar(symbol=symbol, ts=r["ts"], interval_seconds=r.get("interval_seconds", 86400), open=r["open"],
            high=r["high"], low=r["low"], close=r["close"], volume=r.get("volume", 0.0),
            available_time=r.get("available_time"), source=r.get("source", "file"))
        for r in doc
    ]
    return BarSeries(symbol, bars)


def _run_validation(results: dict) -> dict:
    returns = np.asarray(results.get("returns") or [], dtype=float)
    if len(returns) < 20:
        return {"note": "insufficient returns for validation battery", "metrics": {}}
    met = metrics.all_metrics(returns)
    psr = probabilistic_sharpe_ratio(returns, benchmark_sr=0.0)
    dsr = deflated_sharpe_ratio(returns, n_trials=1)
    boot = bootstrap_sharpe(returns, n_iter=300, seed=0)
    return {
        "metrics": met,
        "psr": psr,
        "dsr": dsr,
        "bootstrap95": boot["ci95"],
    }


def _build_from_config(cfg) -> tuple[dict[str, BarSeries], list[str] | None]:
    """Materialize the RunConfig's data into streams plus optional mapping rows."""
    from quantlab.synthetic.markets import generate, generate_pairs

    streams: dict[str, BarSeries] = {}
    symbols = cfg.data.symbols
    if cfg.data.process == "pairs":
        if len(symbols) != 2:
            raise ValueError("pairs process requires exactly two symbols")
        a, b = generate_pairs(n=int(cfg.data.n_bars), seed=int(cfg.data.seed), symbols=(symbols[0], symbols[1]))
        streams = {a.symbol: a, b.symbol: b}
        return streams, None
    for sym in symbols:
        streams[sym] = generate(cfg.data.process, n=int(cfg.data.n_bars), seed=int(cfg.data.seed), symbol=sym)
    return streams, None


def _backtest_cfg(cfg) -> BacktestConfig:
    return BacktestConfig(**cfg.backtest_config)


def cmd_run_experiment(args) -> int:
    scaffold_default()
    cfg = load_run_config(args.config)
    streams, _ = _build_from_config(cfg)
    strategy = create_strategy(cfg.strategy, cfg.strategy_params)
    result = BacktestEngine(_backtest_cfg(cfg)).run(strategy, streams)
    results_json = result.to_dict()
    validation = _run_validation(results_json)

    ledger: ExperimentLedger = args.ledger if hasattr(args, "ledger") else default_ledger()
    record = ledger.create(run_name=cfg.name, strategy_name=cfg.strategy, run_config=cfg)
    final = result.final_equity()
    results_payload = {
        "results": results_json,
        "validation": validation,
        "final_equity": final,
        "total_return": final / cfg.initial_cash - 1.0,
        "rejected_orders": len(result.rejects),
    }
    ledger.complete(record["experiment_id"], results_payload)
    out_dir = Path(getattr(args, "report_dir", None) or "experiments") / record["experiment_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results_payload, indent=2), encoding="utf-8")
    print(f"[{record['experiment_id']}] {cfg.strategy} on {cfg.data.process} -> "
          f"final ${final:,.0f} | sharpe {validation['metrics'].get('sharpe', 0):.2f}")
    return 0


def cmd_backtest(args) -> int:
    cfg = load_run_config(args.config)
    streams, _ = _build_from_config(cfg)
    result = BacktestEngine(_backtest_cfg(cfg)).run(create_strategy(cfg.strategy, cfg.strategy_params), streams)
    met = metrics.all_metrics(result.returns)
    print(f"strategy={cfg.strategy} final=${result.final_equity():,.0f}")
    for k, v in met.items():
        print(f"  {k}: {v:.4f}")
    print(f"  fills={len(result.fills)} rejects={len(result.rejects)} signals={len(result.signals)}")
    return 0


def cmd_validate(args) -> int:
    payload = json.loads(Path(args.results).read_text(encoding="utf-8"))
    results = payload.get("results") or payload
    validation = _run_validation(results)
    print(json.dumps(validation, indent=2, default=float))
    return 0


def cmd_stress(args) -> int:
    from quantlab.adversarial.perturbation import (
        cost_perturbation_sweep,
        execution_delay_sweep,
        param_perturbation_sweep,
    )
    from quantlab.adversarial.robustness import robustness_surface

    ledger = args.ledger if hasattr(args, "ledger") else default_ledger()
    record = ledger.get(args.experiment_id)
    run_cfg = record.get("run_config") or {}
    cfg = _record_to_run_config(run_cfg)

    def run_with(overrides: dict):
        bcfg = cfg.backtest_config.copy()
        bcfg.update({k: v for k, v in overrides.items() if k in bcfg})
        params = dict(cfg.strategy_params)
        if "strategy_params" in overrides:
            params.update(overrides["strategy_params"])
        streams, _ = _build_from_config(cfg)
        res = BacktestEngine(BacktestConfig(**bcfg)).run(create_strategy(cfg.strategy, params), streams)
        return type("R", (), {"returns": res.returns, "signals": res.signals, "equity": res.equity})()

    sweeps = {
        "params": param_perturbation_sweep(run_with, cfg.strategy_params),
        "cost": cost_perturbation_sweep(run_with),
        "delay": execution_delay_sweep(run_with),
    }
    surface = robustness_surface(sweeps)
    print(f"robustness rating: {surface['rating']}")
    print(f"worst: {surface['worst']}")
    return 0


def _record_to_run_config(run_config: dict):
    from quantlab.experiments.config import RunConfig
    return RunConfig(**run_config)


def cmd_data(args) -> int:
    from quantlab.synthetic.markets import generate

    if args.action == "generate-synthetic":
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        symbols = args.symbols.split(",") if args.symbols else ["SYN"]
        for i, sym in enumerate(symbols):
            series = generate(args.process, n=int(args.n), seed=int(args.seed) + i, symbol=sym)
            _save_series_json(series, out / f"{sym}.json")
            print(f"wrote {out / (sym + '.json')}: {len(series)} bars")
        return 0
    if args.action in ("audit", "quality"):
        series = _load_series_json(Path(args.file))
        audit = audit_bars(series)
        print(f"audit: n={audit.n_bars} issues={len(audit.issues)} errors={len(audit.errors)}")
        for issue in audit.issues[:20]:
            print(f"  {issue.severity.value:7s} {issue.code} {issue.symbol}: {issue.message}")
        if args.action == "quality":
            from quantlab.data.quality import score_quality

            score = score_quality(audit, survivorship_hint=args.survivorship)
            print(f"quality score: {score.score}/100")
            for k, v in score.risks.items():
                print(f"  {k}: {v}")
        return 0
    raise SystemExit(f"unknown data action {args.action}")


def cmd_regimes(args) -> int:
    from quantlab.regimes.features import regime_features
    from quantlab.regimes.models import ensemble_regimes, rule_based_regimes

    series = _load_series_json(Path(args.file))
    closes = np.asarray(series.closes(), dtype=float)
    returns = np.diff(closes) / closes[:-1]
    feats = regime_features(closes)
    labels, names = rule_based_regimes(feats)
    print("rule-based regimes:")
    for idx, name in enumerate(names):
        print(f"  {name}: {(labels == idx).mean():.1%}")
    ensemble = ensemble_regimes(returns, feats, seed=args.seed)
    print(f"consensus: {ensemble['consensus']:.1%}")
    return 0


def cmd_volatility(args) -> int:
    from quantlab.volatility.evaluate import benchmark_models

    series = _load_series_json(Path(args.file))
    closes = np.asarray(series.closes(), dtype=float)
    returns = np.diff(closes) / closes[:-1]
    report = benchmark_models(returns)
    print(f"winner: {report['winner']}")
    for name, m in report["models"].items():
        print(f"  {name:10s} rmse={m.get('rmse', float('nan')):.5f} qlike={m.get('qlike', float('nan')):.4f}")
    return 0


def cmd_report(args) -> int:
    from quantlab.reports.html import render_html_report

    ledger = args.ledger if hasattr(args, "ledger") else default_ledger()
    record = ledger.get(args.experiment_id)
    payload = (record.get("results") or {})
    results = payload.get("results", payload)
    validation = payload.get("validation", {})
    html = render_html_report(type("R", (), {"returns": results.get("returns", []),
                                             "equity_curve": results.get("equity_curve", []),
                                             "signals": results.get("signals", []),
                                             "fills": results.get("fills", []),
                                             "orders": results.get("orders", []),
                                             "rejects": results.get("rejects", []),
                                             "config": results.get("config", {})})(),
                              validation=validation, experiment_id=args.experiment_id)
    out = Path("experiments") / args.experiment_id / "report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}")
    return 0


def cmd_graveyard(args) -> int:
    from quantlab.graveyard.recorder import Graveyard

    gy = Graveyard()
    if args.action == "list":
        for e in gy.list_entries()[-20:]:
            print(f"{e.get('id')} {e.get('status', ''):12s} {e.get('primary_failure', ''):28s} raw={e.get('raw_sharpe', '?')} net={e.get('net_sharpe', '?')}")
        print(f"(total entries: {gy.count()})")
        return 0
    raise SystemExit(f"unknown graveyard action {args.action}")


def cmd_reproduce(args) -> int:
    from quantlab.experiments.reproducibility import reproduce

    ledger = args.ledger if hasattr(args, "ledger") else default_ledger()
    record = ledger.get(args.experiment_id)

    def run_fn(rec):
        cfg = _record_to_run_config(rec.get("run_config") or {})
        streams, _ = _build_from_config(cfg)
        res = BacktestEngine(_backtest_cfg(cfg)).run(create_strategy(cfg.strategy, cfg.strategy_params), streams)
        return {"final_equity": res.final_equity(), "sharpe": metrics.sharpe_ratio(res.returns)}

    out = reproduce(record, run_fn, ledger)
    print(json.dumps({k: v for k, v in out.items() if k != "result"}, indent=2))
    print("result:", out.get("result"))
    return 0


def cmd_sizing(args) -> int:
    from quantlab.sizing.kelly import fractional_kelly, kelly_fraction
    print("binary Kelly for p=.55,b=1:", kelly_fraction(0.55, 1.0))
    print("half Kelly:", fractional_kelly(kelly_fraction(0.55, 1.0), 0.5))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quantlab", description=quantlab.TAGLINE)
    parser.add_argument("--version", action="version", version=f"quantlab {quantlab.__version__}")
    sub = parser.add_subparsers(dest="command")

    d = sub.add_parser("data", help="data integrity actions")
    d.add_argument("action", choices=["generate-synthetic", "audit", "quality"])
    d.add_argument("--file", default="")
    d.add_argument("--out", default="datasets")
    d.add_argument("--process", default="random_walk")
    d.add_argument("--n", default=504)
    d.add_argument("--seed", type=int, default=7)
    d.add_argument("--symbols", default="")
    d.add_argument("--survivorship", default="unknown")
    d.set_defaults(func=cmd_data)

    b = sub.add_parser("backtest", help="run a config and print metrics")
    b.add_argument("config")
    b.set_defaults(func=cmd_backtest)

    re_ = sub.add_parser("run-experiment", help="backtest + validation + ledger + report")
    re_.add_argument("config")
    re_.add_argument("--report-dir", default="")
    re_.add_argument("--ledger", default="")
    re_.set_defaults(func=cmd_run_experiment)

    v = sub.add_parser("validate", help="run validation battery on results.json")
    v.add_argument("results")
    v.set_defaults(func=cmd_validate)

    s = sub.add_parser("stress", help="adversarial stress of an experiment")
    s.add_argument("experiment_id")
    s.set_defaults(func=cmd_stress)

    rg = sub.add_parser("regimes", help="regime decomposition of a series file")
    rg.add_argument("file")
    rg.add_argument("--seed", type=int, default=0)
    rg.set_defaults(func=cmd_regimes)

    vo = sub.add_parser("volatility", help="benchmark volatility models on a series file")
    vo.add_argument("file")
    vo.set_defaults(func=cmd_volatility)

    rp = sub.add_parser("report", help="generate HTML report for an experiment")
    rp.add_argument("experiment_id")
    rp.set_defaults(func=cmd_report)

    gy = sub.add_parser("graveyard", help="strategy graveyard")
    gy.add_argument("action", choices=["list"])
    gy.set_defaults(func=cmd_graveyard)

    rd = sub.add_parser("reproduce", help="re-run an experiment from its ledger record")
    rd.add_argument("experiment_id")
    rd.add_argument("--ledger", default="")
    rd.set_defaults(func=cmd_reproduce)

    sz = sub.add_parser("sizing", help="position sizing quick checks")
    sz.add_argument("--simulate", action="store_true")
    sz.set_defaults(func=cmd_sizing)

    parser.add_argument("--strategies", action="store_true", help="list available strategies")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "strategies", False):
        print("\n".join(list_strategies()))
        return 0
    if not getattr(args, "func", None):
        parser.print_help()
        return 1
    ledger_root = vars(args).get("ledger")
    if ledger_root:
        args.ledger = ExperimentLedger(ledger_root)
    elif "ledger" in vars(args):
        args.ledger = default_ledger()
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
