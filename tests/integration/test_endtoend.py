"""End-to-end CLI integration: run-experiment, backtest, validate, reproduce."""

from __future__ import annotations

import json

from quantlab.cli import main
from quantlab.experiments.config import RunConfig, dump_run_config


def _make_config(tmp_path, name: str = "it", process: str = "random_walk", n_bars: int = 60) -> str:
    cfg = RunConfig(
        name=name,
        strategy="buy_and_hold",
        strategy_params={},
        data={"symbols": ["SYN"], "process": process, "n_bars": n_bars, "seed": 3},
        execution_delay=1,
        profile="optimistic",
    )
    p = tmp_path / f"{name}.yaml"
    dump_run_config(cfg, p)
    return str(p)


def test_run_experiment_end_to_end(tmp_path) -> None:
    cfg_path = _make_config(tmp_path)
    ledger_dir = tmp_path / "ledger"
    report_dir = tmp_path / "exp"
    rc = main(
        [
            "run-experiment",
            cfg_path,
            "--ledger",
            str(ledger_dir),
            "--report-dir",
            str(report_dir),
        ]
    )
    assert rc == 0
    ledger_file = ledger_dir / "ledger.jsonl"
    assert ledger_file.exists()
    lines = ledger_file.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    assert record["experiment_id"] == "EXP-0001"
    assert record["status"] == "completed"
    assert "sharpe" in record["results"]["validation"]["metrics"]
    out_dir = report_dir / "EXP-0001"
    assert (out_dir / "results.json").exists()


def test_backtest_command_reports_numbers(tmp_path, capsys) -> None:
    cfg_path = _make_config(tmp_path, process="trend", n_bars=100)
    rc = main(["backtest", cfg_path])
    out = capsys.readouterr().out
    assert rc == 0
    assert "final=" in out
    assert "sharpe" in out


def test_validate_command_on_results_file(tmp_path, capsys) -> None:
    cfg_path = _make_config(tmp_path, n_bars=80)
    assert main(["backtest", cfg_path]) == 0
    # write a fake results payload
    res = tmp_path / "res.json"
    res.write_text(json.dumps({"returns": [0.001] * 60, "metrics": {}}), encoding="utf-8")
    rc = main(["validate", str(res)])
    assert rc == 0
    assert "metrics" in capsys.readouterr().out


def test_reproduce_command(tmp_path, capsys) -> None:
    cfg_path = _make_config(tmp_path)
    ledger = tmp_path / "ledger"
    rc = main(["run-experiment", cfg_path, "--ledger", str(ledger)])
    assert rc == 0
    rc = main(["reproduce", "EXP-0001", "--ledger", str(ledger)])
    assert rc == 0
    assert "result" in capsys.readouterr().out


def test_data_generate_and_audit_cli(tmp_path) -> None:
    out = tmp_path / "data"
    rc = main(
        [
            "data",
            "generate-synthetic",
            "--out",
            str(out),
            "--process",
            "random_walk",
            "--seed",
            "5",
            "--symbols",
            "SYN",
        ]
    )
    assert rc == 0
    f = out / "SYN.json"
    assert f.exists()
    rc = main(["data", "quality", "--file", str(f)])
    assert rc == 0
    rc = main(["data", "audit", "--file", str(f)])
    assert rc == 0


def test_cli_version_and_strategies(capsys) -> None:
    import quantlab

    try:
        main(["--version"])
    except SystemExit as exc:
        assert exc.code == 0
    captured = capsys.readouterr()
    assert quantlab.__version__ in captured.out + captured.err
    rc = main(["--strategies"])
    assert rc == 0
    assert "sma_crossover" in capsys.readouterr().out
