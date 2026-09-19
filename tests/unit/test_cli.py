"""CLI surface tests: entry point, strategy listing, synthetic data generation."""

from __future__ import annotations

import json

import pytest
from quantlab.cli import main


def test_version_prints() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0


def test_strategies_lists_all() -> None:
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        assert main(["--strategies"]) == 0
    names = {line for line in buf.getvalue().splitlines() if line.strip()}
    assert names == {
        "breakout",
        "buy_and_hold",
        "mean_reversion",
        "mean_reversion_rsi",
        "momentum",
        "pairs",
        "random",
        "sma_crossover",
        "vol_target",
    }


def test_no_command_returns_error() -> None:
    assert main([]) == 1


def test_generate_synthetic_random_walk(tmp_path) -> None:
    rc = main(
        [
            "data",
            "generate-synthetic",
            "--out",
            str(tmp_path),
            "--process",
            "random_walk",
            "--seed",
            "3",
            "--symbols",
            "AAA",
        ]
    )
    assert rc == 0
    doc = json.loads((tmp_path / "AAA.json").read_text(encoding="utf-8"))
    assert doc["symbol"] == "AAA"
    assert doc["source"] == "synthetic"
    assert len(doc["bars"]) == 504


def test_generate_synthetic_pairs(tmp_path) -> None:
    rc = main(
        [
            "data",
            "generate-synthetic",
            "--out",
            str(tmp_path),
            "--process",
            "pairs",
            "--seed",
            "11",
            "--symbols",
            "A,B",
        ]
    )
    assert rc == 0
    assert (tmp_path / "A.json").exists()
    assert (tmp_path / "B.json").exists()


def test_generate_synthetic_pairs_requires_two_symbols(tmp_path) -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "data",
                "generate-synthetic",
                "--out",
                str(tmp_path),
                "--process",
                "pairs",
                "--seed",
                "11",
                "--symbols",
                "A",
            ]
        )


def test_generate_synthetic_unknown_process(tmp_path) -> None:
    with pytest.raises(ValueError):
        main(["data", "generate-synthetic", "--out", str(tmp_path), "--process", "nope"])


def test_audit_written_synthetic(tmp_path) -> None:
    from contextlib import redirect_stdout
    from io import StringIO

    main(["data", "generate-synthetic", "--out", str(tmp_path), "--seed", "3"])
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["data", "audit", "--file", str(tmp_path / "SYN.json")])
    assert rc == 0
    assert "issues=0 errors=0" in buf.getvalue()
