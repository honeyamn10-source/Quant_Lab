"""Unit tests for the strategy graveyard package."""

from __future__ import annotations

import pytest
from quantlab.graveyard.postmortem import (
    classify_failure,
    generate_postmortem,
    postmortem_from_evidence,
)
from quantlab.graveyard.recorder import Graveyard
from quantlab.graveyard.taxonomy import FAILURE_DESCRIPTIONS, FailureCategory, verdict_for


def test_taxonomy_has_all_13_members() -> None:
    expected = {
        "OVERFIT",
        "LOOKAHEAD",
        "SURVIVORSHIP",
        "SLIPPAGE",
        "MARKET_IMPACT",
        "REGIME_DEPENDENT",
        "PARAMETER_UNSTABLE",
        "DATA_ERROR",
        "CORRELATION_FAILURE",
        "TAIL_RISK",
        "INSUFFICIENT_SAMPLE",
        "ALPHA_DECAY",
        "NO_NET_EDGE",
    }
    members = {category.name for category in FailureCategory}
    assert len(members) == 13
    assert members == expected
    assert set(FAILURE_DESCRIPTIONS) == set(FailureCategory)


def test_verdict_for_insufficient_sample_is_inconclusive() -> None:
    assert verdict_for(FailureCategory.INSUFFICIENT_SAMPLE) == "INCONCLUSIVE"
    for category in FailureCategory:
        expected = "INCONCLUSIVE" if category is FailureCategory.INSUFFICIENT_SAMPLE else "REJECTED"
        assert verdict_for(category) == expected


def test_record_entry_assigns_increasing_ids_and_persists(tmp_path) -> None:
    graveyard = Graveyard(tmp_path / "graveyard")
    first = graveyard.record_entry(
        {"hypothesis": "momentum", "symbols": ["AAPL"], "status": "REJECTED"}
    )
    second = graveyard.record_entry({"hypothesis": "mean_reversion", "status": "REJECTED"})

    assert first["id"] == "STRATEGY-00001"
    assert second["id"] == "STRATEGY-00002"
    assert int(first["id"].split("-")[1]) < int(second["id"].split("-")[1])
    assert first["created_iso"]
    assert graveyard.count() == 2
    assert graveyard.get_entry("STRATEGY-00002")["hypothesis"] == "mean_reversion"
    assert graveyard.list_entries() == [first, second]
    with pytest.raises(KeyError):
        graveyard.get_entry("STRATEGY-99999")

    reloaded = Graveyard(tmp_path / "graveyard")
    third = reloaded.record_entry({"hypothesis": "breakout", "status": "INCONCLUSIVE"})
    assert third["id"] == "STRATEGY-00003"
    assert [entry["id"] for entry in reloaded.list_entries()] == [
        "STRATEGY-00001",
        "STRATEGY-00002",
        "STRATEGY-00003",
    ]


def test_search_filters(tmp_path) -> None:
    graveyard = Graveyard(tmp_path / "graveyard")
    graveyard.record_entry(
        {
            "hypothesis": "a",
            "status": "REJECTED",
            "failure_categories": ["overfit"],
            "symbols": ["AAPL"],
        }
    )
    graveyard.record_entry({"hypothesis": "b", "status": "INCONCLUSIVE", "symbols": ["TSLA"]})

    assert len(graveyard.search(status="REJECTED")) == 1
    assert len(graveyard.search(category="overfit")) == 1
    assert len(graveyard.search(symbols=["AAPL"])) == 1
    assert len(graveyard.search(status="REJECTED", category="overfit", symbols=["AAPL"])) == 1
    assert not graveyard.search(status="REJECTED", symbols=["TSLA"])


def test_generate_postmortem_exact_header_lines() -> None:
    record = {
        "id": "STRATEGY-00482",
        "hypothesis": "Weekend drift persists ex costs",
        "raw_sharpe": 1.1,
        "net_sharpe": 0.03,
        "primary_failure": "Net edge lost to costs",
        "secondary_failure": "Regime-dependent",
        "pbo": 0.6,
        "walk_forward": "Failed out of sample",
        "status": "REJECTED",
    }
    text = generate_postmortem(record)
    lines = text.splitlines()

    assert lines[0] == "STRATEGY-00482"
    assert "Hypothesis: Weekend drift persists ex costs" in lines
    assert "Raw Sharpe: 1.100" in lines
    assert "Realistic-cost Sharpe: 0.030" in lines
    assert "Main failure: Net edge lost to costs" in lines
    assert "Secondary failure: Regime-dependent" in lines
    assert "PBO: 0.600" in lines
    assert "Walk-forward: Failed out of sample" in lines
    assert lines[-1] == "Status: REJECTED."


def test_generate_postmortem_tolerates_missing_keys() -> None:
    text = generate_postmortem({})
    lines = text.splitlines()

    assert lines[0] == "STRATEGY-UNKNOWN"
    assert "Hypothesis: Unknown" in lines
    assert lines[-1] == "Status: REJECTED."


def test_classify_failure_high_pbo_low_dsr_is_overfit() -> None:
    assert classify_failure({"pbo": 0.8, "dsr": 0.2}) == [FailureCategory.OVERFIT]
    assert classify_failure({"pbo": 0.6}) == [FailureCategory.OVERFIT]
    assert classify_failure({"dsr": 0.1}) == [FailureCategory.OVERFIT]


def test_classify_failure_zero_net_edge_with_gross_edge() -> None:
    evidence = {"net_sharpe": 0.0, "gross_sharpe": 1.2}
    assert classify_failure(evidence) == [FailureCategory.NO_NET_EDGE]


def test_classify_failure_returns_multiple_categories() -> None:
    evidence = {
        "dsr": 0.3,
        "net_sharpe": 0.02,
        "gross_sharpe": 1.0,
        "regime_dependency": True,
        "n_obs": 100,
    }
    classified = classify_failure(evidence)
    assert FailureCategory.OVERFIT in classified
    assert FailureCategory.NO_NET_EDGE in classified
    assert FailureCategory.REGIME_DEPENDENT in classified
    assert FailureCategory.INSUFFICIENT_SAMPLE in classified


def test_postmortem_from_evidence_builds_standard_entry() -> None:
    entry = postmortem_from_evidence(
        {"id": "EXP-1", "hypothesis": "h", "gross_sharpe": 0.9, "net_sharpe": 0.05},
        {"dsr": 0.9, "pbo": 0.1, "n_obs": 500},
        [("no_net_edge", "edge lost to costs")],
    )

    assert entry["id"] == "EXP-1"
    assert entry["status"] == "REJECTED"
    assert FailureCategory.NO_NET_EDGE in entry["failure_categories"]
    assert entry["primary_failure"] == "edge lost to costs"
    assert entry["secondary_failure"] is None
    assert entry["validation"] == {"dsr": 0.9, "pbo": 0.1, "n_obs": 500}
    assert entry["raw_sharpe"] is None
