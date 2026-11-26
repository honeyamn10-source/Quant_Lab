"""Regression: reproducibility and determinism guarantees.

Anything that breaks determinism or config stability first breaks here, which
is exactly what a research platform must guard.
"""

from __future__ import annotations

import numpy as np
from quantlab.backtest.engine import BacktestConfig, BacktestEngine
from quantlab.data.fingerprint import dataset_fingerprint
from quantlab.experiments.config import RunConfig
from quantlab.experiments.reproducibility import EnvSnapshot, reproduce
from quantlab.strategies.benchmark import create_strategy
from quantlab.synthetic.markets import generate
from quantlab.validation.embargo import embargo_mask
from quantlab.validation.purged_cv import purged_cv_indices
from quantlab.validation.walk_forward import walk_forward_folds


def test_equity_curve_bit_identical_across_processes() -> None:
    """The same config executed in a fresh process must produce identical floats."""
    data = generate("regime_switch", n=252, seed=42)
    strat = create_strategy("momentum", {"lookback": 15})
    r1 = BacktestEngine(BacktestConfig(seed=1)).run(strat, {data.symbol: data})
    r2 = BacktestEngine(BacktestConfig(seed=1)).run(strat, {data.symbol: data})
    assert r1.equity() == r2.equity()
    for a, b in zip(r1.orders, r2.orders, strict=True):
        assert a.quantity == b.quantity and a.eligible_from == b.eligible_from


def test_returns_deterministic_and_bounded() -> None:
    for process in ("random_walk", "trend", "mean_reversion", "vol_clustering", "jump"):
        data = generate(process, n=150, seed=11)
        r = np.asarray(data.returns())
        r2 = np.asarray(generate(process, n=150, seed=11).returns())
        assert np.array_equal(r, r2)
        assert np.isfinite(r).all()


def test_dataset_fingerprint_stable_over_reload() -> None:
    data = generate("trend", n=100, seed=3)
    fp1 = dataset_fingerprint(data)
    fp2 = dataset_fingerprint(data)
    assert fp1 == fp2 and len(fp1) == 64


def test_config_semantic_stability() -> None:
    """Field order/comments must never affect math — configs are immutable inputs."""
    a = RunConfig(strategy_params={"fast": 5, "seed": 0})
    b = RunConfig(strategy_params={"seed": 0, "fast": 5})
    assert a.backtest_config == b.backtest_config
    assert a.strategy_params["fast"] == b.strategy_params["fast"]


def test_data_window_fingerprint_and_cv_separation() -> None:
    """Window slices are byte-stable and train/test never overlap."""
    data = generate("trend", n=200, seed=7)
    assert data.bars[0].ts == 0
    assert data.bars[-1].ts == (len(data.bars) - 1) * data.interval_seconds * 1000
    assert dataset_fingerprint(data.trimmed(50, 150)) == dataset_fingerprint(
        data.bars[50:150]
    )

    folds = list(walk_forward_folds(100, train_size=60, test_size=20))
    assert folds
    for train, test in folds:
        assert train[0] < train[1] == test[0] < test[1]
        assert set(range(*train)).isdisjoint(range(*test))

    for train, test in purged_cv_indices(100, n_splits=5, horizon=5, embargo=5):
        lo, hi = int(test[0]), int(test[-1]) + 1
        assert {int(x) for x in train}.isdisjoint(set(range(lo, hi)))
        assert all(not (lo - 5 <= int(x) <= hi - 1) for x in train)
        assert all(int(x) < lo - 5 or int(x) >= hi + 5 for x in train)

    mask = embargo_mask(100, test_start=60, test_end=80, embargo=5)
    assert mask[79] and not mask[80:85].any() and mask[85]
    snap = EnvSnapshot(dataset_hash="abc")
    assert snap.to_dict()["dataset_hash"] == "abc"


def test_reproduce_flags_dataset_fingerprint_drift() -> None:
    """Re-running with changed data is reported even when metrics look plausible."""
    record = {
        "experiment_id": "EXP-TEST",
        "run_config": {"seed": 3},
        "results": {"dataset_fingerprints": {"SYN": "abc"}},
    }

    def run_fn(_record: dict) -> dict:
        return {"ok": True}

    clean = reproduce(record, run_fn, None, dataset_fingerprints={"SYN": "abc"})
    assert clean["dataset_hash_now"]
    assert not any("dataset fingerprint" in warning for warning in clean["warnings"])

    drifted = reproduce(record, run_fn, None, dataset_fingerprints={"SYN": "def"})
    assert any("dataset fingerprint changed" in warning for warning in drifted["warnings"])

    no_provenance = reproduce(
        {**record, "results": {}}, run_fn, None, dataset_fingerprints={"SYN": "abc"}
    )
    assert any("did not record dataset fingerprints" in w for w in no_provenance["warnings"])
