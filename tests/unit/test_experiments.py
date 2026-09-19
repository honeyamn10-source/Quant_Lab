"""Experiment registry + config + reproducibility tests."""

from __future__ import annotations

from quantlab.experiments.config import RunConfig, dump_run_config, load_run_config
from quantlab.experiments.registry import ExperimentLedger
from quantlab.experiments.reproducibility import EnvSnapshot, git_commit


def test_run_config_roundtrip_yaml(tmp_path) -> None:
    cfg = RunConfig(name="t", strategy="momentum", strategy_params={"lookback": 30}, seed=17)
    p = tmp_path / "run.yaml"
    dump_run_config(cfg, p)
    loaded = load_run_config(p)
    assert loaded.name == "t"
    assert loaded.strategy_params == {"lookback": 30}
    assert loaded.seed == 17


def test_run_config_roundtrip_json(tmp_path) -> None:
    cfg = RunConfig(name="j", data={"symbols": ["A", "B"], "process": "pairs", "n_bars": 100})
    p = tmp_path / "run.json"
    dump_run_config(cfg, p)
    loaded = load_run_config(p)
    assert loaded.name == "j"
    assert loaded.data.symbols == ["A", "B"]
    assert loaded.data.process == "pairs"


def test_ledger_append_only_ids_increment(tmp_path) -> None:
    led = ExperimentLedger(tmp_path)
    r1 = led.create("run a", "sma_crossover", RunConfig())
    r2 = led.create("run b", "momentum", RunConfig())
    assert r1["experiment_id"] == "EXP-0001"
    assert r2["experiment_id"] == "EXP-0002"
    assert led.count() == 2


def test_ledger_complete_and_reject_flows(tmp_path) -> None:
    led = ExperimentLedger(tmp_path)
    rec = led.create("run", "sma_crossover", RunConfig())
    led.complete(rec["experiment_id"], {"sharpe": 1.2})
    rec = led.get(rec["experiment_id"])
    assert rec["status"] == "completed"
    assert rec["results"]["sharpe"] == 1.2
    rec2 = led.create("failing", "random", RunConfig())
    led.complete(rec2["experiment_id"], {}, failure_reason="numpy.linalg.LinAlgError")
    assert led.get(rec2["experiment_id"])["status"] == "rejected"


def test_ledger_search_and_export(tmp_path) -> None:
    led = ExperimentLedger(tmp_path)
    led.create("same", "sma_crossover", RunConfig())
    led.create("diff", "momentum", RunConfig())
    hits = led.search(strategy_name="sma_crossover")
    assert len(hits) == 1
    assert len(led.export_json()) == 2


def test_ledger_persists_across_instances(tmp_path) -> None:
    ExperimentLedger(tmp_path).create("x", "pairs", RunConfig())
    led2 = ExperimentLedger(tmp_path)
    assert led2.count() == 1
    assert led2.get("EXP-0001")["strategy_name"] == "pairs"


def test_env_snapshot_records_key_fields(tmp_path) -> None:
    snap = EnvSnapshot(seed=3)
    d = snap.to_dict()
    assert d["seed"] == 3
    assert d["git_commit"] == git_commit(".")
    assert "numpy" in d["packages"]
    assert d["dataset_hash"] == ""


def test_reproduce_reruns_with_pinned_seed(tmp_path) -> None:
    from quantlab.experiments.reproducibility import reproduce

    led = ExperimentLedger(tmp_path)
    cfg = RunConfig(seed=99)
    rec = led.create("r", "sma_crossover", cfg)
    calls = []

    def run_fn(experiment_record: dict) -> dict:
        calls.append(experiment_record["run_config"]["seed"])
        return {"used_seed": experiment_record["run_config"]["seed"]}

    res = reproduce(rec, run_fn, led)
    assert res["result"]["used_seed"] == 99
    assert calls == [99]
    assert isinstance(res["environment_matches"], bool)
