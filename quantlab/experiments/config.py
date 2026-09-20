"""Strategy run configuration (YAML/JSON). Every experiment is fully specified
by one of these files plus a seed, so results are reproducible."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    symbols: list[str] = Field(default_factory=lambda: ["SYN"])
    process: str = "random_walk"
    n_bars: int = 504
    seed: int = 7
    source: str = "synthetic"


class RunConfig(BaseModel):
    """Everything needed to run and reproduce an experiment."""

    name: str = "untitled"
    strategy: str = "sma_crossover"
    strategy_params: dict = Field(default_factory=dict)
    data: DataConfig = Field(default_factory=DataConfig)
    initial_cash: float = 1_000_000.0
    execution_delay: int = 1
    execution_mode: str = "next_open"
    profile: str = "normal"
    max_leverage: float = 2.0
    max_position_weight: float = 1.0
    rebalance_tolerance: float = 0.005
    commission_bps: float = 5.0
    participation_cap: float = 0.10
    borrow_rate_year: float = 0.0
    seed: int = 0
    notes: str = ""

    @property
    def backtest_config(self) -> dict:
        return {
            "initial_cash": self.initial_cash,
            "execution_delay": self.execution_delay,
            "execution_mode": self.execution_mode,
            "profile": self.profile,
            "max_leverage": self.max_leverage,
            "max_position_weight": self.max_position_weight,
            "rebalance_tolerance": self.rebalance_tolerance,
            "commission_bps": self.commission_bps,
            "participation_cap": self.participation_cap,
            "borrow_rate_year": self.borrow_rate_year,
            "seed": self.seed,
        }


def load_run_config(path: str | Path) -> RunConfig:
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    data = yaml.safe_load(raw) if p.suffix in (".yaml", ".yml") else json.loads(raw)
    return RunConfig(**data)


def dump_run_config(cfg: RunConfig, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix in (".yaml", ".yml"):
        p.write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False), encoding="utf-8")
    else:
        p.write_text(json.dumps(cfg.model_dump(mode="json"), indent=2), encoding="utf-8")
