"""Stress-scenario analytics for a returns panel.

A stress scenario is a named calendar window; the report measures how
diversification and volatility behave inside the window versus outside it.
Because the caller provides no date index, calendar dates are mapped to rows
proportionally: the full returns history is assumed to span the reference
window ``[min scenario start, max scenario end]`` across ``SCENARIOS`` and each
nominal month is placed at the row position with the same proportion of the
remaining span.  Integer ``start``/``end`` values are treated as literal row
indices instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantlab.validation.metrics import max_drawdown, volatility

__all__ = [
    "SCENARIOS",
    "StressScenario",
    "normal_correlation",
    "portfolio_stress_report",
    "stress_correlation",
]


@dataclass(frozen=True)
class StressScenario:
    """A named stress window.

    ``start``/``end`` are either nominal ``"YYYY-MM"`` strings (mapped to rows
    proportionally) or integer row indices (used verbatim).
    """

    name: str
    description: str
    start: str | int
    end: str | int


SCENARIOS: list[StressScenario] = [
    StressScenario(
        name="2008_financial_crisis",
        description="global financial crisis",
        start="2008-01",
        end="2009-03",
    ),
    StressScenario(
        name="2020_covid_crash",
        description="covid-19 market crash",
        start="2020-02",
        end="2020-04",
    ),
    StressScenario(
        name="2022_rate_shock",
        description="interest-rate shock / bond rout",
        start="2022-01",
        end="2022-10",
    ),
    StressScenario(
        name="2023_banking_stress",
        description="regional-bank (SVB) and Credit Suisse stress",
        start="2023-03",
        end="2023-05",
    ),
]


def _month_index(label: str | int) -> int:
    """Nominal ``"YYYY-MM"`` label -> month ordinal (ints pass through)."""
    if isinstance(label, int):
        return label
    year, month = (int(part) for part in label.split("-"))
    return year * 12 + month - 1


def _scenario_mask(n_rows: int, scenario: StressScenario) -> np.ndarray:
    """Boolean row mask for a scenario over a ``n_rows`` returns history.

    Integer bounds are used as row indices directly.  Nominal ``"YYYY-MM"``
    bounds are mapped proportionally across the reference window spanned by
    ``SCENARIOS`` (see the module docstring).
    """
    mask = np.zeros(n_rows, dtype=bool)
    if isinstance(scenario.start, int) and isinstance(scenario.end, int):
        lo = max(0, scenario.start)
        hi = min(n_rows, scenario.end + 1)
        mask[lo:hi] = True
        return mask
    ref_lo = min(_month_index(s.start) for s in SCENARIOS)
    ref_hi = max(_month_index(s.end) for s in SCENARIOS)
    span = max(1, ref_hi - ref_lo)
    s_lo = _month_index(scenario.start)
    s_hi = _month_index(scenario.end)
    start_idx = int(round((s_lo - ref_lo) / span * (n_rows - 1)))
    end_idx = int(round((s_hi - ref_lo) / span * (n_rows - 1)))
    start_idx = int(np.clip(start_idx, 0, n_rows - 1))
    end_idx = int(np.clip(end_idx, 0, n_rows - 1))
    mask[start_idx : end_idx + 1] = True
    return mask


def _mean_pairwise_corr(rows: np.ndarray) -> float:
    """Mean off-diagonal Pearson correlation over the given rows."""
    n_assets = rows.shape[1]
    if rows.shape[0] < 2 or n_assets < 2:
        return np.nan
    corr = np.corrcoef(rows, rowvar=False)
    off = corr[~np.eye(n_assets, dtype=bool)]
    return float(np.nanmean(off))


def stress_correlation(returns: np.ndarray, mask: np.ndarray) -> float:
    """Mean pairwise correlation over the masked (stressed) rows."""
    r = np.asarray(returns, dtype=float)
    if r.ndim == 1:
        r = r.reshape(-1, 1)
    rows = r[mask]
    rows = rows[~np.isnan(rows).any(axis=1)]
    return _mean_pairwise_corr(rows)


def normal_correlation(returns: np.ndarray, mask: np.ndarray) -> float:
    """Mean pairwise correlation over the complement of ``mask``."""
    r = np.asarray(returns, dtype=float)
    if r.ndim == 1:
        r = r.reshape(-1, 1)
    rows = r[~mask]
    rows = rows[~np.isnan(rows).any(axis=1)]
    return _mean_pairwise_corr(rows)


def portfolio_stress_report(
    returns: np.ndarray,
    portfolio_returns: np.ndarray | None = None,
    scoring: bool = False,
) -> dict[str, object]:
    """Build the scenario stress report for a returns panel.

    ``portfolio_returns`` defaults to the equal-weight row mean of ``returns``
    (NaN-safe).  Scenario windows map to rows per the module docstring.  Every
    scenario dict carries:

    ``scenario``            scenario name
    ``normal_corr``         mean pairwise correlation outside the window
    ``stress_corr``         mean pairwise correlation inside the window
    ``corr_expansion``      ``stress_corr - normal_corr``
    ``stress_vol``          annualized volatility inside the window
    ``normal_vol``          annualized volatility outside the window
    ``vol_ratio``           ``stress_vol / normal_vol`` (NaN if flat normal vol)
    ``stress_max_drawdown`` max drawdown inside the window
    ``stress_p95_return``   95th-percentile portfolio return inside the window

    The top-level dict adds ``scenarios`` (the list above), ``worst_scenario``
    (the scenario with the largest correlation expansion) and
    ``diversification_degradation`` (that maximum).  With ``scoring=True`` a
    per-scenario ``score`` (``corr_expansion * vol_ratio``) is added.
    """
    r = np.asarray(returns, dtype=float)
    if r.ndim == 1:
        r = r.reshape(-1, 1)
    n_rows = r.shape[0]
    if portfolio_returns is None:
        pf = np.nanmean(r, axis=1)
    else:
        pf = np.asarray(portfolio_returns, dtype=float)
    scenario_rows: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        mask = _scenario_mask(n_rows, scenario)
        stress_rows = pf[mask]
        normal_rows = pf[~mask]
        stress_corr = stress_correlation(r, mask)
        normal_corr = normal_correlation(r, mask)
        expansion = stress_corr - normal_corr
        stress_vol = volatility(stress_rows)
        normal_vol = volatility(normal_rows)
        vol_ratio = stress_vol / normal_vol if normal_vol else np.nan
        row: dict[str, object] = {
            "scenario": scenario.name,
            "normal_corr": normal_corr,
            "stress_corr": stress_corr,
            "corr_expansion": expansion,
            "stress_vol": stress_vol,
            "normal_vol": normal_vol,
            "vol_ratio": vol_ratio,
            "stress_max_drawdown": max_drawdown(stress_rows),
            "stress_p95_return": (
                float(np.quantile(stress_rows, 0.95)) if stress_rows.size else np.nan
            ),
        }
        if scoring:
            row["score"] = float(expansion * vol_ratio)
        scenario_rows.append(row)
    valid = []
    for row in scenario_rows:
        corr = row["corr_expansion"]
        if isinstance(corr, float) and not np.isnan(corr):
            valid.append(row)
    best_row: tuple[float, str] | None = None
    for row in valid:
        corr = row["corr_expansion"]
        assert isinstance(corr, float)
        name = str(row["scenario"])
        if best_row is None or corr > best_row[0]:
            best_row = (corr, name)
    if best_row is not None:
        degradation, worst_name = float(best_row[0]), best_row[1]
    else:
        worst_name = ""
        degradation = np.nan
    return {
        "scenarios": scenario_rows,
        "worst_scenario": worst_name,
        "diversification_degradation": degradation,
    }
