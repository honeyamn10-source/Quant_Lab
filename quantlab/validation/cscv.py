"""Combinatorially Symmetric Cross-Validation (CSCV) and Probability of
Backtest Overfitting (PBO), following Bailey, Borwein, Lopez de Prado & Zhu.

Convention (documented to avoid ambiguity):
  * Matrix is (T returns, N strategies/configs); rows are split into
    `num_splits` contiguous submatrices; every combination of S/2 submatrices is
    the in-sample block, the complement is the out-of-sample block.
  * Within each block, strategies are ranked by Sharpe ratio.
  * ``oos_ranks`` = the 1-based out-of-sample rank of the in-sample winner:
        1 = IS winner is also the OOS winner (top),
        N = IS winner is the OOS loser (bottom).
  * ``logit`` = ln( (N - r + 1) / r )   positive <=> IS winner in the top half
  * A preference inversion (overfit) is IS winner landing in the BOTTOM half:
        r > N / 2.
  * PBO = fraction of combinations with an inversion.
"""

from __future__ import annotations

import itertools
import math

import numpy as np


def _block_sharpe(matrix: np.ndarray) -> np.ndarray:
    """Per-column Sharpe (NaN columns -> 0.0)."""
    out = []
    for c in range(matrix.shape[1]):
        r = matrix[:, c]
        r = r[~np.isnan(r)]
        if len(r) < 2:
            out.append(0.0)
            continue
        sd = float(np.std(r, ddof=1))
        out.append(float(np.mean(r) / sd) if sd > 0 else 0.0)
    return np.asarray(out, dtype=float)


def _submatrices(matrix: np.ndarray, num_splits: int) -> list[np.ndarray]:
    rows, _ = matrix.shape
    if rows < num_splits or num_splits % 2 != 0:
        raise ValueError("need rows >= num_splits and num_splits even")
    block = rows // num_splits
    return [matrix[k * block : (k + 1) * block] for k in range(num_splits)]


def cscv_logits(matrix: np.ndarray, num_splits: int = 6) -> dict:
    """Return per-combination logits plus the raw out-of-sample ranks."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2-D (T x N)")
    rows, cols = matrix.shape
    blocks = _submatrices(matrix, num_splits)
    logits: list[float] = []
    oos_ranks: list[float] = []
    n_inv = 0
    for chosen in itertools.combinations(range(num_splits), num_splits // 2):
        is_block = np.concatenate([blocks[k] for k in chosen])
        oos_block = np.concatenate([blocks[k] for k in range(num_splits) if k not in chosen])
        is_sr = _block_sharpe(is_block)          # higher = better
        oos_sr = _block_sharpe(oos_block)
        is_best = int(np.argmax(is_sr))          # in-sample winner
        # 1-based OOS rank: 1 = top (best performer OOS), N = bottom (worst)
        r = int((oos_sr > oos_sr[is_best]).sum()) + 1
        oos_ranks.append(float(r))
        logits.append(math.log((cols - r + 1) / max(r, 1e-9)))
        if r > cols / 2:
            n_inv += 1
    return {
        "logits": logits,
        "oos_ranks": oos_ranks,
        "n_inversions": n_inv,
        "n_combinations": len(logits),
        "y_s": num_splits,
        "n_strategies": cols,
        "n_returns": rows,
    }


def probability_of_backtest_overfitting(matrix: np.ndarray, num_splits: int = 6) -> dict:
    """PBO = share of combinations where the in-sample winner ranks in the
    bottom half of the out-of-sample ranking (preference inversion)."""
    out = cscv_logits(matrix, num_splits)
    logits = out["logits"]
    pbo = float(np.mean([logit < 0 for logit in logits]))
    out["pbo"] = pbo
    out["median_logit"] = float(np.median(logits))
    out["min_logit"] = float(np.min(logits))
    out["max_logit"] = float(np.max(logits))
    out["interpretation"] = (
        "ELEVATED OVERFIT RISK" if pbo > 0.5 else
        "MODERATE OVERFIT RISK" if pbo > 0.2 else
        "LOW OVERFIT RISK"
    )
    return out
