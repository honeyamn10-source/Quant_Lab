"""Regime detection models over an asset return series.

Four model families are provided and can be blended by :func:`ensemble_regimes`:

    rule_based_regimes        threshold rules over extracted features
    gaussian_mixture_regimes  pure-numpy EM fit of a Gaussian mixture
    hmm_regimes               Gaussian-emission HMM fit by forward-backward EM
    change_point_regimes      binary-segmentation mean-shift detection

State remapping heuristic
-------------------------
Canonical regime ids (0..3) correspond to the ordered names
``low_vol_trend``, ``high_vol_trend``, ``high_vol_sideways``, ``crisis_stress``.
Learned model states are dimensionless ids, so each detected state is assigned
to a canonical id by sorting the fitted state moments ascending on
``(mean return, mean volatility)`` and mapping the sorted position i to the
canonical id i. A state with low mean return and low volatility therefore maps
to ``low_vol_trend`` (0) and a state with high volatility and low mean return
maps highest (towards ``crisis_stress``).
"""

from __future__ import annotations

import numpy as np

CANONICAL_NAMES = ["low_vol_trend", "high_vol_trend", "high_vol_sideways", "crisis_stress"]
_VAR_FLOOR = 1e-12
_REGIME_COUNT = 4


def _finite_returns(returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the finite returns and the index into the original array."""
    r = np.asarray(returns, dtype=np.float64)
    idx = np.flatnonzero(np.isfinite(r))
    return r[idx], idx


def _map_states_by_moments(
    labels: np.ndarray, returns: np.ndarray, state_count: int
) -> np.ndarray:
    """Map raw state ids onto canonical regime ids via moment ordering."""
    r, valid = _finite_returns(returns)
    mapping = np.full(state_count, -1, dtype=np.int64)
    if r.size and state_count > 0:
        means = []
        vols = []
        for state in range(state_count):
            sel = labels[valid] == state
            means.append(float(np.mean(r[sel])) if np.count_nonzero(sel) else np.inf)
            vols.append(float(np.std(r[sel], ddof=1)) if np.count_nonzero(sel) > 1 else 0.0)
        ranking = np.lexsort((np.array(vols), np.array(means)))
        for position, raw_state in enumerate(ranking):
            mapping[raw_state] = position
    mapped = np.full(labels.shape, -1, dtype=np.int64)
    mapped[valid] = mapping[labels[valid]]
    return mapped


def rule_based_regimes(features: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    """Assign integer regime labels 0..3 from feature thresholds.

    Rules (quantiles taken over finite values of each feature):
        crisis_stress (3): tail_stress >= 2, or (drawdown below the 20th
            percentile and realized_vol above the 80th percentile).
        otherwise high vol (realized_vol above 80th pct):
            high_vol_trend (1) if trend above the median, else
            high_vol_sideways (2).
        otherwise low vol: low_vol_trend (0).
    Returns ``(labels, names)`` with names of length 4.
    """
    names = list(CANONICAL_NAMES)
    trend = np.asarray(features["trend"], dtype=np.float64)
    realized_vol = np.asarray(features["realized_vol"], dtype=np.float64)
    drawdown = np.asarray(features["drawdown"], dtype=np.float64)
    tail_stress = np.asarray(features["tail_stress"], dtype=np.float64)

    labels = np.zeros(trend.shape, dtype=np.int64)
    vol_hi = np.nanpercentile(realized_vol, 80.0)
    trend_med = np.nanmedian(trend)
    dd_lo = np.nanpercentile(drawdown, 20.0)

    high_vol = realized_vol >= vol_hi
    trending = trend >= trend_med
    crisis = (tail_stress >= 2.0) | (high_vol & (drawdown <= dd_lo))

    labels = np.where(crisis, 3, np.where(high_vol, np.where(trending, 1, 2), 0))
    return np.asarray(labels, dtype=np.int64), names


def gaussian_mixture_regimes(
    returns: np.ndarray, n_states: int = 3, seed: int = 0, n_iter: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a Gaussian mixture on returns by pure-numpy EM.

    Returns ``(hard_labels, responsibilities)`` where responsibilities has shape
    (T, n_states) and hard_labels is the per-bar argmax (soft labels share the
    mixture's state ordering but are arbitrary). NaN return bars are assigned
    hard label -1 and a NaN responsibility row.
    """
    r_full = np.asarray(returns, dtype=np.float64)
    n_total = r_full.size
    r, valid = _finite_returns(r_full)
    rng = np.random.default_rng(seed)
    k = int(n_states)

    responsibilities = np.full((n_total, k), np.nan, dtype=np.float64)
    hard = np.full(n_total, -1, dtype=np.int64)
    if r.size == 0:
        return hard, responsibilities
    if r.size < k:
        k = 1

    means = r[rng.choice(r.size, size=k, replace=r.size < k)] + rng.normal(0.0, 1e-3, k)
    variances = np.full(k, max(float(np.var(r)) if r.size > 1 else 1.0, _VAR_FLOOR))
    weights = np.full(k, 1.0 / k)
    log_r = r[:, None]
    response = np.zeros((r.size, k))

    for _ in range(int(n_iter)):
        variances = np.maximum(variances, _VAR_FLOOR)
        log_lik = (
            -0.5 * ((log_r - means[None, :]) ** 2) / variances[None, :]
            - 0.5 * np.log(2.0 * np.pi * variances[None, :])
            + np.log(weights[None, :])
        )
        log_lik = log_lik - np.max(log_lik, axis=1, keepdims=True)
        response = np.exp(log_lik)
        row_sum = response.sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum > 0.0, row_sum, 1.0)
        response = response / row_sum

        counts = response.sum(axis=0)
        counts = np.maximum(counts, _VAR_FLOOR)
        weights = counts / response.shape[0]
        means = (response * r[:, None]).sum(axis=0) / counts
        variances = (response * (r[:, None] - means[None, :]) ** 2).sum(axis=0) / counts
        variances = np.maximum(variances, _VAR_FLOOR)

    responsibilities[valid] = response
    hard[valid] = np.argmax(response, axis=1)
    return hard, responsibilities


def hmm_regimes(
    returns: np.ndarray, n_states: int = 3, seed: int = 0, n_iter: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a Gaussian-emission HMM by forward-backward EM.

    The M-step re-estimates the initial distribution, transition matrix, state
    means and state variances; the E-step runs scaled forward-backward. Returns
    ``(states, posterior)`` where `states` is the per-bar state with the highest
    forward probability (viterbi-style) and `posterior` has shape (T, n_states).
    NaN return bars are assigned state -1 and a NaN posterior row.
    """
    r_full = np.asarray(returns, dtype=np.float64)
    n_total = r_full.size
    r, valid = _finite_returns(r_full)
    rng = np.random.default_rng(seed)
    k = int(n_states)

    states = np.full(n_total, -1, dtype=np.int64)
    posterior = np.full((n_total, k), np.nan, dtype=np.float64)
    if r.size == 0:
        return states, posterior
    if r.size < k:
        k = 1

    means = r[rng.choice(r.size, size=k, replace=r.size < k)] + rng.normal(0.0, 1e-3, k)
    variances = np.full(k, max(float(np.var(r)) if r.size > 1 else 1.0, _VAR_FLOOR))
    weights = np.full(k, 1.0 / k)
    transition = np.full((k, k), 0.01 / (k - 1) if k > 1 else 0.0)
    np.fill_diagonal(transition, 0.99 if k > 1 else 1.0)
    t_len = r.size

    gamma = np.zeros((t_len, k))
    xi = np.zeros((t_len - 1, k, k))

    for _ in range(int(n_iter)):
        variances = np.maximum(variances, _VAR_FLOOR)
        emission = np.exp(
            -0.5 * ((r[:, None] - means[None, :]) ** 2) / variances[None, :]
            - 0.5 * np.log(2.0 * np.pi * variances[None, :])
        )
        alpha = np.zeros((t_len, k))
        scale = np.zeros(t_len)
        alpha[0] = weights * emission[0]
        scale[0] = alpha[0].sum()
        alpha[0] = alpha[0] / max(scale[0], _VAR_FLOOR)
        for t in range(1, t_len):
            alpha[t] = (alpha[t - 1] @ transition) * emission[t]
            scale[t] = alpha[t].sum()
            alpha[t] = alpha[t] / max(scale[t], _VAR_FLOOR)

        beta = np.zeros((t_len, k))
        beta[-1] = 1.0
        for t in range(t_len - 2, -1, -1):
            beta[t] = (transition @ (emission[t + 1] * beta[t + 1])) / max(
                scale[t + 1], _VAR_FLOOR
            )

        gamma = alpha * beta
        gamma = gamma / gamma.sum(axis=1, keepdims=True)

        for t in range(t_len - 1):
            numerator = (alpha[t, :, None] * transition) * (
                emission[t + 1][None, :] * beta[t + 1][None, :]
            )
            denom = max(np.sum(numerator), _VAR_FLOOR)
            xi[t] = numerator / denom

        weights = gamma[0]
        norm_xi = xi.sum(axis=0)
        norm_gamma = gamma[:-1].sum(axis=0)
        transition = np.divide(
            norm_xi,
            np.where(norm_gamma[:, None] > _VAR_FLOOR, norm_gamma[:, None], 1.0),
        )
        transition = transition / transition.sum(axis=1, keepdims=True)
        transition = np.where(np.isfinite(transition), transition, 0.0)

        counts = gamma.sum(axis=0)
        counts = np.maximum(counts, _VAR_FLOOR)
        means = (gamma * r[:, None]).sum(axis=0) / counts
        variances = (gamma * (r[:, None] - means[None, :]) ** 2).sum(axis=0) / counts
        variances = np.maximum(variances, _VAR_FLOOR)

    posterior[valid] = gamma
    states[valid] = np.argmax(alpha, axis=1)
    return states, posterior


def _segment_cost(x: np.ndarray) -> float:
    """Sum of squared deviations of x from its mean (float64)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    return float(np.sum((x - np.mean(x)) ** 2))


def _best_split(x: np.ndarray) -> tuple[float, int]:
    """Best split index in ``lo < i < hi`` by minimal combined SSE (>= min_len)."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 4:
        return 0.0, -1
    min_len = 2
    cum = np.cumsum(x, dtype=np.float64)
    cum_sq = np.cumsum(x**2, dtype=np.float64)
    best_gain = 0.0
    best_idx = -1
    total_var = cum_sq[-1] - cum[-1] ** 2 / n
    for i in range(min_len, n - min_len + 1):
        left_var = cum_sq[i - 1] - cum[i - 1] ** 2 / i
        right_n = n - i
        right_var = (cum_sq[-1] - cum_sq[i - 1]) - (cum[-1] - cum[i - 1]) ** 2 / right_n
        gain = total_var - (left_var + right_var)
        if gain > best_gain:
            best_gain = gain
            best_idx = i
    return best_gain, best_idx


def change_point_regimes(
    returns: np.ndarray, max_changes: int = 3, penalty: float = 1.0
) -> np.ndarray:
    """Binary-segment `returns` on mean shifts; return integer segment ids.

    Greedy recursive splitting: repeatedly apply the best single split that
    reduces the within-segment sum of squares by more than `penalty`, up to
    `max_changes` splits. NaN bars are masked out. Segment ids are 0-based and
    chronological.
    """
    r_full = np.asarray(returns, dtype=np.float64)
    r, valid = _finite_returns(r_full)
    labels = np.zeros(r_full.shape, dtype=np.int64)
    if r.size == 0:
        return labels

    queue = [np.arange(0, r.size, dtype=np.int64)]
    for _ in range(int(max_changes)):
        best_gain = 0.0
        best_seg = -1
        best_idx = -1
        for seg_pos, seg in enumerate(queue):
            gain, idx = _best_split(r[seg])
            if idx > 0 and gain > best_gain:
                best_gain = gain
                best_seg = seg_pos
                best_idx = int(idx)
        if best_gain <= penalty or best_seg < 0 or best_idx < 0:
            break
        seg = queue.pop(best_seg)
        queue.insert(best_seg, seg[best_idx:])
        queue.insert(best_seg, seg[:best_idx])

    for seg_id, seg in enumerate(queue):
        labels[valid[seg]] = seg_id
    return labels


def ensemble_regimes(
    returns: np.ndarray, features: dict[str, np.ndarray], seed: int = 0
) -> dict:
    """Blend the four regime detectors into one canonical-state ensemble.

    Every model's states are mapped onto the 4 canonical regimes via the
    mean-return / mean-vol ordering heuristic described in the module docstring.
    Returns a dict with keys ``regime_names``, ``labels_rule``, ``labels_gmm``,
    ``labels_hmm``, ``labels_segment``, ``ensemble_labels``, ``probabilities``
    (T x 4, rows sum to 1 over contributing models) and ``consensus`` (average
    fraction of contributing models matching the dominant canonical state).
    """

    def _votes(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        contrib = labels >= 0
        rows = np.arange(labels.size)
        valid_rows = rows[contrib]
        matrix = np.zeros((labels.size, _REGIME_COUNT), dtype=np.float64)
        matrix[valid_rows, labels[valid_rows]] = 1.0
        return matrix, contrib

    names = list(CANONICAL_NAMES)
    rule_labels, _ = rule_based_regimes(features)
    rule_labels = np.where(np.isfinite(np.asarray(features["trend"])), rule_labels, -1)

    gmm_labels, _ = gaussian_mixture_regimes(returns, n_states=3, seed=seed)
    hmm_labels, _ = hmm_regimes(returns, n_states=3, seed=seed)
    segment_labels = change_point_regimes(returns)

    state_count = 3
    gmm_mapped = _map_states_by_moments(gmm_labels, returns, state_count)
    hmm_mapped = _map_states_by_moments(hmm_labels, returns, state_count)

    seg_count = int(np.max(segment_labels)) + 1 if segment_labels.size else 1
    segment_mapped = _map_states_by_moments(segment_labels, returns, seg_count)

    votes_matrix = np.zeros((returns.size, _REGIME_COUNT), dtype=np.float64)
    contrib = np.zeros(returns.size, dtype=bool)
    for labels in (rule_labels, gmm_mapped, hmm_mapped, segment_mapped):
        matrix, mask = _votes(labels)
        votes_matrix += matrix
        contrib |= mask

    probabilities = np.zeros((returns.size, _REGIME_COUNT), dtype=np.float64)
    valid_rows = np.argwhere(contrib).ravel()
    if valid_rows.size:
        denom = np.count_nonzero(
            np.stack(
                (
                    rule_labels[valid_rows] >= 0,
                    gmm_mapped[valid_rows] >= 0,
                    hmm_mapped[valid_rows] >= 0,
                    segment_mapped[valid_rows] >= 0,
                ),
                axis=0,
            ),
            axis=0,
        )
        denom = np.where(denom > 0, denom, 1.0)
        probabilities[valid_rows] = votes_matrix[valid_rows] / denom[:, None]

    row_totals = probabilities.sum(axis=1, keepdims=True)
    probabilities = np.divide(
        probabilities, np.where(row_totals > 0.0, row_totals, 1.0)
    )
    ensemble_labels = np.argmax(np.nan_to_num(probabilities, nan=0.0), axis=1)

    dominant = probabilities.max(axis=1)
    consensus = float(np.nanmean(np.where(contrib, dominant, np.nan)))

    return {
        "regime_names": names,
        "labels_rule": rule_labels,
        "labels_gmm": gmm_mapped,
        "labels_hmm": hmm_mapped,
        "labels_segment": segment_mapped,
        "ensemble_labels": ensemble_labels,
        "probabilities": probabilities,
        "consensus": consensus,
    }
