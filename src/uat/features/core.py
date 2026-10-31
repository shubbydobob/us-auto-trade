"""Primitive indicator maths, numpy/pandas only (no ta-lib).

Every function here is trailing-only by construction. The invariance test in
tests/test_no_lookahead.py deletes all bars after T, recomputes, and asserts the
value at T is unchanged -- which is what catches center=True, shift(-n), bfill,
and full-sample normalisation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)


def wilder_atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's ATR = EMA of true range with alpha = 1/n."""
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def percent_rank(series: pd.Series, window: int) -> pd.Series:
    """Fraction of the trailing `window` values strictly below the current one.

    Uses the same symbol's own history, so it answers "is this unusual for THIS
    stock" rather than "is this unusual across the market".
    """
    return series.rolling(window, min_periods=max(20, window // 4)).rank(pct=True)


def ols_slope(series: pd.Series, window: int) -> pd.Series:
    """Rolling OLS slope against t = 0..window-1.

    beta = cov(t, y) / var(t); var(t) is constant for a fixed window, so this is
    a dot product with a fixed centred-time vector -- cheap and exact.
    """
    t = np.arange(window, dtype=float)
    t_centred = t - t.mean()
    denom = (t_centred**2).sum()

    def _beta(vals: np.ndarray) -> float:
        if np.isnan(vals).any():
            return np.nan
        return float(np.dot(t_centred, vals - vals.mean()) / denom)

    return series.rolling(window, min_periods=window).apply(_beta, raw=True)


def zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=window // 2).mean()
    std = series.rolling(window, min_periods=window // 2).std()
    return (series - mean) / std.replace(0, np.nan)


def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


def clip01(x: pd.Series) -> pd.Series:
    return x.clip(lower=0.0, upper=1.0)


def rolling_median(series: pd.Series, window: int) -> pd.Series:
    """Median, never mean, for volume baselines.

    Penny-stock volume routinely has single-day 50x outliers; a mean baseline is
    permanently inflated by one spike and makes every later day look quiet.
    """
    return series.rolling(window, min_periods=max(5, window // 2)).median()
