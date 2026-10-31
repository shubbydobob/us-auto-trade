"""Triple-barrier labelling, measured from the assumed T+1 entry price.

Barriers are anchored to the T+1 OPEN, not the T close. The signal fires at the
close of T; the earliest fill is the next session. Labelling off the T close
would quietly hand the strategy a free overnight gap.

Unachievable-trade hygiene (excluded from training, not silently labelled 1):
  * T+1 opens above the profit barrier  -- a gap-up you could never have joined
  * T+1 opens below -15%                -- thesis already dead before entry
  * T+1 has no bar                      -- halted / delisted

Barrier touches are judged on intraday high/low, which is optimistic: touching a
high is not the same as filling there. That optimism is deliberately confined to
labelling -- all execution realism lives in backtest/costs.py. Never read label
precision as backtest P&L.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LABEL_COLUMNS = [
    "symbol",
    "dt",
    "entry_px",
    "y_explode",
    "mfe",
    "mae",
    "exit_reason",
    "days_held",
    "excluded",
    "exclude_reason",
]


def label_symbol(
    df: pd.DataFrame,
    horizon: int = 20,
    upper: float = 0.60,
    lower: float = -0.20,
    gap_up_exclude: float = 0.60,
    gap_down_exclude: float = -0.15,
) -> pd.DataFrame:
    """Label one symbol's bars. `df` must be sorted by dt with as-traded prices."""
    n = len(df)
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)

    entry_px = np.full(n, np.nan)
    y = np.full(n, np.nan)
    mfe = np.full(n, np.nan)
    mae = np.full(n, np.nan)
    days = np.full(n, np.nan)
    reason = np.full(n, "", dtype=object)
    excluded = np.zeros(n, dtype=bool)
    excl_reason = np.full(n, "", dtype=object)

    # The final bar has no T+1 to enter on. Mark it excluded rather than leaving
    # it with excluded=False and y=NaN, which would slip past a caller that
    # filters on `excluded` alone.
    if n:
        excluded[n - 1] = True
        excl_reason[n - 1] = "no_next_bar"

    for i in range(n - 1):
        entry = o[i + 1]
        if not np.isfinite(entry) or entry <= 0:
            excluded[i] = True
            excl_reason[i] = "no_next_bar"
            continue

        gap = entry / c[i] - 1
        if gap > gap_up_exclude:
            excluded[i] = True
            excl_reason[i] = "gap_up_unachievable"
            continue
        if gap < gap_down_exclude:
            excluded[i] = True
            excl_reason[i] = "gap_down_thesis_dead"
            continue

        entry_px[i] = entry
        stop_px = entry * (1 + lower)
        target_px = entry * (1 + upper)

        end = min(i + 1 + horizon, n)
        window_h = h[i + 1 : end]
        window_l = l[i + 1 : end]
        if window_h.size == 0:
            excluded[i] = True
            excl_reason[i] = "no_forward_window"
            continue

        mfe[i] = np.nanmax(window_h) / entry - 1
        mae[i] = np.nanmin(window_l) / entry - 1

        hit_up = np.flatnonzero(window_h >= target_px)
        hit_dn = np.flatnonzero(window_l <= stop_px)
        first_up = hit_up[0] if hit_up.size else np.inf
        first_dn = hit_dn[0] if hit_dn.size else np.inf

        if first_up < first_dn:
            y[i], reason[i], days[i] = 1.0, "target", first_up + 1
        elif first_dn < np.inf:
            y[i], reason[i], days[i] = 0.0, "stop", first_dn + 1
        else:
            y[i], reason[i], days[i] = 0.0, "time", window_h.size

        # Same-bar ambiguity: if both barriers are touched on the same day we
        # cannot tell the order from daily bars, so assume the adverse one.
        if first_up == first_dn < np.inf:
            y[i], reason[i] = 0.0, "stop_ambiguous"

    out = pd.DataFrame(
        {
            "symbol": df["symbol"].to_numpy(),
            "dt": df["dt"].to_numpy(),
            "entry_px": entry_px,
            "y_explode": y,
            "mfe": mfe,
            "mae": mae,
            "exit_reason": reason,
            "days_held": days,
            "excluded": excluded,
            "exclude_reason": excl_reason,
        }
    )
    return out[LABEL_COLUMNS]


def build_labels(bars: pd.DataFrame, **kwargs) -> pd.DataFrame:
    frames = [
        label_symbol(g.reset_index(drop=True), **kwargs)
        for _, g in bars.sort_values(["symbol", "dt"]).groupby("symbol", sort=False)
    ]
    if not frames:
        return pd.DataFrame(columns=LABEL_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def average_uniqueness(labels: pd.DataFrame, horizon: int = 20) -> pd.Series:
    """Lopez de Prado sample weights: 1 / mean concurrent label count.

    Labels overlap by up to `horizon`-1 days and positives cluster (one sector
    squeeze creates hundreds of correlated positives), so the nominal row count
    wildly overstates the information content. These weights, plus purged CV,
    are what keep the effective sample size honest.
    """
    df = labels.dropna(subset=["y_explode"]).copy()
    if df.empty:
        return pd.Series(dtype=float)
    dt = pd.to_datetime(df["dt"])
    counts = dt.value_counts().sort_index()
    rolling_concurrency = counts.rolling(horizon, min_periods=1).sum()
    conc = dt.map(rolling_concurrency).astype(float)
    return (1.0 / conc.replace(0, np.nan)).fillna(0.0)


def effective_sample_size(labels: pd.DataFrame, cluster_days: int = 10) -> dict:
    """Independent-episode count: the number that actually bounds model complexity.

    Two layers of dependence inflate the nominal positive count:

    1. *Within a symbol*, consecutive days of the same run are one event -- a
       stock that doubles produces ~20 positive rows for one move. Positives are
       therefore collapsed per symbol when they fall within `cluster_days`.
    2. *Across symbols*, one sector squeeze lights up dozens of names at once.
       The calendar-cluster count caps how many of those symbol-episodes can be
       treated as independent draws.

    Clustering globally by date alone is meaningless at universe scale (some
    symbol explodes almost every session), so the per-symbol pass comes first.
    """
    pos = labels[(labels["y_explode"] == 1) & (~labels["excluded"])].copy()
    if pos.empty:
        return {
            "nominal_positives": 0,
            "symbol_episodes": 0,
            "calendar_clusters": 0,
            "inflation_factor": 0.0,
        }

    pos["dt"] = pd.to_datetime(pos["dt"])
    pos = pos.sort_values(["symbol", "dt"])
    gap = pos.groupby("symbol")["dt"].diff().dt.days
    new_episode = gap.isna() | (gap > cluster_days)
    pos["episode_id"] = new_episode.cumsum()

    episodes = pos.groupby("episode_id").agg(symbol=("symbol", "first"),
                                             start=("dt", "min"))
    symbol_episodes = len(episodes)

    starts = episodes["start"].sort_values()
    clusters, last = 1, starts.iloc[0]
    for d in starts.iloc[1:]:
        if (d - last).days > cluster_days:
            clusters += 1
        last = d

    return {
        "nominal_positives": int(len(pos)),
        "symbol_episodes": int(symbol_episodes),
        "calendar_clusters": int(clusters),
        "inflation_factor": round(len(pos) / max(symbol_episodes, 1), 1),
    }
