"""The quiet-accumulation detector.

DESIGN NOTE -- read before touching the thresholds
--------------------------------------------------
The original thesis was "volatility compression + rising volume precedes the
explosion". A probe on 53 penny names (12,087 observations, 2y) found that only
HALF of that holds:

    feature                 bottom decile   top decile     (P of +100% in 20d)
    volume regime 10/50          0.99%          3.97%      <- thesis confirmed
    RVOL 5/50                    1.16%          4.14%      <- thesis confirmed
    BB BandWidth pct             0.50%          3.58%      <- thesis INVERTED
    ATR percentile               0.75%          5.62%      <- thesis INVERTED
    (baseline P(+100%) = 1.79%)

Compressed volatility had roughly one third the base rate, and the composite
signal that required compression scored WORSE than random (0.83% vs 1.79%).

That probe is not conclusive -- small sample, and survivorship bias removes
exactly the quiet-then-exploded-then-delisted names the thesis is about. So this
module computes BOTH directions and commits to neither. `research/evaluate.py`
decides, per feature, with confidence intervals. Nothing here hard-codes a
direction; the scoring weights live in model/rules.py and are set from the lift
table, not from intuition.

Point-in-time contract: every column at date T uses only bars with dt <= T.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from uat.features.core import (
    EPS,
    ols_slope,
    percent_rank,
    rolling_median,
    safe_div,
    true_range,
    wilder_atr,
    zscore,
)


def _adjusted(df: pd.DataFrame) -> tuple[pd.Series, ...]:
    """Continuous (split-adjusted) OHLC for technical maths.

    Raw as-traded prices stay available as df['close'] etc. for the price band,
    dollar volume and fill simulation -- see data/yf_source.py.
    """
    f = df["adj_factor"]
    return (
        df["open"] / f,
        df["high"] / f,
        df["low"] / f,
        df["close"] / f,
        df["volume"] * f,
    )


def compute_symbol_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features for ONE symbol. `df` must be sorted by dt ascending."""
    o, h, l, c, v = _adjusted(df)
    out = pd.DataFrame(index=df.index)
    out["dt"] = df["dt"].values
    out["symbol"] = df["symbol"].values

    # As-traded context (universe gating, sizing, cost model)
    out["close_raw"] = df["close"].values
    out["dollar_volume"] = (df["close"] * df["volume"]).values
    out["addv_20"] = rolling_median(pd.Series(out["dollar_volume"].values, index=df.index), 20)
    out["medvol_20"] = rolling_median(df["volume"], 20)

    # ---------- volatility: compression AND expansion, both signed ----------
    bb_mid = c.rolling(20, min_periods=20).mean()
    bb_std = c.rolling(20, min_periods=20).std()
    out["bbw"] = safe_div(4 * bb_std, bb_mid)
    out["bbw_pct_252"] = percent_rank(out["bbw"], 252)
    out["bbw_pct_126"] = percent_rank(out["bbw"], 126)

    atr14 = wilder_atr(h, l, c, 14)
    atr50 = wilder_atr(h, l, c, 50)
    out["atr14"] = atr14
    out["atrp"] = safe_div(atr14, c)
    out["atr_ratio"] = safe_div(atr14, atr50)
    out["atrp_pct_252"] = percent_rank(out["atrp"], 252)

    tr = true_range(h, l, c)
    out["nr7_count_20"] = (
        tr.le(tr.rolling(7, min_periods=7).min()).rolling(20, min_periods=20).sum()
    )
    out["inside_count_20"] = (
        (h.lt(h.shift(1)) & l.gt(l.shift(1))).rolling(20, min_periods=20).sum()
    )

    donch20 = safe_div(h.rolling(20, min_periods=20).max() - l.rolling(20, min_periods=20).min(), c)
    donch60 = safe_div(h.rolling(60, min_periods=60).max() - l.rolling(60, min_periods=60).min(), c)
    out["donch20"] = donch20
    out["donch_ratio"] = safe_div(donch20, donch60)

    logret = np.log(c).diff()
    out["cc_vol_ratio"] = safe_div(
        logret.rolling(20, min_periods=20).std(), logret.rolling(60, min_periods=60).std()
    )

    # ---------- volume expansion (the half the data supports) ----------
    medvol_50 = rolling_median(v, 50)
    medvol_60 = rolling_median(v, 60)
    medvol_100 = rolling_median(v, 100)
    out["rvol_5"] = safe_div(rolling_median(v, 5), medvol_50)
    out["rvol_10"] = safe_div(rolling_median(v, 10), medvol_60)
    out["rvol_20"] = safe_div(rolling_median(v, 20), medvol_100)
    out["vol_regime"] = safe_div(rolling_median(v, 10), medvol_50)

    log_vol_ma10 = np.log(v.rolling(10, min_periods=10).mean().replace(0, np.nan))
    out["vol_slope_20"] = ols_slope(log_vol_ma10, 20)
    out["vol_z_ma10"] = zscore(np.log(v.replace(0, np.nan)), 60).rolling(10, min_periods=5).mean()

    up = v.where(c > c.shift(1), 0.0).rolling(20, min_periods=20).sum()
    down = v.where(c < c.shift(1), 0.0).rolling(20, min_periods=20).sum()
    out["udvr_20"] = safe_div(up, down)

    # THE thesis in one number: volume expanding faster than range.
    out["vol_price_div"] = out["vol_slope_20"] - ols_slope(
        np.log(atr14.replace(0, np.nan)), 20
    )

    # Wyckoff effort vs result: more volume per unit of price movement = absorption.
    effort = v.rolling(20, min_periods=20).sum()
    result = logret.abs().rolling(20, min_periods=20).sum()
    out["evr"] = safe_div(effort, result)
    out["evr_pct_252"] = percent_rank(out["evr"], 252)

    # ---------- accumulation / distribution ----------
    obv = (np.sign(c.diff()).fillna(0.0) * v).cumsum()
    out["obv_slope_20"] = safe_div(ols_slope(obv, 20), medvol_60)
    price_slope_20 = ols_slope(np.log(c), 20)
    out["price_slope_20"] = price_slope_20
    out["obv_div_20"] = zscore(out["obv_slope_20"], 120) - zscore(price_slope_20, 120)

    rng = (h - l).replace(0, np.nan)
    ad = (v * (((c - l) - (h - c)) / rng)).fillna(0.0).cumsum()
    out["ad_slope_20"] = safe_div(ols_slope(ad, 20), medvol_60)
    out["chaikin_osc"] = safe_div(
        ad.ewm(span=3, adjust=False).mean() - ad.ewm(span=10, adjust=False).mean(), medvol_60
    )

    tp = (h + l + c) / 3.0
    flow = tp * v
    pos_flow = flow.where(tp > tp.shift(1), 0.0).rolling(14, min_periods=14).sum()
    neg_flow = flow.where(tp < tp.shift(1), 0.0).rolling(14, min_periods=14).sum()
    out["mfi_14"] = 100 - 100 / (1 + safe_div(pos_flow, neg_flow))

    # Anchored VWAP from the 120-day low: are base buyers in profit?
    out["px_vs_avwap"] = _anchored_vwap_ratio(tp, v, l, window=120)

    vwap20 = safe_div(
        (tp * v).rolling(20, min_periods=20).sum(), v.rolling(20, min_periods=20).sum()
    )
    out["px_vs_vwap20"] = c / vwap20 - 1
    out["close_strength_20"] = ((c - l) / rng).rolling(20, min_periods=20).mean()

    # ---------- structure ----------
    out["dist_from_high_60"] = c / c.rolling(60, min_periods=60).max() - 1
    out["dist_from_high_252"] = c / c.rolling(252, min_periods=120).max() - 1
    out["ret_20"] = c / c.shift(20) - 1
    out["ret_60"] = c / c.shift(60) - 1

    base_hi = h.rolling(60, min_periods=60).max()
    base_lo = l.rolling(60, min_periods=60).min()
    out["base_depth_60"] = safe_div(base_hi - base_lo, base_lo)
    out["base_pos_60"] = safe_div(c - base_lo, base_hi - base_lo)
    out["higher_low_20"] = (
        l.rolling(20, min_periods=20).min() > l.shift(20).rolling(20, min_periods=20).min()
    ).astype(float)

    sma10 = c.rolling(10, min_periods=10).mean()
    sma20 = c.rolling(20, min_periods=20).mean()
    # Emitted, not just used locally: exits.py reads bar.sma20 for the
    # thesis-break rule. It used to be computed in scripts/run_backtest.py, so
    # anything else building Bars from this panel -- the live loop, the position
    # view -- silently got sma20=0, which disables thesis-break entirely
    # (the rule is guarded by `bar.sma20 > 0`). In the backtest that exit
    # accounted for -$2,908 over 83 trades, so losing it live is not cosmetic.
    # NOTE: this is the SPLIT-ADJUSTED mean while Bar.close is as-traded. The
    # two agree whenever adj_factor is 1, which is every recent bar, but they
    # do not across a split. Left as-is deliberately: changing the basis moves
    # the Phase 3 numbers and needs its own backtest run, not a quiet edit.
    out["sma20"] = sma20
    sma50 = c.rolling(50, min_periods=50).mean()
    sma60 = c.rolling(60, min_periods=60).mean()
    sma120 = c.rolling(120, min_periods=100).mean()
    sma200 = c.rolling(200, min_periods=150).mean()
    out["ma_stack"] = ((c > sma20) & (sma20 > sma20.shift(10))).astype(float)
    out["sma50_rising"] = (sma50 > sma50.shift(20)).astype(float)
    # Informational only -- most penny bases sit below the 200MA, so gating on
    # this would delete the entire opportunity set.
    out["px_vs_sma200"] = c / sma200 - 1
    out["px_vs_sma120"] = c / sma120 - 1

    # ---------- moving-average crossovers and long-term breakouts ----------
    # A golden cross and a reclaim of the long-term average are the two things
    # visible on the reference chart right before the vertical move. Measured,
    # not assumed: research/evaluate.py decides whether they add anything over
    # the trend features already present, several of which they correlate with.
    out["golden_cross_20_60"] = (sma20 > sma60).astype(float)
    out["gc_age_20_60"] = _event_age(sma20 > sma60, cap=120)
    out["gc_fresh_20_60"] = (out["gc_age_20_60"] <= 10).astype(float)

    out["ma_align_score"] = (
        (c > sma10).astype(float)
        + (sma10 > sma20).astype(float)
        + (sma20 > sma60).astype(float)
        + (sma60 > sma120).astype(float)
    )

    out["above_sma200"] = (c > sma200).astype(float)
    out["brk200_age"] = _event_age(c > sma200, cap=120)
    out["brk200_fresh"] = (out["brk200_age"] <= 10).astype(float)
    out["above_sma120"] = (c > sma120).astype(float)
    out["brk120_age"] = _event_age(c > sma120, cap=120)
    out["brk120_fresh"] = (out["brk120_age"] <= 10).astype(float)

    out["sma200_slope"] = safe_div(sma200 - sma200.shift(20), sma200)
    out["sma120_slope"] = safe_div(sma120 - sma120.shift(20), sma120)

    # Volume-confirmed versions. A breakout on no volume is noise; the thesis
    # is specifically that volume arrives with the move. Multiplicative so a
    # quiet breakout cannot score on the breakout leg alone.
    rvol_term = out["rvol_10"].clip(lower=0, upper=5).fillna(1.0)
    out["gc_vol_confirmed"] = out["gc_fresh_20_60"] * rvol_term
    out["brk200_vol_confirmed"] = out["brk200_fresh"] * rvol_term
    out["brk120_vol_confirmed"] = out["brk120_fresh"] * rvol_term
    out["align_vol_confirmed"] = (out["ma_align_score"] / 4.0) * rvol_term

    # ---------- liquidity / microstructure ----------
    out["gap_freq_60"] = (
        (o / c.shift(1) - 1).abs().gt(0.05).rolling(60, min_periods=60).mean()
    )
    out["amihud_20"] = (
        safe_div(logret.abs(), pd.Series(out["dollar_volume"].values, index=df.index))
        .rolling(20, min_periods=20)
        .mean()
        * 1e6
    )
    return out


def _event_age(condition: pd.Series, cap: int = 120) -> pd.Series:
    """Sessions since `condition` last turned False->True, capped.

    Trailing-only: the value at T counts backwards from T and never inspects a
    later bar. NaN until the condition has been observable.
    """
    cond = condition.fillna(False).to_numpy(bool)
    n = len(cond)
    age = np.full(n, np.nan)
    since = None
    for i in range(n):
        if cond[i]:
            if i == 0 or not cond[i - 1]:
                since = i
            age[i] = min(i - since, cap)
        else:
            since = None
            age[i] = cap
    return pd.Series(age, index=condition.index)


def _anchored_vwap_ratio(tp: pd.Series, v: pd.Series, low: pd.Series, window: int) -> pd.Series:
    """Price vs VWAP anchored at the rolling-window low, computed trailing-only."""
    n = len(tp)
    tp_v = (tp * v).values
    vv = v.values
    lows = low.values
    cum_tpv = np.concatenate([[0.0], np.nancumsum(tp_v)])
    cum_v = np.concatenate([[0.0], np.nancumsum(vv)])
    out = np.full(n, np.nan)
    for i in range(n):
        if i < window - 1:
            continue
        lo = i - window + 1
        anchor = lo + int(np.nanargmin(lows[lo : i + 1]))
        vol = cum_v[i + 1] - cum_v[anchor]
        if vol <= EPS:
            continue
        avwap = (cum_tpv[i + 1] - cum_tpv[anchor]) / vol
        if avwap > EPS:
            out[i] = tp.values[i] / avwap - 1
    return pd.Series(out, index=tp.index)


def build_features(bars: pd.DataFrame, min_bars: int = 260) -> pd.DataFrame:
    """Feature panel for all symbols in a long-format bar frame."""
    frames = []
    for sym, g in bars.sort_values(["symbol", "dt"]).groupby("symbol", sort=False):
        if len(g) < min_bars:
            continue
        frames.append(compute_symbol_features(g.reset_index(drop=True)))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    cols = ["symbol", "dt"] + [c for c in out.columns if c not in ("symbol", "dt")]
    return out[cols]
