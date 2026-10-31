"""Signal validation lab -- the Phase 2 decision gate.

This module exists because a probe on real data contradicted half the original
thesis: volume expansion predicted explosions, volatility COMPRESSION predicted
the opposite, and the composite signal that demanded compression scored worse
than the base rate. Rather than hard-code either direction, every candidate
feature is measured against the base rate with a confidence interval, and the
scoring weights in model/rules.py are derived from that table.

If a feature cannot beat the base rate with its interval clear of 1.0, it does
not get a weight. If the composite cannot beat a random pick from the same
universe on the same days, the strategy does not proceed to execution code.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from uat.logging import get

log = get("research.evaluate")


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval -- correct for the small, low-rate counts here.

    A normal-approximation interval on a 1.8% base rate with a few hundred
    samples produces nonsense (often negative lower bounds).
    """
    if trials == 0:
        return (np.nan, np.nan)
    p = successes / trials
    denom = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denom
    margin = z * np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


@dataclass
class LiftRow:
    feature: str
    bucket: str
    n: int
    hits: int
    rate: float
    lift: float
    lift_lo: float
    lift_hi: float
    significant: bool


def decile_lift(
    df: pd.DataFrame,
    feature: str,
    target: str = "y_explode",
    n_buckets: int = 10,
) -> pd.DataFrame:
    """Hit rate by feature decile, with Wilson intervals expressed as lift.

    Deciles are computed cross-sectionally WITHIN each date, so the comparison is
    'best names today vs the rest today' -- which is the decision the scanner
    actually makes -- rather than being dominated by market-wide regime shifts.
    """
    d = df[[feature, target, "dt"]].dropna()
    if d.empty:
        return pd.DataFrame()

    base_rate = d[target].mean()
    if base_rate <= 0:
        return pd.DataFrame()

    ranks = d.groupby("dt")[feature].rank(pct=True, method="average")
    buckets = np.ceil(ranks * n_buckets).clip(1, n_buckets).astype(int)

    rows: list[LiftRow] = []
    for b in range(1, n_buckets + 1):
        mask = buckets == b
        n = int(mask.sum())
        if n == 0:
            continue
        hits = int(d.loc[mask, target].sum())
        rate = hits / n
        lo, hi = wilson_interval(hits, n)
        rows.append(
            LiftRow(
                feature=feature,
                bucket=f"D{b}",
                n=n,
                hits=hits,
                rate=rate,
                lift=rate / base_rate,
                lift_lo=lo / base_rate,
                lift_hi=hi / base_rate,
                significant=bool(lo / base_rate > 1.0 or hi / base_rate < 1.0),
            )
        )
    out = pd.DataFrame([r.__dict__ for r in rows])
    out.attrs["base_rate"] = base_rate
    return out


def feature_summary(
    df: pd.DataFrame, features: list[str], target: str = "y_explode"
) -> pd.DataFrame:
    """One row per feature: top-decile vs bottom-decile lift and its direction.

    `direction` is the key output. It tells model/rules.py which tail to score,
    and it is the column that settles the compression-vs-expansion question.
    """
    rows = []
    base_rate = df[target].dropna().mean()
    for f in features:
        table = decile_lift(df, f, target)
        if table.empty or len(table) < 10:
            continue
        top, bot = table.iloc[-1], table.iloc[0]
        top_sig = top.lift_lo > 1.0
        bot_sig = bot.lift_lo > 1.0
        if top_sig and not bot_sig:
            direction = "high"
        elif bot_sig and not top_sig:
            direction = "low"
        elif top_sig and bot_sig:
            direction = "both_tails"
        else:
            direction = "none"
        rows.append(
            {
                "feature": f,
                "base_rate": base_rate,
                "bottom_decile_rate": bot.rate,
                "top_decile_rate": top.rate,
                "bottom_lift": bot.lift,
                "top_lift": top.lift,
                "bottom_lift_lo": bot.lift_lo,
                "top_lift_lo": top.lift_lo,
                "best_lift": max(top.lift, bot.lift),
                "direction": direction,
                "n_top": top.n,
                "n_bottom": bot.n,
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values("best_lift", ascending=False).reset_index(drop=True) if len(out) else out


def mfe_by_decile(df: pd.DataFrame, feature: str, n_buckets: int = 10) -> pd.DataFrame:
    """MFE distribution per decile.

    This is a fat-tailed strategy, so the mean is close to meaningless -- what
    matters is whether the top decile owns the right tail.
    """
    d = df[[feature, "mfe", "dt"]].dropna()
    if d.empty:
        return pd.DataFrame()
    ranks = d.groupby("dt")[feature].rank(pct=True, method="average")
    d = d.assign(bucket=np.ceil(ranks * n_buckets).clip(1, n_buckets).astype(int))
    return (
        d.groupby("bucket")["mfe"]
        .agg(
            n="size",
            median="median",
            p75=lambda s: s.quantile(0.75),
            p90=lambda s: s.quantile(0.90),
            p99=lambda s: s.quantile(0.99),
            mean="mean",
        )
        .reset_index()
    )


def shuffled_control(
    df: pd.DataFrame, target: str = "y_explode", n_trials: int = 200, top_k: int = 15, seed: int = 7
) -> dict:
    """Random picks from the same universe on the same days.

    The honest null hypothesis. If a signal cannot beat this, it has discovered
    small-cap squeeze beta, not an edge.
    """
    rng = np.random.default_rng(seed)
    d = df[["dt", target]].dropna()
    if d.empty:
        return {}
    rates = []
    for _ in range(n_trials):
        picked = d.groupby("dt", group_keys=False).apply(
            lambda g: g.sample(min(top_k, len(g)), random_state=int(rng.integers(1e9))),
            include_groups=False,
        )
        rates.append(picked[target].mean())
    rates = np.array(rates)
    return {
        "mean_rate": float(rates.mean()),
        "p05": float(np.percentile(rates, 5)),
        "p95": float(np.percentile(rates, 95)),
        "n_trials": n_trials,
        "top_k": top_k,
    }


def precision_at_k(df: pd.DataFrame, score_col: str, target: str = "y_explode", k: int = 15) -> dict:
    """Precision of the daily top-K -- the metric that matches what the scanner does."""
    d = df[["dt", score_col, target]].dropna()
    if d.empty:
        return {}
    picked = (
        d.sort_values([ "dt", score_col], ascending=[True, False])
        .groupby("dt")
        .head(k)
    )
    base = d[target].mean()
    rate = picked[target].mean()
    hits, n = int(picked[target].sum()), len(picked)
    lo, hi = wilson_interval(hits, n)
    return {
        "k": k,
        "n_picked": n,
        "precision": rate,
        "base_rate": base,
        "lift": rate / base if base else np.nan,
        "lift_lo": lo / base if base else np.nan,
        "lift_hi": hi / base if base else np.nan,
        "beats_base": lo > base,
    }


def format_summary(summary: pd.DataFrame, base_rate: float) -> str:
    lines = [
        f"base rate P(target) = {base_rate*100:.2f}%",
        "",
        f"{'feature':<22}{'bot D1':>9}{'top D10':>9}{'bot lift':>10}{'top lift':>10}  direction",
        "-" * 72,
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"{r.feature:<22}{r.bottom_decile_rate*100:>8.2f}%{r.top_decile_rate*100:>8.2f}%"
            f"{r.bottom_lift:>10.2f}{r.top_lift:>10.2f}  {r.direction}"
        )
    return "\n".join(lines)
