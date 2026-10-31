"""Out-of-sample check: derive weights on train, score on test, never peek.

The EVIDENCE weights in model/rules.py were read off a lift table computed on
the whole panel, which makes them in-sample by construction -- including the
human-in-the-loop part where a person picked which features to include. This
script removes the human: it recomputes the lift table on the training window
only, auto-derives weights from it, and scores the untouched test window.

If the out-of-sample lift collapses toward the shuffled control, the signal is
an artefact and Phase 3 does not start.
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from uat.config import project_root  # noqa: E402
from uat.model.rules import THESIS_WEIGHTS, score  # noqa: E402
from uat.research.evaluate import (  # noqa: E402
    feature_summary,
    precision_at_k,
    shuffled_control,
)

CANDIDATES = [
    "bbw_pct_252", "bbw_pct_126", "atr_ratio", "atrp_pct_252", "donch_ratio",
    "cc_vol_ratio", "nr7_count_20", "inside_count_20",
    "rvol_5", "rvol_10", "rvol_20", "vol_regime", "vol_slope_20", "vol_z_ma10",
    "udvr_20", "evr_pct_252", "vol_price_div",
    "obv_slope_20", "obv_div_20", "ad_slope_20", "chaikin_osc", "mfi_14",
    "px_vs_avwap", "px_vs_vwap20", "close_strength_20",
    "dist_from_high_60", "dist_from_high_252", "ret_20", "ret_60",
    "base_depth_60", "base_pos_60", "higher_low_20", "ma_stack", "sma50_rising",
    "gap_freq_60", "amihud_20", "addv_20",
]


def derive_weights(train: pd.DataFrame, top_n: int = 9, min_lift: float = 1.25):
    """Auto-derive (feature, direction, weight) from the TRAIN lift table only."""
    summary = feature_summary(train, [c for c in CANDIDATES if c in train.columns])
    if summary.empty:
        return []
    picks = []
    for _, r in summary.iterrows():
        if r.direction == "high":
            lift, direction = r.top_lift, "high"
        elif r.direction == "low":
            lift, direction = r.bottom_lift, "low"
        elif r.direction == "both_tails":
            if r.top_lift >= r.bottom_lift:
                lift, direction = r.top_lift, "high"
            else:
                lift, direction = r.bottom_lift, "low"
        else:
            continue
        if lift < min_lift:
            continue
        picks.append((r.feature, direction, float(lift - 1.0)))
    picks.sort(key=lambda x: -x[2])
    picks = picks[:top_n]
    total = sum(w for _, _, w in picks) or 1.0
    return [(f, d, 100.0 * w / total) for f, d, w in picks]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="2024-01-01")
    ap.add_argument("--k", type=int, default=15)
    args = ap.parse_args()

    panel = pd.read_parquet(project_root() / "reports" / "panel.parquet")
    panel["dt"] = pd.to_datetime(panel["dt"])
    split = pd.Timestamp(args.split)

    # Embargo the horizon so training labels cannot overlap the test window.
    embargo_end = split - pd.Timedelta(days=30)
    train = panel[panel["dt"] < embargo_end]
    test = panel[panel["dt"] >= split]

    print(f"train: {len(train):,} rows  {train.dt.min().date()} .. {train.dt.max().date()}")
    print(f"test : {len(test):,} rows  {test.dt.min().date()} .. {test.dt.max().date()}")
    print(f"(30-day embargo between them; label horizon is 20 sessions)")
    print()
    print(f"train base rate: {train.y_explode.mean()*100:.2f}%")
    print(f"test  base rate: {test.y_explode.mean()*100:.2f}%")
    print()

    weights = derive_weights(train)
    print("weights auto-derived from TRAIN ONLY:")
    for f, d, w in weights:
        print(f"  {f:<22} {d:<5} {w:5.1f}")
    print()

    ctrl = shuffled_control(test, top_k=args.k)
    print(f"shuffled control on TEST: mean {ctrl['mean_rate']*100:.2f}%  "
          f"p95 {ctrl['p95']*100:.2f}%")
    print()

    print(f"{'scorer':<24}{'precision@%d' % args.k:>13}{'lift':>8}{'95% CI':>16}  verdict")
    print("-" * 74)
    rows = [
        ("THESIS (compression)", THESIS_WEIGHTS),
        ("EVIDENCE (train-derived)", weights),
    ]
    for name, w in rows:
        if not w:
            continue
        scored = score(test, w)
        r = precision_at_k(scored, "score", k=args.k)
        beats = r["precision"] > ctrl["p95"]
        print(
            f"{name:<24}{r['precision']*100:>12.2f}%{r['lift']:>8.2f}"
            f"  [{r['lift_lo']:.2f}, {r['lift_hi']:.2f}]   "
            f"{'HOLDS out-of-sample' if beats else 'FAILS out-of-sample'}"
        )

    print()
    print("tradeability of what the train-derived scorer picks on TEST:")
    scored = score(test, weights)
    picked = (
        scored.dropna(subset=["score"])
        .sort_values(["dt", "score"], ascending=[True, False])
        .groupby("dt")
        .head(args.k)
    )
    print(f"  median close      ${picked.close_raw.median():.2f}")
    print(f"  median ADDV       ${picked.addv_20.median():,.0f}")
    print(f"  median 60d range  {picked.base_depth_60.median()*100:.0f}%")
    print(f"  median MFE        {picked.mfe.median()*100:.1f}%")
    print(f"  median MAE        {picked.mae.median()*100:.1f}%")
    print(f"  (universe medians: ${test.close_raw.median():.2f} / "
          f"${test.addv_20.median():,.0f} / {test.base_depth_60.median()*100:.0f}%)")


if __name__ == "__main__":
    main()
