"""Phase 2 decision gate: measure every candidate feature against the base rate.

Usage:
    python scripts/run_lift.py --start 2019-01-01 --max-symbols 1200

Writes reports/lift_table.csv and prints the summary that decides whether the
compression half of the thesis survives.
"""
from __future__ import annotations

import argparse
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from uat.config import load_settings, project_root  # noqa: E402
from uat.data.ingest_daily import load_bars  # noqa: E402
from uat.features.pipeline import build_features  # noqa: E402
from uat.labeling.triple_barrier import (  # noqa: E402
    build_labels,
    effective_sample_size,
)
from uat.research.evaluate import (  # noqa: E402
    decile_lift,
    feature_summary,
    format_summary,
    mfe_by_decile,
    shuffled_control,
)
from uat.universe.filters import bar_level_metrics, classify_tier  # noqa: E402

CANDIDATE_FEATURES = [
    # volatility -- compression hypothesis AND its inverse, both measured
    "bbw_pct_252",
    "bbw_pct_126",
    "atr_ratio",
    "atrp_pct_252",
    "donch_ratio",
    "cc_vol_ratio",
    "nr7_count_20",
    "inside_count_20",
    # volume expansion
    "rvol_5",
    "rvol_10",
    "rvol_20",
    "vol_regime",
    "vol_slope_20",
    "vol_z_ma10",
    "udvr_20",
    "evr_pct_252",
    "vol_price_div",
    # accumulation
    "obv_slope_20",
    "obv_div_20",
    "ad_slope_20",
    "chaikin_osc",
    "mfi_14",
    "px_vs_avwap",
    "px_vs_vwap20",
    "close_strength_20",
    # structure
    "dist_from_high_60",
    "dist_from_high_252",
    "ret_20",
    "ret_60",
    "base_depth_60",
    "base_pos_60",
    "higher_low_20",
    "ma_stack",
    "sma50_rising",
    "px_vs_sma200",
    # liquidity
    "gap_freq_60",
    "amihud_20",
    "addv_20",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--max-symbols", type=int, default=None)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    s = load_settings()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) if args.end else None

    print("loading bars ...")
    bars = load_bars(start=start, end=end)
    print(f"  {len(bars):,} bars / {bars.symbol.nunique():,} symbols")

    # Universe gate first: only score names we could actually have traded.
    print("applying universe gate ...")
    metrics = bar_level_metrics(bars)
    tiered = classify_tier(metrics, settings=s)
    tradeable = tiered[tiered["tier"].isin(["core", "watch"])]
    keep_syms = tradeable["symbol"].unique().tolist()
    print(f"  symbols with >=1 tradeable session: {len(keep_syms):,}")
    print(f"  tradeable (symbol,date) pairs: {len(tradeable):,}")

    if args.max_symbols and len(keep_syms) > args.max_symbols:
        rng = np.random.default_rng(args.seed)
        keep_syms = sorted(rng.choice(keep_syms, args.max_symbols, replace=False).tolist())
        print(f"  sampled down to {len(keep_syms):,} symbols")

    sub = bars[bars.symbol.isin(keep_syms)]

    print("building features ...")
    feats = build_features(sub)
    print(f"  {feats.shape[0]:,} rows x {feats.shape[1]} cols")

    print("labelling (triple barrier, T+1 anchored) ...")
    labels = build_labels(
        sub,
        horizon=int(s.get("labeling.horizon_days")),
        upper=float(s.get("labeling.upper_barrier")),
        lower=float(s.get("labeling.lower_barrier")),
        gap_up_exclude=float(s.get("labeling.exclude_gap_up_above")),
        gap_down_exclude=float(s.get("labeling.exclude_gap_down_below")),
    )

    panel = feats.merge(labels, on=["symbol", "dt"], how="inner")
    # Restrict to sessions that passed the universe gate at that date.
    gate = tradeable[["symbol", "dt"]].drop_duplicates()
    panel = panel.merge(gate, on=["symbol", "dt"], how="inner")
    panel = panel[~panel["excluded"]].dropna(subset=["y_explode"])
    print(f"  labelled, gated panel: {len(panel):,} rows")

    ess = effective_sample_size(panel)
    base_rate = panel["y_explode"].mean()
    print()
    print("=" * 74)
    print(f"base rate P(+{s.get('labeling.upper_barrier'):.0%} before "
          f"{s.get('labeling.lower_barrier'):.0%} in {s.get('labeling.horizon_days')}d) "
          f"= {base_rate*100:.2f}%")
    print(f"nominal positives      : {ess['nominal_positives']:,}")
    print(f"symbol episodes        : {ess['symbol_episodes']:,}  "
          f"(inflation {ess['inflation_factor']}x)")
    print(f"calendar clusters      : {ess['calendar_clusters']:,}")
    print("  ^ model complexity is bounded by the episode count, not the row count")
    print("=" * 74)
    print()

    available = [f for f in CANDIDATE_FEATURES if f in panel.columns]
    summary = feature_summary(panel, available)
    print(format_summary(summary, base_rate))

    out_dir = project_root() / "reports"
    out_dir.mkdir(exist_ok=True)
    summary.to_csv(out_dir / "lift_summary.csv", index=False)

    full = pd.concat(
        [decile_lift(panel, f) for f in available if not decile_lift(panel, f).empty],
        ignore_index=True,
    )
    full.to_csv(out_dir / "lift_table.csv", index=False)

    print()
    print("=" * 74)
    print("THE COMPRESSION QUESTION")
    print("=" * 74)
    for f in ("bbw_pct_252", "atrp_pct_252", "atr_ratio", "donch_ratio"):
        row = summary[summary.feature == f]
        if row.empty:
            continue
        r = row.iloc[0]
        verdict = {
            "low": "COMPRESSION wins (thesis supported)",
            "high": "EXPANSION wins (thesis inverted)",
            "both_tails": "both tails beat base",
            "none": "no significant edge",
        }[r.direction]
        print(f"  {f:<16} bottom {r.bottom_lift:>5.2f}x | top {r.top_lift:>5.2f}x  -> {verdict}")

    print()
    print("MFE distribution for the strongest feature:")
    if not summary.empty:
        best = summary.iloc[0].feature
        print(f"  ({best})")
        print(mfe_by_decile(panel, best).to_string(index=False))

    print()
    print("shuffled control (random top-15 from the same universe/day):")
    ctrl = shuffled_control(panel, top_k=15)
    if ctrl:
        print(f"  mean {ctrl['mean_rate']*100:.2f}%  "
              f"[p05 {ctrl['p05']*100:.2f}%, p95 {ctrl['p95']*100:.2f}%]  "
              f"over {ctrl['n_trials']} trials")
        print("  any composite score must clear the p95 of this band to be real.")

    panel.to_parquet(out_dir / "panel.parquet", index=False)
    print()
    print(f"written: {out_dir/'lift_summary.csv'}, {out_dir/'lift_table.csv'}, "
          f"{out_dir/'panel.parquet'}")


if __name__ == "__main__":
    main()
