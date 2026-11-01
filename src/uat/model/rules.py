"""Rule-based composite scorer, with weights derived from the lift table.

Weights are NOT intuition. `scripts/run_lift.py` measures each feature's
top/bottom decile hit rate against the base rate with Wilson intervals, and the
direction column there decides which tail each feature scores. A feature whose
interval does not clear 1.0 gets no weight.

Two scorers are provided deliberately:

* THESIS  -- the user's original "quiet accumulation" formulation (volatility
  compression + rising volume). Kept as a measurable baseline so the thesis can
  be tracked rather than quietly abandoned.
* EVIDENCE -- what the full-universe lift table actually supports.

Both run in every evaluation. If EVIDENCE cannot beat the shuffled control, no
execution code gets written; if THESIS beats EVIDENCE on better data later, the
weights move. Scoring is cross-sectional within each date, which is the decision
the scanner actually makes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# (feature, direction, weight). direction 'high' scores the top tail, 'low' the
# bottom tail. Weights sum to 100 within each scorer.
THESIS_WEIGHTS: list[tuple[str, str, float]] = [
    ("vol_price_div", "high", 25.0),
    ("vol_regime", "high", 20.0),
    ("bbw_pct_252", "low", 15.0),      # compression -- the contested half
    ("atr_ratio", "low", 10.0),
    ("obv_div_20", "high", 10.0),
    ("evr_pct_252", "high", 10.0),
    ("base_pos_60", "high", 5.0),
    ("udvr_20", "high", 5.0),
]

EVIDENCE_WEIGHTS: list[tuple[str, str, float]] = [
    ("gap_freq_60", "high", 20.0),
    ("base_depth_60", "high", 18.0),
    ("dist_from_high_252", "low", 17.0),
    ("amihud_20", "high", 12.0),
    ("rvol_20", "high", 10.0),
    ("rvol_10", "high", 8.0),
    ("vol_regime", "high", 7.0),
    ("obv_slope_20", "high", 5.0),
    ("vol_slope_20", "high", 3.0),
]

# The user's chart-reading hypothesis stated as a scorer: buy the confirmed
# move -- a fresh 20/60 golden cross and a reclaim of the long-term average,
# with volume behind it.
#
# The lift table rates these features in the OPPOSITE direction (no golden
# cross 17.05% vs golden cross 4.36% against a 4.89% base), but that is
# mechanical: a breakout is confirmed partway into the move, so less of the
# +60% target remains. Lift alone cannot settle it, because buying confirmation
# should trade hit rate against payoff. Only a backtest with real exits can,
# which is why this is a first-class scorer rather than a footnote.
BREAKOUT_WEIGHTS: list[tuple[str, str, float]] = [
    ("brk200_vol_confirmed", "high", 20.0),
    ("gc_vol_confirmed", "high", 18.0),
    ("brk120_vol_confirmed", "high", 15.0),
    ("ma_align_score", "high", 12.0),
    ("rvol_10", "high", 12.0),
    ("vol_slope_20", "high", 8.0),
    ("sma200_slope", "high", 7.0),
    ("px_vs_sma200", "high", 5.0),
    ("obv_slope_20", "high", 3.0),
]

SCORERS = {
    "evidence": EVIDENCE_WEIGHTS,
    "thesis": THESIS_WEIGHTS,
    "breakout": BREAKOUT_WEIGHTS,
}

# Hard gates applied before scoring. Deliberately few: the universe gate in
# universe/filters.py already handles liquidity, price band and listing status,
# and every extra gate here is another degree of freedom to overfit.
GATES: list[tuple[str, str, float]] = [
    ("mfi_14", "max", 85.0),        # already extended
    ("ret_20", "max", 2.00),        # already tripled in a month: too late
]


def _cross_sectional_rank(df: pd.DataFrame, col: str, direction: str) -> pd.Series:
    """Percentile rank within each date, oriented so higher is always better."""
    r = df.groupby("dt")[col].rank(pct=True, method="average")
    return r if direction == "high" else 1.0 - r


def apply_gates(df: pd.DataFrame, gates: list[tuple[str, str, float]] | None = None) -> pd.Series:
    """Boolean mask of rows that pass every hard gate."""
    gates = GATES if gates is None else gates
    ok = pd.Series(True, index=df.index)
    for col, kind, value in gates:
        if col not in df.columns:
            continue
        series = df[col]
        passes = series.le(value) if kind == "max" else series.ge(value)
        ok &= passes.fillna(False)
    return ok


def score(
    df: pd.DataFrame,
    weights: list[tuple[str, str, float]] | None = None,
    use_gates: bool = True,
) -> pd.DataFrame:
    """Return df + `score` (0-100) and `score_components`.

    Rows failing a gate get score NaN so they can never be ranked into the top-K.
    """
    weights = EVIDENCE_WEIGHTS if weights is None else weights
    available = [(c, d, w) for c, d, w in weights if c in df.columns]
    if not available:
        raise ValueError("none of the scoring features are present in the frame")

    total_w = sum(w for _, _, w in available)
    out = df.copy()
    acc = pd.Series(0.0, index=df.index)
    components: dict[str, pd.Series] = {}

    for col, direction, w in available:
        r = _cross_sectional_rank(out, col, direction)
        components[col] = r
        acc = acc.add(r.fillna(0.5) * w, fill_value=0.0)

    out["score"] = 100.0 * acc / total_w
    for col, series in components.items():
        out[f"cmp_{col}"] = series

    if use_gates:
        out.loc[~apply_gates(out), "score"] = np.nan
    return out


def top_k(df: pd.DataFrame, k: int = 15, max_per_symbol_day: bool = True) -> pd.DataFrame:
    """Daily top-K by score, highest first."""
    d = df.dropna(subset=["score"])
    return (
        d.sort_values(["dt", "score"], ascending=[True, False])
        .groupby("dt", group_keys=False)
        .head(k)
        .reset_index(drop=True)
    )
