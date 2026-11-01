"""EOD scan: rank the universe and keep only what this account can actually buy.

For a plumbing-test account the affordability filter matters more than the
ranking. At $8.91 most of the universe is unbuyable -- a $13 stock means zero
shares -- so the scanner reports both the ideal ranking and the subset that
survives the cash constraint, and never silently substitutes one for the other.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from uat.config import load_settings
from uat.data.ingest_daily import load_bars
from uat.features.pipeline import build_features
from uat.logging import get
from uat.model.rules import SCORERS, score, top_k
from uat.universe.filters import bar_level_metrics, classify_tier

log = get("live.scanner")


@dataclass
class Candidate:
    symbol: str
    rank: int
    score: float
    close: float
    addv_20: float
    atr14: float
    medvol_20: float
    affordable_shares: int
    notional: float
    spread_est: float
    tier: str

    @property
    def cost_drag_pct(self) -> float:
        """Round-trip spread plus commission as a share of the position.

        At $0.31 a single cent of tick is 3.2%, so this is the number that
        decides whether a name is worth touching at all with a tiny account.
        """
        return self.spread_est + 0.005


def scan(
    asof: date | None = None,
    cash_usd: float = 0.0,
    scorer: str = "evidence",
    k: int = 15,
    lookback_days: int = 420,
) -> tuple[list[Candidate], pd.DataFrame]:
    """Return (affordable candidates, full ranked frame) for the given session."""
    s = load_settings()
    bars = load_bars()
    if bars.empty:
        raise RuntimeError("no bars in the database -- run uat.data.ingest_daily first")

    latest = asof or max(bars["dt"])
    start = pd.Timestamp(latest) - pd.Timedelta(days=lookback_days)
    bars = bars[pd.to_datetime(bars["dt"]) >= start]

    tiered = classify_tier(bar_level_metrics(bars), settings=s)
    gate = tiered.loc[
        tiered["tier"].isin(["core", "watch"]), ["symbol", "dt", "tier", "cs_spread_20"]
    ]

    feats = build_features(bars, min_bars=220)
    panel = feats.merge(gate, on=["symbol", "dt"], how="inner")
    day = panel[panel["dt"] == latest]
    if day.empty:
        raise RuntimeError(f"no gated rows for {latest}")

    ranked = top_k(score(day, SCORERS[scorer]), k=k)
    log.info(
        "scan complete",
        extra={"extra_fields": {"asof": str(latest), "universe": len(day),
                                "ranked": len(ranked)}},
    )

    out: list[Candidate] = []
    for i, r in enumerate(ranked.itertuples(index=False), start=1):
        px = float(r.close_raw)
        shares = int(cash_usd // px) if px > 0 else 0
        spread = float(getattr(r, "cs_spread_20", 0.0) or 0.0)
        tick_floor = 0.01 / px if px > 0 else 0.0
        out.append(
            Candidate(
                symbol=r.symbol,
                rank=i,
                score=float(r.score),
                close=px,
                addv_20=float(getattr(r, "addv_20", 0.0) or 0.0),
                atr14=float(getattr(r, "atr14", 0.0) or 0.0),
                medvol_20=float(getattr(r, "medvol_20", 0.0) or 0.0),
                affordable_shares=shares,
                notional=shares * px,
                spread_est=max(spread, tick_floor),
                tier=str(getattr(r, "tier", "")),
            )
        )
    return out, ranked


def affordable(cands: list[Candidate], min_shares: int = 1) -> list[Candidate]:
    return [c for c in cands if c.affordable_shares >= min_shares]
