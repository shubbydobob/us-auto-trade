"""Build the point-in-time universe and persist it to dim.universe_snapshot.

Two call sites with different needs:

* ``build_tradeable_symbols()`` -- today's listing-level pass, used by the daily
  ingest job to decide which tickers to even download.
* ``build_snapshots(bars, meta)`` -- the bar-level pass that assigns core/watch/
  rejected per (symbol, dt). This is the one that must be point-in-time.

Rejects are stored alongside keeps. The universe silently shrinking is a common
way to misread a backtest, and the reason column makes that visible.
"""
from __future__ import annotations

import argparse
from datetime import date

import pandas as pd

from uat.config import load_settings
from uat.data.nasdaq_trader import fetch_symbol_directory, store_snapshot
from uat.logging import get
from uat.storage.db import query, upsert_df
from uat.universe.filters import (
    bar_level_metrics,
    classify_tier,
    derived_min_addv,
    filter_listings,
)

log = get("universe.build")


def build_tradeable_symbols(
    refresh: bool = True, asof: date | None = None
) -> tuple[list[str], pd.DataFrame]:
    """Listing-level pass. Returns (symbols, annotated directory)."""
    asof = asof or date.today()
    meta = fetch_symbol_directory(asof) if refresh else _load_directory(asof)
    annotated = filter_listings(meta)
    if refresh:
        store_snapshot(meta, asof)
    kept = annotated[annotated["reject_reason"].isna()]
    log.info(
        "listing filter applied",
        extra={
            "extra_fields": {
                "asof": str(asof),
                "total": len(annotated),
                "kept": len(kept),
                "rejects": annotated["reject_reason"].value_counts().to_dict(),
            }
        },
    )
    return kept["symbol"].tolist(), annotated


def _load_directory(asof: date) -> pd.DataFrame:
    df = query(
        """
        SELECT symbol, exchange, security_name, is_etf, is_test,
               financial_status, round_lot, source
        FROM raw.symbol_directory
        WHERE as_of = (SELECT max(as_of) FROM raw.symbol_directory WHERE as_of <= ?)
        """,
        [asof],
    )
    if df.empty:
        raise RuntimeError("no stored symbol directory; run with refresh=True first")
    return df


def current_min_addv(settings=None) -> float:
    """Derive the dollar-volume floor from the account's target position size."""
    s = settings or load_settings()
    equity = float(s.get("risk.equity_usd_override") or 0) or 10_000.0
    target_notional = equity * float(s.get("risk.max_notional_pct_equity"))
    return derived_min_addv(
        target_position_notional=target_notional,
        participation_cap=float(s.get("risk.max_participation")),
        entry_days=int(s.get("entry.bt_carry_sessions")) + 1,
        floor=float(s.get("universe.min_addv_floor")),
    )


def build_snapshots(
    bars: pd.DataFrame, meta: pd.DataFrame | None = None, settings=None
) -> pd.DataFrame:
    """Bar-level pass -> one row per (symbol, dt) with tier + reject_reason."""
    s = settings or load_settings()
    if bars.empty:
        return pd.DataFrame(columns=["as_of", "symbol", "tier", "reject_reason"])

    metrics = bar_level_metrics(bars)

    if meta is not None:
        listing = filter_listings(meta, s)[["symbol", "reject_reason"]].rename(
            columns={"reject_reason": "listing_reject"}
        )
        metrics = metrics.merge(listing, on="symbol", how="left")
    else:
        metrics["listing_reject"] = pd.NA

    tiered = classify_tier(metrics, min_addv=current_min_addv(s), settings=s)

    # A listing-level rejection outranks any bar-level reason.
    listing_bad = tiered["listing_reject"].notna()
    tiered.loc[listing_bad, "reject_reason"] = tiered.loc[listing_bad, "listing_reject"]
    tiered.loc[listing_bad, "tier"] = "rejected"

    out = tiered[["dt", "symbol", "tier", "reject_reason"]].rename(columns={"dt": "as_of"})
    return out.reset_index(drop=True)


def persist_snapshots(snapshots: pd.DataFrame) -> int:
    return upsert_df("dim.universe_snapshot", snapshots, ["as_of", "symbol"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the tradeable universe")
    ap.add_argument("--listings-only", action="store_true")
    args = ap.parse_args()

    symbols, annotated = build_tradeable_symbols(refresh=True)
    print(f"KIS-tradeable common stock: {len(symbols)}")
    print(annotated["reject_reason"].value_counts(dropna=False).to_string())
    print()
    print(f"derived min ADDV floor: ${current_min_addv():,.0f}")


if __name__ == "__main__":
    main()
