"""Daily OHLCV ingest into raw.ohlcv_daily.

Reports an ingest success rate, which feeds the data-quality circuit breaker:
if too much of the universe fails to download, the system must refuse to trade
rather than act on a partial picture.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta

import pandas as pd

from uat.config import load_settings
from uat.data.base import PriceSource
from uat.data.yf_source import YFinanceSource
from uat.logging import get
from uat.storage.db import query, upsert_df

log = get("data.ingest")


def get_source(name: str = "yfinance") -> PriceSource:
    if name == "yfinance":
        return YFinanceSource()
    raise ValueError(f"unknown price source {name!r}")


def ingest_daily(
    symbols: list[str],
    start: date,
    end: date,
    source: PriceSource | None = None,
    chunk: int = 400,
) -> dict:
    """Download and upsert daily bars. Returns an ingest report."""
    src = source or get_source()
    total_rows = 0
    got: set[str] = set()

    for i in range(0, len(symbols), chunk):
        part = symbols[i : i + chunk]
        bars = src.daily_bars(part, start, end)
        if bars.empty:
            continue
        bars = bars.copy()
        bars["ingested_at"] = datetime.now()
        total_rows += upsert_df(
            "raw.ohlcv_daily",
            bars[
                [
                    "symbol",
                    "dt",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "adj_factor",
                    "ingested_at",
                ]
            ],
            ["symbol", "dt"],
        )
        got.update(bars["symbol"].unique().tolist())
        log.info(
            "ingest progress",
            extra={
                "extra_fields": {
                    "done": min(i + chunk, len(symbols)),
                    "of": len(symbols),
                    "rows": total_rows,
                }
            },
        )

    rate = len(got) / len(symbols) if symbols else 0.0
    report = {
        "requested": len(symbols),
        "with_data": len(got),
        "missing": len(symbols) - len(got),
        "rows": total_rows,
        "success_rate": round(rate, 4),
    }
    threshold = float(load_settings().get("risk.min_universe_ingest_rate"))
    report["below_quality_threshold"] = rate < threshold
    if report["below_quality_threshold"]:
        log.error("ingest below data-quality threshold", extra={"extra_fields": report})
    else:
        log.info("ingest complete", extra={"extra_fields": report})
    return report


def ingest_corporate_actions(symbols: list[str], source: PriceSource | None = None) -> int:
    src = source or get_source()
    ca = src.corporate_actions(symbols)
    if ca.empty:
        return 0
    return upsert_df("raw.corporate_actions", ca, ["symbol", "dt", "kind"])


def load_bars(
    start: date | None = None,
    end: date | None = None,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    """Read as-traded bars back out of DuckDB, long format, sorted."""
    where, params = [], []
    if start:
        where.append("dt >= ?")
        params.append(start)
    if end:
        where.append("dt <= ?")
        params.append(end)
    if symbols:
        placeholders = ", ".join("?" * len(symbols))
        where.append(f"symbol IN ({placeholders})")
        params.extend(symbols)
    clause = f"WHERE {' AND '.join(where)}" if where else ""
    df = query(
        f"""
        SELECT symbol, dt, open, high, low, close, volume, adj_factor
        FROM raw.ohlcv_daily {clause}
        ORDER BY symbol, dt
        """,
        params,
    )
    if not df.empty:
        df["dt"] = pd.to_datetime(df["dt"]).dt.date
    return df


def main() -> None:
    from uat.universe.build import build_tradeable_symbols

    ap = argparse.ArgumentParser(description="Ingest daily OHLCV")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--limit", type=int, default=None, help="cap symbol count (testing)")
    ap.add_argument("--symbols", default=None, help="comma-separated override")
    args = ap.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) if args.end else date.today() + timedelta(days=1)

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        symbols, _ = build_tradeable_symbols(refresh=False)
    if args.limit:
        symbols = symbols[: args.limit]

    report = ingest_daily(symbols, start, end)
    for k, v in report.items():
        print(f"{k:>26}: {v}")


if __name__ == "__main__":
    main()
