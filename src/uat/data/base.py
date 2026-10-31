"""PriceSource protocol -- the single swap point for the data vendor.

Everything downstream depends only on this interface, so moving from yfinance to
Polygon.io means adding one file. ``symbol_meta(asof=...)`` carries an as-of date
specifically so a vendor with delisting history can reconstruct a point-in-time
universe later; yfinance ignores it and returns today's snapshot.

Bar contract (non-negotiable):
    open/high/low/close/volume are AS TRADED on that date -- NOT back-adjusted.
    adj_factor is the cumulative split factor for splits occurring AFTER dt, so

        adjusted_price  = close / adj_factor   (continuous series, for features)
        as_traded_price = close                (price band, dollar volume, fills)

See ``yf_source.reconstruct_as_traded`` for why this matters.
"""
from __future__ import annotations

from datetime import date
from typing import Protocol, runtime_checkable

import pandas as pd

BAR_COLUMNS = ["symbol", "dt", "open", "high", "low", "close", "volume", "adj_factor"]

MINUTE_COLUMNS = ["dt", "open", "high", "low", "close", "volume"]

META_COLUMNS = [
    "symbol",
    "exchange",
    "security_name",
    "is_etf",
    "is_test",
    "financial_status",
    "round_lot",
    "source",
]


@runtime_checkable
class PriceSource(Protocol):
    name: str

    def daily_bars(self, symbols: list[str], start: date, end: date) -> pd.DataFrame:
        """Long-format daily bars with BAR_COLUMNS. As-traded prices."""
        ...

    def minute_bars(self, symbol: str, day: date) -> pd.DataFrame:
        """Intraday bars for one symbol/day with MINUTE_COLUMNS (dt tz-aware)."""
        ...

    def symbol_meta(self, asof: date | None = None) -> pd.DataFrame:
        """Listing metadata with META_COLUMNS.

        ``asof`` is honoured by vendors that carry historical listing data;
        yfinance returns the current snapshot and logs a warning.
        """
        ...

    def corporate_actions(self, symbols: list[str]) -> pd.DataFrame:
        """symbol, dt, kind ('split'|'dividend'), ratio."""
        ...
