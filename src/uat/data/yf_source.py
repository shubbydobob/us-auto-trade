"""yfinance implementation of PriceSource.

THE ADJUSTMENT TRAP
-------------------
Yahoo back-adjusts OHLC for splits even when ``auto_adjust=False``. A stock that
reverse-split 1:10 in 2024 shows its 2023 prices multiplied by 10 -- a name that
actually traded at $0.30 appears as $3.00. Left uncorrected this fabricates the
universe (sub-$1 names look investable), corrupts every volume feature, and can
manufacture a beautiful equity curve out of nothing.

yfinance reports a 1:10 reverse split as ratio 0.1 and a 2:1 forward split as
2.0. Yahoo's displayed series relates to the as-traded series through the product
of split ratios occurring strictly AFTER the bar date::

    split_factor(t)  = prod(ratio_i for split dates s_i > t)
    displayed_price  = as_traded_price  / split_factor
    displayed_volume = as_traded_volume * split_factor

so ingest inverts it and stores as-traded prices plus ``split_factor`` as
``adj_factor``. Dollar volume is invariant under the transform, which is what
tests/test_adjustment.py asserts.

``auto_adjust=True`` is never used in this codebase; tests/test_no_lookahead.py
greps the source tree for it.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import yfinance as yf

from uat.data.base import BAR_COLUMNS, MINUTE_COLUMNS
from uat.logging import get

log = get("data.yf")

BATCH_SIZE = 200


def _naive_days(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = index.tz_localize(None) if index.tz is not None else index
    return idx.normalize()


def split_factor(index: pd.DatetimeIndex, splits: pd.Series) -> pd.Series:
    """Cumulative product of split ratios occurring strictly after each bar date."""
    factor = pd.Series(1.0, index=index)
    if splits is None or len(splits) == 0:
        return factor
    bars = _naive_days(index)
    for split_dt, ratio in splits.items():
        ratio = float(ratio)
        if ratio <= 0:
            continue
        split_day = pd.Timestamp(split_dt)
        if split_day.tzinfo is not None:
            split_day = split_day.tz_localize(None)
        factor[bars < split_day.normalize()] *= ratio
    return factor


def reconstruct_as_traded(df: pd.DataFrame, splits: pd.Series) -> pd.DataFrame:
    """Undo Yahoo's split back-adjustment, returning as-traded OHLCV + adj_factor."""
    out = df.copy()
    f = split_factor(out.index, splits)
    for col in ("Open", "High", "Low", "Close"):
        if col in out.columns:
            out[col] = out[col] * f
    out["Volume"] = (out["Volume"] / f).round()
    out["adj_factor"] = f
    return out


def _splits_from_frame(d: pd.DataFrame) -> pd.Series:
    if "Stock Splits" not in d.columns:
        return pd.Series(dtype=float)
    s = pd.to_numeric(d["Stock Splits"], errors="coerce").fillna(0.0)
    return s[s > 0]


class YFinanceSource:
    name = "yfinance"

    def daily_bars(self, symbols: list[str], start: date, end: date) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        missing: list[str] = []

        for i in range(0, len(symbols), BATCH_SIZE):
            chunk = symbols[i : i + BATCH_SIZE]
            raw = yf.download(
                chunk,
                start=start,
                end=end,
                interval="1d",
                auto_adjust=False,  # see module docstring; never True
                actions=True,
                progress=False,
                threads=True,
                group_by="ticker",
            )
            if raw is None or raw.empty:
                missing.extend(chunk)
                continue

            available = set(raw.columns.get_level_values(0))
            for sym in chunk:
                if sym not in available:
                    missing.append(sym)
                    continue
                d = raw[sym].dropna(subset=["Close"])
                if d.empty:
                    missing.append(sym)
                    continue
                adj = reconstruct_as_traded(d, _splits_from_frame(d))
                frames.append(
                    pd.DataFrame(
                        {
                            "symbol": sym,
                            "dt": _naive_days(adj.index),
                            "open": adj["Open"].astype(float).values,
                            "high": adj["High"].astype(float).values,
                            "low": adj["Low"].astype(float).values,
                            "close": adj["Close"].astype(float).values,
                            "volume": adj["Volume"].fillna(0).astype("int64").values,
                            "adj_factor": adj["adj_factor"].astype(float).values,
                        }
                    )
                )
            log.info(
                "daily_bars batch done",
                extra={"extra_fields": {"offset": i, "size": len(chunk), "ok": len(frames)}},
            )

        if missing:
            log.warning(
                "symbols returned no data (delisted or bad ticker)",
                extra={"extra_fields": {"count": len(missing), "sample": missing[:10]}},
            )
        if not frames:
            return pd.DataFrame(columns=BAR_COLUMNS)

        out = pd.concat(frames, ignore_index=True)
        out["dt"] = pd.to_datetime(out["dt"]).dt.date
        return out[BAR_COLUMNS]

    def minute_bars(self, symbol: str, day: date) -> pd.DataFrame:
        """yfinance serves roughly the last 7 days of 1-minute history.

        That is far too short to backtest the intraday entry layer -- it can only
        spot-check it. See docs/data_biases.md.
        """
        raw = yf.download(
            symbol,
            start=day,
            end=pd.Timestamp(day) + pd.Timedelta(days=1),
            interval="1m",
            auto_adjust=False,
            progress=False,
        )
        if raw is None or raw.empty:
            return pd.DataFrame(columns=MINUTE_COLUMNS)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        return pd.DataFrame(
            {
                "dt": raw.index,
                "open": raw["Open"].astype(float).values,
                "high": raw["High"].astype(float).values,
                "low": raw["Low"].astype(float).values,
                "close": raw["Close"].astype(float).values,
                "volume": raw["Volume"].fillna(0).astype("int64").values,
            }
        )

    def symbol_meta(self, asof: date | None = None) -> pd.DataFrame:
        from uat.data.nasdaq_trader import fetch_symbol_directory

        if asof is not None and asof != date.today():
            log.warning(
                "yfinance has no historical listing data; returning today's snapshot",
                extra={"extra_fields": {"requested_asof": str(asof)}},
            )
        return fetch_symbol_directory()

    def corporate_actions(self, symbols: list[str]) -> pd.DataFrame:
        rows = []
        for sym in symbols:
            try:
                sp = yf.Ticker(sym).splits
            except Exception:
                continue
            if sp is None or len(sp) == 0:
                continue
            for dt, ratio in sp.items():
                ts = pd.Timestamp(dt)
                if ts.tzinfo is not None:
                    ts = ts.tz_localize(None)
                rows.append(
                    {"symbol": sym, "dt": ts.date(), "kind": "split", "ratio": float(ratio)}
                )
        return pd.DataFrame(rows, columns=["symbol", "dt", "kind", "ratio"])
