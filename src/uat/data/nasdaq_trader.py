"""NASDAQ Trader symbol directory -> normalised listing metadata.

Two pipe-delimited files, both served over HTTPS:
  nasdaqlisted.txt  Symbol|Security Name|Market Category|Test Issue|
                    Financial Status|Round Lot Size|ETF|NextShares
  otherlisted.txt   ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|
                    Round Lot Size|Test Issue|NASDAQ Symbol

otherlisted's Exchange column is a single letter:
  A = NYSE American (AMEX)   N = NYSE    P = NYSE Arca
  Z = Cboe BZX               V = IEX     (plus newer venues, e.g. TXSE)

Only NASDAQ / NYSE / AMEX are tradeable through KIS, so Arca/BZX/IEX primaries
are mapped but filtered out downstream in universe/filters.py.

Financial Status (NASDAQ only):
  N normal, D deficient (listing-standard shortfall -> reverse-split risk),
  E delinquent, Q bankrupt, G/H/J/K various deficiency combinations.
"""
from __future__ import annotations

import io
from datetime import date

import pandas as pd
import requests

from uat.data.base import META_COLUMNS
from uat.logging import get

log = get("data.nasdaq_trader")

NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

EXCHANGE_MAP = {
    "A": "AMEX",
    "N": "NYSE",
    "P": "ARCA",
    "Z": "BZX",
    "V": "IEX",
}

_TIMEOUT = 30


def _fetch_pipe(url: str) -> pd.DataFrame:
    resp = requests.get(url, timeout=_TIMEOUT)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), sep="|")
    # Both files end with a "File Creation Time:" trailer row.
    return df[~df.iloc[:, 0].astype(str).str.startswith("File Creation Time")].copy()


def _yn(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper().eq("Y")


def fetch_symbol_directory(asof: date | None = None) -> pd.DataFrame:
    """Download and normalise both files into META_COLUMNS."""
    asof = asof or date.today()

    nas = _fetch_pipe(NASDAQ_LISTED_URL)
    nas = nas[nas["Symbol"].notna()]
    nasdaq = pd.DataFrame(
        {
            "symbol": nas["Symbol"].astype(str).str.strip(),
            "exchange": "NASDAQ",
            "security_name": nas["Security Name"].astype(str).str.strip(),
            "is_etf": _yn(nas["ETF"]),
            "is_test": _yn(nas["Test Issue"]),
            "financial_status": nas["Financial Status"].astype(str).str.strip(),
            "round_lot": pd.to_numeric(nas["Round Lot Size"], errors="coerce"),
            "source": "nasdaqlisted",
        }
    )

    oth = _fetch_pipe(OTHER_LISTED_URL)
    oth = oth[oth["ACT Symbol"].notna()]
    other = pd.DataFrame(
        {
            "symbol": oth["ACT Symbol"].astype(str).str.strip(),
            "exchange": oth["Exchange"].astype(str).str.strip().map(EXCHANGE_MAP),
            "security_name": oth["Security Name"].astype(str).str.strip(),
            "is_etf": _yn(oth["ETF"]),
            "is_test": _yn(oth["Test Issue"]),
            # otherlisted carries no financial-status column.
            "financial_status": "N",
            "round_lot": pd.to_numeric(oth["Round Lot Size"], errors="coerce"),
            "source": "otherlisted",
        }
    )

    out = pd.concat([nasdaq, other], ignore_index=True)
    out["exchange"] = out["exchange"].fillna("OTHER")
    out["round_lot"] = out["round_lot"].fillna(100).astype(int)
    out = out.drop_duplicates(subset=["symbol"], keep="first").reset_index(drop=True)

    log.info(
        "symbol directory fetched",
        extra={
            "extra_fields": {
                "asof": str(asof),
                "total": len(out),
                "by_exchange": out["exchange"].value_counts().to_dict(),
            }
        },
    )
    return out[META_COLUMNS]


def store_snapshot(df: pd.DataFrame, asof: date | None = None) -> int:
    """Persist one point-in-time snapshot into raw.symbol_directory."""
    from uat.storage.db import upsert_df

    asof = asof or date.today()
    rows = df.copy()
    rows.insert(0, "as_of", asof)
    return upsert_df("raw.symbol_directory", rows, ["as_of", "symbol"])
