"""Universe filters: listing-level (static) and bar-level (point-in-time).

Every rejected symbol keeps its reason so dim.universe_snapshot can explain why
the universe shrank on any given day -- silent universe drift is one of the
easiest ways to misread a backtest.

The dollar-volume floor is DERIVED, not a magic constant: you must be able to
build the intended position without exceeding the participation cap, i.e.

    min_addv >= target_position_notional / (participation_cap * entry_days)

so the universe automatically tightens as the account grows.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from uat.config import load_settings

# Non-common-stock share classes. Units/warrants/rights behave nothing like the
# underlying and their "volume" is mostly arbitrage noise.
# Plurals matter: the directory writes "- Units" and "- Warrants", which a
# \bunit\b / \bwarrant\b pattern silently misses and lets SPAC units through.
NON_COMMON_PATTERNS = re.compile(
    r"\b(?:warrants?|units?|rights?|preferred|depositary\s+shares?|depositary\s+receipts?|"
    r"subordinated|debentures?|notes?\s+due|trust\s+preferred|convertible\s+preferred)\b",
    re.IGNORECASE,
)

# "Ordinary Shares" marks a foreign-domiciled issuer (typically a 20-F filer or a
# Cayman SPAC); their volume is driven by non-US sessions and corrupts RVOL.
ADR_PATTERNS = re.compile(r"american\s+depositar|ordinary\s+shares?", re.IGNORECASE)

# NASDAQ appends a bare 5th character for non-common issues (W warrant, U unit,
# R rights); otherlisted uses dotted/dashed suffixes like ".WS" / ".U" / ".R".
SUFFIX_PATTERN = re.compile(r"[.\-](?:WS|WT|U|RT|R|P[A-Z]?)$", re.IGNORECASE)
NASDAQ_5TH_CHAR_PATTERN = re.compile(r"^[A-Z]{4}[WUR]$")


def derived_min_addv(
    target_position_notional: float,
    participation_cap: float,
    entry_days: int,
    floor: float,
) -> float:
    """Smallest average dollar volume that can absorb the intended position."""
    if participation_cap <= 0 or entry_days <= 0:
        raise ValueError("participation_cap and entry_days must be positive")
    required = target_position_notional / (participation_cap * entry_days)
    return max(float(floor), float(required))


def filter_listings(meta: pd.DataFrame, settings=None) -> pd.DataFrame:
    """Apply listing-level rules. Returns meta + `reject_reason` (None if kept)."""
    s = settings or load_settings()
    allowed = set(s.get("universe.exchanges"))
    bad_status = set(s.get("universe.exclude_financial_status") or [])

    out = meta.copy()
    reason = pd.Series(pd.NA, index=out.index, dtype="object")

    def mark(mask: pd.Series, label: str) -> None:
        nonlocal reason
        reason = reason.where(~(mask & reason.isna()), label)

    mark(~out["exchange"].isin(allowed), "exchange_not_kis_tradeable")
    mark(out["is_etf"].fillna(False), "etf")
    mark(out["is_test"].fillna(False), "test_issue")
    mark(out["financial_status"].astype(str).str.strip().isin(bad_status), "bankrupt")
    mark(out["security_name"].astype(str).str.contains(NON_COMMON_PATTERNS), "not_common_stock")
    mark(out["security_name"].astype(str).str.contains(ADR_PATTERNS), "adr")
    mark(out["symbol"].astype(str).str.contains(SUFFIX_PATTERN), "non_common_suffix")
    is_nasdaq = out["exchange"].eq("NASDAQ")
    mark(
        is_nasdaq & out["symbol"].astype(str).str.match(NASDAQ_5TH_CHAR_PATTERN),
        "nasdaq_5th_char_non_common",
    )

    out["reject_reason"] = reason
    return out


def _rolling_median(df: pd.DataFrame, col: str, window: int) -> pd.Series:
    return df.groupby("symbol", sort=False)[col].transform(
        lambda s: s.rolling(window, min_periods=window).median()
    )


def corwin_schultz_spread(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Corwin-Schultz (2012) high/low spread estimator, rolling median.

    Uses as-traded high/low. Known to be noisy for illiquid issues, hence the
    configurable `costs.cs_spread_multiplier` calibration knob.
    """
    g = df.groupby("symbol", sort=False)
    hi, lo = df["high"], df["low"]
    hi_next, lo_next = g["high"].shift(-1), g["low"].shift(-1)

    with np.errstate(divide="ignore", invalid="ignore"):
        beta = np.log(hi / lo) ** 2 + np.log(hi_next / lo_next) ** 2
        hi2 = np.maximum(hi, hi_next)
        lo2 = np.minimum(lo, lo_next)
        gamma = np.log(hi2 / lo2) ** 2

        k = 3 - 2 * np.sqrt(2)
        alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
        spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))

    spread = pd.Series(spread, index=df.index).clip(lower=0).replace([np.inf, -np.inf], np.nan)
    tmp = df[["symbol"]].copy()
    tmp["cs"] = spread
    # Shift by 1 so day T only sees pairs that completed on or before T.
    return tmp.groupby("symbol", sort=False)["cs"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=max(5, window // 2)).median()
    )


def bar_level_metrics(bars: pd.DataFrame) -> pd.DataFrame:
    """Liquidity / listing-age metrics used by the universe gate.

    `bars` must be long-format as-traded bars sorted by (symbol, dt).
    Everything here is trailing-only: day T sees only bars up to and including T.
    """
    df = bars.sort_values(["symbol", "dt"]).reset_index(drop=True)
    df["dollar_volume"] = df["close"] * df["volume"]
    df["addv_20"] = _rolling_median(df, "dollar_volume", 20)
    df["addv_60"] = _rolling_median(df, "dollar_volume", 60)
    df["medvol_20"] = _rolling_median(df, "volume", 20)
    df["sessions_listed"] = df.groupby("symbol", sort=False).cumcount() + 1
    df["min_volume_20"] = df.groupby("symbol", sort=False)["volume"].transform(
        lambda s: s.rolling(20, min_periods=20).min()
    )
    df["cs_spread_20"] = corwin_schultz_spread(df)
    df["split_recent"] = (
        df.groupby("symbol", sort=False)["adj_factor"]
        .transform(lambda s: s.diff().ne(0).rolling(30, min_periods=1).max())
        .fillna(0)
        .astype(bool)
    )
    return df


def classify_tier(
    df: pd.DataFrame, min_addv: float | None = None, settings=None
) -> pd.DataFrame:
    """Assign core / watch / rejected per (symbol, dt) with a reject reason.

    Price checks use as-traded close. Using back-adjusted close here is the
    single most dangerous bug in the system (see data/yf_source.py).
    """
    s = settings or load_settings()
    u = lambda k: s.get(f"universe.{k}")  # noqa: E731
    min_addv = float(min_addv if min_addv is not None else u("min_addv_floor"))

    out = df.copy()
    reason = pd.Series(pd.NA, index=out.index, dtype="object")

    def mark(mask, label: str) -> None:
        nonlocal reason
        mask = pd.Series(mask, index=out.index).fillna(True)
        reason = reason.where(~(mask & reason.isna()), label)

    mark(out["sessions_listed"] < u("min_listed_sessions"), "too_recently_listed")
    mark(out["split_recent"], "split_within_30_sessions")
    mark(out["min_volume_20"].le(0), "zero_volume_day")
    mark(out["medvol_20"] < u("min_median_volume_20"), "median_volume_too_low")
    mark(out["addv_20"] < min_addv, "addv_20_too_low")
    mark(out["addv_60"] < u("min_addv_60"), "addv_60_too_low")
    mark(out["cs_spread_20"] > u("max_cs_spread"), "spread_too_wide")
    mark(
        (out["close"] < u("watch_price_min")) | (out["close"] > u("watch_price_max")),
        "outside_price_band",
    )

    in_core = out["close"].between(u("core_price_min"), u("core_price_max"))
    out["reject_reason"] = reason
    out["tier"] = np.where(
        reason.notna(), "rejected", np.where(in_core, "core", "watch")
    )
    return out
