"""LULD (limit up / limit down) bands and halt detection from daily bars.

Penny stocks halt constantly, and a backtest that fills orders on halted or
limit-locked bars is reporting trades that could never have happened.

Tier 2 reference bands (NMS Plan, percentage of the reference price):
    price >= $3.00            -> 10%
    $0.75 <= price < $3.00    -> 20%
    price <  $0.75            -> lesser of 75% or $0.15

Bands are DOUBLED during 09:30-09:45 and 15:35-16:00. Daily bars cannot tell us
when in the session the move happened, so the doubled band is used as the
detection threshold -- the conservative choice, since assuming the narrow band
would mark ordinary volatile days as halted and understate the opportunity set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def luld_band(reference_price: float, doubled: bool = True) -> float:
    """LULD band as a fraction of the reference price."""
    if reference_price <= 0:
        return np.inf
    if reference_price >= 3.00:
        band = 0.10
    elif reference_price >= 0.75:
        band = 0.20
    else:
        band = min(0.75, 0.15 / reference_price)
    return band * (2.0 if doubled else 1.0)


def luld_band_series(reference_price: pd.Series, doubled: bool = True) -> pd.Series:
    mult = 2.0 if doubled else 1.0
    band = pd.Series(np.nan, index=reference_price.index, dtype=float)
    band[reference_price >= 3.00] = 0.10
    band[(reference_price >= 0.75) & (reference_price < 3.00)] = 0.20
    low = reference_price < 0.75
    band[low] = np.minimum(0.75, 0.15 / reference_price[low].replace(0, np.nan))
    return band * mult


def flag_halts(bars: pd.DataFrame) -> pd.DataFrame:
    """Add `halted` and `limit_locked` columns to long-format bars.

    halted       -- intraday range exceeded the (doubled) LULD band, so trading
                    was very likely interrupted. No new entries on these bars.
    limit_locked -- open == high == close and the move exceeded the band: the
                    classic limit-up lock where no fill is obtainable at all.
    """
    df = bars.sort_values(["symbol", "dt"]).copy()
    prev_close = df.groupby("symbol", sort=False)["close"].shift(1)
    band = luld_band_series(prev_close)

    with np.errstate(divide="ignore", invalid="ignore"):
        intraday_move = (df["high"] - df["low"]) / prev_close
        gap_move = (df["close"] / prev_close - 1).abs()

    df["luld_band"] = band
    df["halted"] = (intraday_move > band).fillna(False)

    locked = (
        np.isclose(df["open"], df["high"])
        & np.isclose(df["open"], df["close"])
        & (gap_move >= band)
    )
    df["limit_locked"] = locked.fillna(False)
    return df


def can_enter(bar: pd.Series) -> bool:
    """Entries are forbidden on halted or limit-locked bars."""
    return not (bool(bar.get("halted", False)) or bool(bar.get("limit_locked", False)))


def exit_price_on_halt(bar: pd.Series, desired_px: float) -> float:
    """Exit fill on a halted bar: the worse of the desired level and the band edge."""
    band = bar.get("luld_band", np.nan)
    if not np.isfinite(band):
        return desired_px
    floor = bar["open"] * (1 - band)
    return max(min(desired_px, bar["high"]), floor)
