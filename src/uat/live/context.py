"""Market context for a held symbol, built from the same features the backtest uses.

`exits.evaluate_exits` needs more than a price: a `Bar` carrying atr14, sma20
and rvol_5, plus the previous session's low and a median volume for the
exhaustion rule. The backtest gets those from its feature panel; anything live
has to produce the identical numbers from the identical code or the shared-exit
invariant is a fiction.

So this module runs `compute_symbol_features` -- not a hand-rolled ATR -- over
the symbol's own history and reports how stale the result is. Staleness is
returned rather than hidden: the daily ingest is a separate job, and an exit
evaluated against a three-session-old bar is a different decision than one
evaluated against yesterday's.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from uat.clock import current_session_date, sessions
from uat.data.ingest_daily import load_bars
from uat.features.pipeline import compute_symbol_features
from uat.logging import get
from uat.strategy.types import Bar

log = get("live.context")

# compute_symbol_features needs 200+ bars for the long moving averages; 420
# calendar days is comfortably more than that without loading whole history.
LOOKBACK_DAYS = 420


@dataclass
class SymbolContext:
    symbol: str
    bar: Bar                      # last COMPLETED daily bar
    prev_low: float
    median_volume: float
    stale_sessions: int           # trading sessions between bar.dt and today
    history: pd.DataFrame         # raw bars, ascending, for high-water reconstruction
    features: pd.DataFrame        # feature rows aligned to history

    @property
    def is_current(self) -> bool:
        """True when the newest bar is the last completed session."""
        return self.stale_sessions <= 0


def _feature_row(features: pd.DataFrame, on_or_before: date) -> pd.Series | None:
    rows = features[pd.to_datetime(features["dt"]).dt.date <= on_or_before]
    return None if rows.empty else rows.iloc[-1]


def build(symbol: str, asof: date | None = None,
          lookback_days: int = LOOKBACK_DAYS) -> SymbolContext | None:
    """Context for `symbol`, or None when there is not enough history."""
    end = asof or date.today()
    start = end - pd.Timedelta(days=lookback_days).to_pytimedelta()
    bars = load_bars(start=start, end=end, symbols=[symbol])
    if bars.empty:
        log.warning("no bars for %s", symbol)
        return None
    bars = bars.sort_values("dt").reset_index(drop=True)
    feats = compute_symbol_features(bars)

    last, flast = bars.iloc[-1], feats.iloc[-1]
    bar = Bar(
        dt=pd.Timestamp(last["dt"]).date(),
        open=float(last["open"]),
        high=float(last["high"]),
        low=float(last["low"]),
        close=float(last["close"]),
        volume=float(last["volume"]),
        atr14=float(flast.get("atr14") or 0.0),
        sma20=float(flast.get("sma20") or 0.0),
        rvol_5=float(flast.get("rvol_5") or 1.0),
        cs_spread=flast.get("cs_spread_20"),
    )
    prev_low = float(bars.iloc[-2]["low"]) if len(bars) > 1 else bar.low
    medvol = float(flast.get("medvol_20") or 0.0)

    # Sessions strictly after the newest bar, up to the last completed one.
    today_session = current_session_date()
    gap = [d for d in sessions(bar.dt, today_session) if d > bar.dt]
    return SymbolContext(symbol, bar, prev_low, medvol, len(gap), bars, feats)


def atr_on(ctx: SymbolContext, on_or_before: date) -> float:
    """ATR14 as of a past session -- the ATR a position was actually sized with.

    Adopting a position entered three sessions ago with today's ATR would set a
    stop the entry never implied, so the stop is anchored to the entry date.
    """
    row = _feature_row(ctx.features, on_or_before)
    return float(row.get("atr14") or 0.0) if row is not None else ctx.bar.atr14


def high_water(ctx: SymbolContext, since: date) -> tuple[float, float, int]:
    """(highest high, highest close, sessions elapsed) strictly after `since`.

    The backtest advances these once per bar from the session after entry, so
    the same window is used here. With no bars yet after entry the caller falls
    back to the entry price, which is what the engine starts from.
    """
    after = ctx.history[pd.to_datetime(ctx.history["dt"]).dt.date > since]
    if after.empty:
        return 0.0, 0.0, 0
    return float(after["high"].max()), float(after["close"].max()), len(after)
