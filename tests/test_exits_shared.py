"""The shared-code invariant: the backtest and the live path must agree.

docs/exit_loop.md names this file as the test without which "the backtest
numbers describe a different strategy than the one running". It covers three
ways that equality can break, each of which has already broken once:

1. The engine and the live loop drive `exits.py` in a different order.
2. The engine builds a `Bar` from a column the feature pipeline does not emit,
   so the live path silently gets a default (this is what happened to sma20:
   `evaluate_exits` guards thesis-break with `bar.sma20 > 0`, so the exit
   disappeared entirely outside the backtest script).
3. The stop level is computed in two places with two different formulas.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from uat.backtest.costs import CostParams
from uat.backtest.engine import Backtester, EntryParams, PortfolioParams
from uat.features.pipeline import compute_symbol_features
from uat.strategy.exits import ExitParams, apply_fill, evaluate_exits, update_trackers
from uat.strategy.sizing import size_position, stop_level
from uat.strategy.types import Bar, Position, PositionState

EXITS = ExitParams()
# Zero out the cost model. This test is about WHICH exit fires on WHICH bar,
# and a fill-price penalty would mask a sequencing difference behind a price
# difference. Costs have their own tests in test_strategy.py.
FREE = CostParams(
    commission_per_side=0.0, sec_fee_rate=0.0, finra_taf_per_share=0.0,
    impact_k=0.0, cs_spread_multiplier=0.0, stop_limit_penalty=0.0,
)


def panel(ohlc: list[tuple[float, float, float, float]], *, symbol="TEST",
          volume=5_000_000, atr=0.10, sma20=0.0, rvol=1.0) -> pd.DataFrame:
    """A feature-complete panel, in the shape the engine indexes."""
    start = date(2026, 1, 5)
    rows = []
    for i, (o, h, l, c) in enumerate(ohlc):
        rows.append({
            "symbol": symbol,
            "dt": start + timedelta(days=i),
            "open": o, "high": h, "low": l, "close": c,
            "volume": volume,
            "adj_factor": 1.0,
            "atr14": atr,
            "sma20": sma20,
            "rvol_5": rvol,
            "medvol_20": volume,
            "addv_20": volume * c,
            "cs_spread_20": 0.0,
            "halted": False,
            "limit_locked": False,
        })
    return pd.DataFrame(rows)


def run_engine(bars: pd.DataFrame, *, equity=100_000.0) -> pd.DataFrame:
    """Backtest the panel with a signal on the first session."""
    signals = pd.DataFrame(
        [{"dt": bars["dt"].iloc[0], "symbol": bars["symbol"].iloc[0], "score": 1.0}]
    )
    bt = Backtester(
        bars=bars,
        signals=signals,
        cost=FREE,
        entry=EntryParams(),
        exits=EXITS,
        portfolio=PortfolioParams(initial_equity=equity, enforce_drawdown_halt=False),
    )
    return bt.run().trades


def replay_live(bars: pd.DataFrame, pos: Position) -> list[tuple[date, str, int]]:
    """Drive exits the way a live loop does: one completed bar at a time.

    Deliberately does NOT reuse the engine's helpers. The point is to express
    the live contract independently -- trackers first, then evaluate, then
    apply -- starting from the session AFTER entry, and see whether the two
    arrive at the same answer.
    """
    fired: list[tuple[date, str, int]] = []
    rows = bars[pd.to_datetime(bars["dt"]).dt.date > pos.opened_on]
    prev_rows = bars[pd.to_datetime(bars["dt"]).dt.date <= pos.opened_on]
    prev_low = float(prev_rows["low"].iloc[-1]) if len(prev_rows) else None

    for r in rows.itertuples(index=False):
        if pos.state is PositionState.CLOSED:
            break
        bar = Bar(
            dt=r.dt, open=r.open, high=r.high, low=r.low, close=r.close,
            volume=r.volume, atr14=r.atr14, sma20=r.sma20, rvol_5=r.rvol_5,
            cs_spread=r.cs_spread_20,
        )
        update_trackers(pos, bar)
        orders = evaluate_exits(
            pos, bar,
            prev_low=prev_low if prev_low is not None else bar.low,
            median_volume=float(r.medvol_20),
            params=EXITS,
        )
        for o in orders:
            fired.append((bar.dt, o.reason.value, min(o.qty, pos.qty_open)))
            # The live loop fills at the level it submitted; costs are off here
            # so the engine does the same.
            apply_fill(pos, o, o.limit_px, EXITS)
        prev_low = bar.low
    return fired


def position_from_trade(trade: pd.Series, bars: pd.DataFrame) -> Position:
    """Rebuild the engine's position at the moment of entry.

    Seeded from the engine's realised entry rather than an assumed one, so the
    comparison isolates the EXIT path. Entry has its own skips and caps and is
    not what this test is about.
    """
    entry_row = bars[pd.to_datetime(bars["dt"]).dt.date == trade["entry_dt"]].iloc[0]
    atr = float(entry_row["atr14"])
    px = float(trade["entry_px"])
    return Position(
        symbol=trade["symbol"],
        opened_on=trade["entry_dt"],
        entry_px=px,
        qty_initial=int(trade["qty"]),
        qty_open=int(trade["qty"]),
        atr_at_entry=atr,
        stop_px=stop_level(px, atr, 2.5, 0.20),
        hh_since_entry=float(entry_row["high"]),
        hc_since_entry=float(entry_row["close"]),
    )


# ------------------------------------------------- 1. same exit, same session


# The engine's `trades` frame only records CLOSED positions and keeps the last
# exit reason, so every scenario below has to run the position to zero.
# Bar 0 exists purely to give the entry bar a predecessor -- `_try_enter`
# rejects a session with no previous bar.
SCENARIOS = {
    "stop": (
        [(1.00, 1.05, 0.98, 1.00), (1.00, 1.02, 0.95, 0.96),
         (0.95, 0.96, 0.70, 0.72), (0.72, 0.75, 0.70, 0.71)],
        {},
        ["stop"],
    ),
    "tp1_then_trail": (
        # Bar 2 takes TP1 at +30% WITHOUT dipping under the trail -- a bar that
        # does both exits on the trail, because adverse checks run first and
        # that priority is itself part of the shared contract. TP1 then freezes
        # ATR and lifts the stop to breakeven, and bar 3 gives the trail back.
        [(1.00, 1.05, 0.98, 1.00), (1.00, 1.02, 0.95, 1.00),
         (1.35, 1.50, 1.30, 1.45), (1.30, 1.32, 1.10, 1.15)],
        {},
        ["tp1", "trail"],
    ),
    "thesis_break": (
        # Below entry, below the 20MA, and volume has gone.
        [(1.00, 1.05, 0.98, 1.00), (1.00, 1.02, 0.95, 1.00),
         (0.98, 0.99, 0.90, 0.91)],
        {"sma20": 1.30, "rvol": 0.5},
        ["thesis"],
    ),
}


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_engine_and_live_replay_fire_the_same_exits(name):
    ohlc, kwargs, expected = SCENARIOS[name]
    bars = panel(ohlc, **kwargs)
    trades = run_engine(bars)
    assert len(trades) == 1, f"{name}: engine closed no trade"
    trade = trades.iloc[0]

    fired = replay_live(bars, position_from_trade(trade, bars))
    assert [reason for _, reason, _ in fired] == expected, f"{name}: {fired}"

    # The engine keeps only the final exit, so that is what can be compared --
    # same reason AND same session. A loop that evaluated before advancing the
    # trackers, or that began on the entry bar, would drift by one session.
    assert trade["exit_reason"] == expected[-1]
    assert fired[-1][0] == trade["exit_dt"]


def test_entry_session_is_excluded_from_exit_evaluation():
    """A collapse on the entry bar itself must not exit that same session.

    The engine skips `pos.opened_on >= dt`; a live loop that starts evaluating
    the moment the fill lands would stop itself out on its own entry bar, at a
    price the backtest never pays.
    """
    bars = panel([(1.00, 1.05, 0.98, 1.00), (1.00, 1.02, 0.50, 0.55),
                  (0.55, 0.60, 0.40, 0.45)])
    trades = run_engine(bars)
    assert len(trades) == 1
    pos = position_from_trade(trades.iloc[0], bars)
    fired = replay_live(bars, pos)
    assert fired, "replay fired nothing"
    assert all(dt > pos.opened_on for dt, _, _ in fired)
    # The entry bar's low of 0.50 is far below the stop; the exit belongs to
    # the following session, not to the fill's own bar.
    assert fired[0][0] > pos.opened_on


# ------------------------------------------------- 2. Bar inputs exist


def test_every_bar_field_the_engine_reads_is_emitted_by_the_pipeline():
    """Guards the sma20 class of bug.

    `Backtester._to_bar` pulls these off the panel row with a silent default.
    If the feature pipeline stops emitting one, the backtest keeps working off
    whatever the backtest script happens to add while every other consumer gets
    the default -- and for sma20 that default switched thesis-break off.
    """
    n = 260
    raw = pd.DataFrame({
        "symbol": ["TEST"] * n,
        "dt": pd.bdate_range("2025-01-02", periods=n).date,
        "open": np.linspace(1.0, 3.0, n),
        "high": np.linspace(1.05, 3.05, n),
        "low": np.linspace(0.95, 2.95, n),
        "close": np.linspace(1.0, 3.0, n),
        "volume": np.full(n, 2_000_000),
        "adj_factor": np.ones(n),
    })
    feats = compute_symbol_features(raw)

    # Fields _to_bar reads from the feature side, not from the raw bars.
    for col in ("atr14", "sma20", "rvol_5"):
        assert col in feats.columns, f"feature pipeline stopped emitting {col}"
        assert pd.notna(feats[col].iloc[-1]), f"{col} is NaN on the newest row"
    # Sizing and the exhaustion rule read these off the same row.
    for col in ("medvol_20", "addv_20"):
        assert col in feats.columns, f"feature pipeline stopped emitting {col}"


# ------------------------------------------------- 3. one stop formula


def test_engine_entry_stop_matches_the_sizer():
    """The engine must not compute its own stop.

    `_try_enter` used to build `stop_px` inline as
    `px - max(atr_mult * atr, stop_min_pct * px)`, which skips the 50%-of-price
    clamp that commit 80f60b4 added to sizing.py. On a name whose ATR is 40% of
    price that inline formula lands at or below zero and gets floored to $0.01
    -- a position with no downside control, sized as though it had a 50% stop.
    """
    atr = 0.40                      # 40% of a $1.00 price
    equity = 100_000.0
    bars = panel([(1.00, 1.05, 0.95, 1.00)] * 3, atr=atr)

    # _try_enter is called directly: the engine only reports CLOSED trades, so
    # the stop a live position carries is otherwise invisible to a test.
    bt = Backtester(
        bars=bars,
        signals=pd.DataFrame([{"dt": bars["dt"].iloc[0], "symbol": "TEST", "score": 1.0}]),
        cost=FREE,
        entry=EntryParams(),
        exits=EXITS,
        portfolio=PortfolioParams(initial_equity=equity, enforce_drawdown_halt=False),
    )
    positions: dict[str, Position] = {}
    result, why = bt._try_enter(
        "TEST", bars["dt"].iloc[1], positions, {}, equity, equity
    )
    assert result is not None, f"entry rejected: {why}"
    pos = positions["TEST"]

    sized = size_position(
        equity=equity, entry_px=pos.entry_px, atr14=atr,
        median_volume_20=5_000_000, addv_20=5_000_000,
    )
    assert pos.stop_px == pytest.approx(sized.stop_px)
    # The clamp has to bite here: 2.5 x 40% is a 100%-of-price distance, which
    # becomes 50%. Without it the stop floors to $0.01.
    assert pos.stop_px == pytest.approx(pos.entry_px * 0.5, rel=1e-6)
    assert pos.stop_px > 0.01


def test_unclamped_stop_would_have_been_worthless():
    """Documents what the inline formula produced, so the clamp cannot regress."""
    px, atr = 1.00, 0.40
    inline = max(0.01, px - max(2.5 * atr, 0.20 * px))
    assert inline == pytest.approx(0.01)
    assert stop_level(px, atr) == pytest.approx(0.50)
