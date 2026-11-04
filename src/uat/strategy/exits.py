"""Exit logic -- imported verbatim by both backtest/engine.py and live/job_intraday.py.

tests/test_exits_shared.py replays identical bars through both paths and asserts
identical fills. That equality is the only reason to trust a penny-stock
backtest, so nothing here may branch on whether it is running live.

THE TRAILING-STOP DECISION
--------------------------
A pure ATR chandelier is the wrong trail for this strategy. On the reference
chart the stock runs $1.60 -> $14.95; by the top, ATR has exploded, so a 3-ATR
trail sits several dollars below price and hands back most of the subsequent
-55% collapse.

So ATR is FROZEN at TP1 and the trail is the WIDER of:
    highest_close * (1 - 25%)                 -- percent-of-move
    highest_high  - 3 * ATR_frozen_at_TP1     -- volatility-of-the-base

Percent-of-move is the right unit during a parabolic phase; ATR is the right
unit during a normal one. Taking the wider of the two avoids being shaken out
of a runner early while still capping the give-back. After costs, reducing the
give-back on spikes is the single largest P&L lever in the system.
"""
from __future__ import annotations

from dataclasses import dataclass

from uat.strategy.types import Bar, ExitOrder, ExitReason, Position, PositionState


@dataclass(frozen=True)
class ExitParams:
    tp1_gain: float = 0.30
    tp1_fraction: float = 1.0 / 3.0
    tp2_gain: float = 0.80
    tp2_fraction: float = 1.0 / 3.0
    trail_pct_of_high: float = 0.25
    trail_pct_tight: float = 0.18
    trail_tighten_above_gain: float = 2.00
    trail_atr_mult: float = 3.0
    exhaustion_min_gain: float = 1.50
    exhaustion_volume_mult: float = 3.0
    time_stop_sessions: int = 15
    time_stop_min_gain: float = 0.12
    # Thesis-break knobs. The defaults reproduce the hard-coded rule exactly
    # (enabled, no loss buffer, fires from the first session, rvol below 0.8),
    # so exposing them changes no behaviour. They exist because the corrected
    # sma20 made this exit fire on 103 of 199 trades at an average -12.2%,
    # which is where the gate failure now concentrates -- and a rule carrying
    # that much P&L cannot be tuned while it is three literals in a function.
    thesis_enabled: bool = True
    thesis_max_rvol: float = 0.80
    thesis_loss_buffer: float = 0.0     # close must sit below entry*(1-buffer)
    thesis_min_sessions: int = 0
    # Gain at which the trail starts tracking, independent of the TP ladder.
    # Previously the trail engaged only after TP1 fired, so disabling the ladder
    # silently disabled the trail too and "let winners run" could not be tested
    # at all -- it degraded to "hold to the hard stop". None keeps the old
    # behaviour of tying activation to tp1_gain.
    trail_activate_gain: float | None = None

    @classmethod
    def from_settings(cls, settings) -> "ExitParams":
        e = settings.get("exits") or {}
        return cls(**{k: v for k, v in e.items() if k in cls.__dataclass_fields__})


def update_trackers(pos: Position, bar: Bar) -> None:
    """Advance high-water marks and the session counter. Call once per bar."""
    pos.hh_since_entry = max(pos.hh_since_entry, bar.high)
    pos.hc_since_entry = max(pos.hc_since_entry, bar.close)
    pos.sessions_held += 1


def trail_is_active(pos: Position, p: ExitParams) -> bool:
    """Whether the trail is tracking yet.

    Either a TP tranche has fired, or the position's peak gain has passed
    `trail_activate_gain` (defaulting to tp1_gain). The second condition is what
    lets the ladder be switched off without also switching off the trail.
    """
    if pos.state in (PositionState.TP1, PositionState.TP2):
        return True
    threshold = p.trail_activate_gain if p.trail_activate_gain is not None else p.tp1_gain
    return pos.gain >= threshold


def trailing_stop(pos: Position, p: ExitParams) -> float:
    """Current trailing level. Only meaningful once TP1 has triggered."""
    atr = pos.atr_frozen if pos.atr_frozen is not None else pos.atr_at_entry
    pct = (
        p.trail_pct_tight
        if pos.gain > p.trail_tighten_above_gain
        else p.trail_pct_of_high
    )
    pct_trail = pos.hc_since_entry * (1 - pct)
    atr_trail = pos.hh_since_entry - p.trail_atr_mult * atr
    return max(pct_trail, atr_trail)


def is_exhausted(pos: Position, bar: Bar, prev_low: float, median_volume: float,
                 p: ExitParams) -> bool:
    """Parabolic blow-off: big gain, closes under yesterday's low, on huge volume."""
    return (
        pos.gain_at(bar.close) > p.exhaustion_min_gain
        and bar.close < prev_low
        and median_volume > 0
        and bar.volume > p.exhaustion_volume_mult * median_volume
    )


def evaluate_exits(
    pos: Position,
    bar: Bar,
    *,
    prev_low: float,
    median_volume: float,
    params: ExitParams,
) -> list[ExitOrder]:
    """Exit orders for this bar, in priority order.

    Evaluated against the bar's OHLC, so the caller must pass the bar AFTER the
    one that opened the position. Stop checks come first: if a bar both gaps
    through the stop and later prints a take-profit high, the stop happened.
    """
    orders: list[ExitOrder] = []
    if pos.qty_open <= 0 or pos.state is PositionState.CLOSED:
        return orders

    # 1. Hard stop / trailing stop -- adverse first, always.
    stop_level = pos.stop_px
    if trail_is_active(pos, params):
        stop_level = max(stop_level, trailing_stop(pos, params))

    if bar.low <= stop_level:
        reason = ExitReason.TRAIL if stop_level > pos.stop_px else ExitReason.STOP
        return [ExitOrder(reason, pos.qty_open, stop_level, is_stop=True)]

    # 2. Parabolic exhaustion -- exit everything at the next open.
    if is_exhausted(pos, bar, prev_low, median_volume, params):
        return [ExitOrder(ExitReason.EXHAUSTION, pos.qty_open, bar.close, is_stop=False)]

    # 3. Scale-out ladder.
    if pos.state is PositionState.OPEN:
        tp1_px = pos.entry_px * (1 + params.tp1_gain)
        if bar.high >= tp1_px:
            qty = min(pos.qty_open, max(1, round(pos.qty_initial * params.tp1_fraction)))
            orders.append(ExitOrder(ExitReason.TP1, qty, tp1_px))
            return orders

    if pos.state is PositionState.TP1:
        tp2_px = pos.entry_px * (1 + params.tp2_gain)
        if bar.high >= tp2_px:
            qty = min(pos.qty_open, max(1, round(pos.qty_initial * params.tp2_fraction)))
            orders.append(ExitOrder(ExitReason.TP2, qty, tp2_px))
            return orders

    # 4. Thesis break -- below entry, below the 20MA, and volume has gone.
    if (
        params.thesis_enabled
        and pos.sessions_held >= params.thesis_min_sessions
        and bar.close < pos.entry_px * (1.0 - params.thesis_loss_buffer)
        and bar.sma20 > 0
        and bar.close < bar.sma20
        and bar.rvol_5 < params.thesis_max_rvol
    ):
        return [ExitOrder(ExitReason.THESIS, pos.qty_open, bar.close)]

    # 5. Time stop -- recycle capital out of positions that never worked.
    if (
        pos.sessions_held >= params.time_stop_sessions
        and pos.gain_at(bar.close) < params.time_stop_min_gain
    ):
        return [ExitOrder(ExitReason.TIME, pos.qty_open, bar.close)]

    return orders


def apply_fill(pos: Position, order: ExitOrder, fill_px: float, params: ExitParams) -> None:
    """Mutate the position after an exit fills. State transitions live here."""
    qty = min(order.qty, pos.qty_open)
    pos.qty_open -= qty
    pos.realised_pnl += (fill_px - pos.entry_px) * qty

    if order.reason is ExitReason.TP1:
        # Freeze ATR and lift the stop to breakeven -- see module docstring.
        pos.atr_frozen = pos.atr_at_entry
        pos.stop_px = max(pos.stop_px, pos.entry_px)
        pos.state = PositionState.TP1
    elif order.reason is ExitReason.TP2:
        pos.state = PositionState.TP2

    if pos.qty_open <= 0:
        pos.state = PositionState.CLOSED
