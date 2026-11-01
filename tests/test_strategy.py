"""Exits, sizing, costs and halts.

The exit tests matter most: `strategy/exits.py` is imported verbatim by both
`backtest/engine.py` and the live loop, so a behaviour change here silently
invalidates every backtest number.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from uat.backtest.costs import (
    CostParams,
    fees,
    fill_price,
    impact_bps,
    spread_bps,
    stop_fill_price,
)
from uat.backtest.halts import luld_band
from uat.strategy.exits import (
    ExitParams,
    apply_fill,
    evaluate_exits,
    trailing_stop,
    update_trackers,
)
from uat.strategy.sizing import (
    MAX_NOTIONAL_PCT_ADDV,
    MAX_NOTIONAL_PCT_EQUITY,
    MAX_PARTICIPATION,
    size_position,
)
from uat.strategy.types import Bar, ExitReason, Position, PositionState

P = ExitParams()


def make_position(entry=1.00, qty=1000, atr=0.10) -> Position:
    return Position(
        symbol="TEST",
        opened_on=date(2026, 1, 5),
        entry_px=entry,
        qty_initial=qty,
        qty_open=qty,
        atr_at_entry=atr,
        stop_px=entry - max(2.5 * atr, 0.20 * entry),
        hh_since_entry=entry,
        hc_since_entry=entry,
    )


def make_bar(high, low, close, *, open_=None, volume=1_000_000, sma20=0.0, rvol=1.0) -> Bar:
    return Bar(
        dt=date(2026, 1, 6),
        open=open_ if open_ is not None else close,
        high=high,
        low=low,
        close=close,
        volume=volume,
        atr14=0.10,
        sma20=sma20,
        rvol_5=rvol,
    )


# ---------------- exits ----------------

def test_stop_takes_priority_over_a_take_profit_on_the_same_bar():
    """A bar that gaps through the stop and later prints a TP high is a stop."""
    pos = make_position()
    bar = make_bar(high=1.40, low=0.60, close=1.35)  # stop is 0.75
    orders = evaluate_exits(pos, bar, prev_low=0.95, median_volume=500_000, params=P)
    assert len(orders) == 1
    assert orders[0].reason is ExitReason.STOP
    assert orders[0].is_stop


def test_tp1_freezes_atr_and_lifts_the_stop_to_breakeven():
    pos = make_position(entry=1.00, atr=0.10)
    bar = make_bar(high=1.35, low=1.05, close=1.30)
    orders = evaluate_exits(pos, bar, prev_low=1.00, median_volume=500_000, params=P)
    assert orders[0].reason is ExitReason.TP1

    apply_fill(pos, orders[0], 1.30, P)
    assert pos.state is PositionState.TP1
    assert pos.atr_frozen == pytest.approx(0.10)
    assert pos.stop_px == pytest.approx(1.00)      # breakeven
    assert pos.qty_open == pytest.approx(1000 - 333, abs=1)


def test_trail_uses_the_wider_of_percent_and_frozen_atr():
    """The core design call: ATR frozen at TP1, take whichever trail is wider.

    After a parabolic run the live ATR would be huge and the ATR trail would sit
    far below price, handing back the spike.
    """
    pos = make_position(entry=1.00, atr=0.10)
    pos.state = PositionState.TP1
    pos.atr_frozen = 0.10
    pos.hh_since_entry = 10.0
    pos.hc_since_entry = 9.0

    pct_trail = 9.0 * (1 - P.trail_pct_of_high)        # 6.75
    atr_trail = 10.0 - P.trail_atr_mult * 0.10          # 9.70
    assert trailing_stop(pos, P) == pytest.approx(max(pct_trail, atr_trail))
    assert trailing_stop(pos, P) == pytest.approx(9.70)


def test_trail_tightens_once_the_gain_exceeds_the_threshold():
    pos = make_position(entry=1.00, atr=0.10)
    pos.state = PositionState.TP1
    pos.atr_frozen = 0.10
    pos.hh_since_entry = 4.0
    pos.hc_since_entry = 4.0      # gain 300% > trail_tighten_above_gain (200%)

    expected_pct = 4.0 * (1 - P.trail_pct_tight)
    expected_atr = 4.0 - P.trail_atr_mult * 0.10
    assert trailing_stop(pos, P) == pytest.approx(max(expected_pct, expected_atr))


def test_exhaustion_requires_gain_reversal_and_volume_together():
    pos = make_position(entry=1.00)
    pos.hc_since_entry = 3.0
    # gain 180% > 150%, closes below yesterday's low, volume 4x median
    bar = make_bar(high=3.1, low=2.7, close=2.80, volume=4_000_000)
    orders = evaluate_exits(pos, bar, prev_low=2.90, median_volume=1_000_000, params=P)
    assert orders[0].reason is ExitReason.EXHAUSTION

    # Same bar but ordinary volume: not exhaustion.
    quiet = make_bar(high=3.1, low=2.7, close=2.80, volume=900_000)
    orders = evaluate_exits(pos, quiet, prev_low=2.90, median_volume=1_000_000, params=P)
    assert not orders or orders[0].reason is not ExitReason.EXHAUSTION


def test_time_stop_only_fires_after_the_session_count():
    pos = make_position()
    bar = make_bar(high=1.05, low=0.95, close=1.02)

    pos.sessions_held = P.time_stop_sessions - 2
    assert not evaluate_exits(pos, bar, prev_low=0.95, median_volume=1e6, params=P)

    pos.sessions_held = P.time_stop_sessions
    orders = evaluate_exits(pos, bar, prev_low=0.95, median_volume=1e6, params=P)
    assert orders[0].reason is ExitReason.TIME


def test_closed_position_emits_no_further_orders():
    pos = make_position()
    pos.qty_open = 0
    pos.state = PositionState.CLOSED
    assert evaluate_exits(pos, make_bar(1.5, 0.5, 1.4), prev_low=1.0,
                          median_volume=1e6, params=P) == []


def test_update_trackers_is_monotonic():
    pos = make_position()
    update_trackers(pos, make_bar(high=2.0, low=0.9, close=1.8))
    update_trackers(pos, make_bar(high=1.5, low=0.8, close=1.1))
    assert pos.hh_since_entry == pytest.approx(2.0)
    assert pos.hc_since_entry == pytest.approx(1.8)
    assert pos.sessions_held == 2


# ---------------- sizing ----------------

def test_hard_caps_cannot_be_exceeded():
    r = size_position(
        equity=100_000, entry_px=2.00, atr14=0.02,   # tiny ATR -> huge raw size
        median_volume_20=10_000_000, addv_20=20_000_000,
    )
    assert r.notional <= 100_000 * MAX_NOTIONAL_PCT_EQUITY + 2.00
    assert r.shares <= 10_000_000 * MAX_PARTICIPATION
    assert r.notional <= 20_000_000 * MAX_NOTIONAL_PCT_ADDV + 2.00


def test_illiquid_name_is_capped_by_participation_not_by_risk():
    r = size_position(
        equity=100_000, entry_px=1.00, atr14=0.05,
        median_volume_20=50_000, addv_20=50_000,     # very thin
    )
    assert r.binding_cap in ("participation_cap", "addv_cap")
    assert r.shares <= 50_000 * MAX_PARTICIPATION


def test_small_account_switches_to_single_position_mode():
    """At KRW 100,000 (~$72) the 1% rule yields a $2.90 position; the engine
    must switch to full allocation rather than emit an untradeable size."""
    small = size_position(equity=72, entry_px=0.50, atr14=0.05,
                          median_volume_20=5_000_000, addv_20=3_000_000)
    assert small.binding_cap == "small_account_full_allocation"
    assert small.notional == pytest.approx(72, rel=0.05)

    big = size_position(equity=14_493, entry_px=0.50, atr14=0.05,
                        median_volume_20=5_000_000, addv_20=3_000_000)
    assert big.binding_cap != "small_account_full_allocation"
    assert big.risk_dollars == pytest.approx(14_493 * 0.01, rel=0.10)


def test_portfolio_heat_exhaustion_rejects_new_positions():
    r = size_position(
        equity=10_000, entry_px=1.00, atr14=0.10,
        median_volume_20=5_000_000, addv_20=3_000_000,
        open_risk_total=600,          # already above 5% heat on 10k
        max_portfolio_heat=0.05,
    )
    assert r.rejected
    assert r.reject_reason == "portfolio_heat_exhausted"


def test_stop_distance_respects_the_minimum_percentage():
    r = size_position(equity=10_000, entry_px=1.00, atr14=0.001,
                      median_volume_20=5_000_000, addv_20=3_000_000)
    assert r.stop_px == pytest.approx(0.80)   # 20% floor, not 2.5 * tiny ATR


# ---------------- costs ----------------

def test_tick_floor_dominates_for_cheap_stocks():
    p = CostParams()
    # At $1.20 one cent is 83bps; a 0.2% statistical estimate must not win.
    assert spread_bps(1.20, cs_spread=0.002, p=p) == pytest.approx(83.33, rel=0.01)
    # At $50 the statistical estimate dominates.
    assert spread_bps(50.0, cs_spread=0.004, p=p) == pytest.approx(60.0, rel=0.01)


def test_impact_follows_the_square_root_law():
    p = CostParams()
    small = impact_bps(1_000, 1_000_000, p)
    big = impact_bps(4_000, 1_000_000, p)
    assert big == pytest.approx(2 * small, rel=1e-6)


def test_cost_multiplier_scales_slippage_and_commission():
    base = CostParams(cost_multiplier=1.0)
    stressed = CostParams(cost_multiplier=2.0)
    b = fill_price("BUY", 1.00, 1000, 1_000_000, 0.02, base)
    s = fill_price("BUY", 1.00, 1000, 1_000_000, 0.02, stressed)
    assert (s - 1.00) == pytest.approx(2 * (b - 1.00), rel=1e-6)
    assert fees("BUY", 1000, 1.00, stressed) == pytest.approx(
        2 * fees("BUY", 1000, 1.00, base)
    )


def test_buy_and_sell_slip_in_opposite_directions():
    p = CostParams()
    assert fill_price("BUY", 1.00, 100, 1_000_000, 0.02, p) > 1.00
    assert fill_price("SELL", 1.00, 100, 1_000_000, 0.02, p) < 1.00


def test_gap_through_fills_at_the_open_not_the_stop():
    """KIS has no stop-market order, so a gap below the stop fills at the open."""
    p = CostParams()
    gapped = stop_fill_price(stop_px=0.80, bar_open=0.60, p=p)
    assert gapped < 0.80
    assert gapped == pytest.approx(0.60 * (1 - p.stop_limit_penalty * 0.5), rel=1e-6)

    # No gap: still worse than the level, because a monitored limit chases.
    normal = stop_fill_price(stop_px=0.80, bar_open=0.95, p=p)
    assert normal == pytest.approx(0.80 * (1 - p.stop_limit_penalty), rel=1e-6)
    assert normal < 0.80


def test_sell_pays_regulatory_fees_and_buy_does_not():
    p = CostParams()
    assert fees("SELL", 1000, 2.00, p) > fees("BUY", 1000, 2.00, p)


# ---------------- halts ----------------

@pytest.mark.parametrize(
    "price,expected_single",
    [(5.00, 0.10), (3.00, 0.10), (2.00, 0.20), (0.80, 0.20), (0.50, 0.30)],
)
def test_luld_band_tiers(price, expected_single):
    assert luld_band(price, doubled=False) == pytest.approx(expected_single)
    assert luld_band(price, doubled=True) == pytest.approx(2 * expected_single)


def test_sub_75_cent_band_is_the_lesser_of_75pct_and_15_cents():
    # At $0.10, 15c/price = 1.5 which exceeds 75%, so 75% binds.
    assert luld_band(0.10, doubled=False) == pytest.approx(0.75)
    # At $0.50, 15c/price = 30% which is below 75%.
    assert luld_band(0.50, doubled=False) == pytest.approx(0.30)


def test_risk_budget_settings_are_internally_consistent():
    """max_concurrent * risk_per_trade must fit inside max_portfolio_heat.

    When they disagree the sizer has to resolve it somehow, and shrinking is
    the wrong answer: it silently produced $28-median positions on a $7,246
    account and made the backtest measure an uninvested portfolio.
    """
    from uat.config import load_settings

    s = load_settings("paper")
    n = s.get("risk.max_concurrent_positions")
    per = s.get("risk.risk_per_trade")
    heat = s.get("risk.max_portfolio_heat")
    assert n * per <= heat + 1e-9, (
        f"{n} positions x {per:.1%} risk = {n*per:.1%} exceeds the {heat:.1%} heat cap"
    )


def test_heat_exhaustion_rejects_rather_than_shrinking_to_dust():
    full = size_position(equity=10_000, entry_px=1.00, atr14=0.10,
                         median_volume_20=5_000_000, addv_20=3_000_000,
                         open_risk_total=0.0, max_portfolio_heat=0.05)
    assert not full.rejected

    # Only 10% of a full-size risk budget left: take nothing, not a dust position.
    nearly_full = size_position(equity=10_000, entry_px=1.00, atr14=0.10,
                                median_volume_20=5_000_000, addv_20=3_000_000,
                                open_risk_total=490, max_portfolio_heat=0.05)
    assert nearly_full.rejected
    assert nearly_full.reject_reason == "portfolio_heat_exhausted"


def test_trail_engages_without_the_tp_ladder():
    """Disabling the TP ladder must not silently disable the trail.

    The trail used to engage only once a TP tranche had fired, so a "let winners
    run" configuration degraded into "hold to the hard stop" -- three different
    trail widths produced byte-identical backtests before this was caught.
    """
    from uat.strategy.exits import trail_is_active

    no_ladder = ExitParams(tp1_gain=9.99, tp1_fraction=0.0,
                           tp2_gain=9.99, tp2_fraction=0.0,
                           trail_activate_gain=0.15)
    pos = make_position(entry=1.00, atr=0.10)

    pos.hc_since_entry = 1.10            # +10%, below the activation gain
    assert not trail_is_active(pos, no_ladder)

    pos.hc_since_entry = 1.25            # +25%, above it
    assert trail_is_active(pos, no_ladder)
    assert pos.state is PositionState.OPEN   # activated without any TP firing


def test_trail_width_actually_changes_the_exit_level():
    """Guards the specific symptom: different widths must give different stops."""
    from uat.strategy.exits import trailing_stop

    # Large ATR pushes the ATR leg far below price, so the percent leg binds.
    # (A small ATR does the opposite: 3 x 0.01 sits just under the high and wins
    # the max(), which is why an earlier version of this test compared equals.)
    pos = make_position(entry=1.00, atr=0.20)
    pos.hh_since_entry = 1.60
    pos.hc_since_entry = 1.60                   # +60%, below trail_tighten_above_gain

    wide = ExitParams(tp1_gain=9.99, tp1_fraction=0.0, trail_pct_of_high=0.35,
                      trail_activate_gain=0.15)
    tight = ExitParams(tp1_gain=9.99, tp1_fraction=0.0, trail_pct_of_high=0.20,
                       trail_activate_gain=0.15)
    assert trailing_stop(pos, tight) > trailing_stop(pos, wide)


def test_tight_trail_takes_over_above_the_tighten_threshold():
    """Past trail_tighten_above_gain the width switches to trail_pct_tight, so
    trail_pct_of_high stops mattering -- worth pinning so it is not mistaken
    for the activation bug."""
    from uat.strategy.exits import trailing_stop

    pos = make_position(entry=1.00, atr=0.01)
    pos.hh_since_entry = 3.00
    pos.hc_since_entry = 3.00                   # +200%, at the threshold

    a = ExitParams(tp1_gain=9.99, tp1_fraction=0.0, trail_pct_of_high=0.35,
                   trail_pct_tight=0.18, trail_activate_gain=0.15)
    b = ExitParams(tp1_gain=9.99, tp1_fraction=0.0, trail_pct_of_high=0.20,
                   trail_pct_tight=0.18, trail_activate_gain=0.15)
    assert trailing_stop(pos, a) == pytest.approx(trailing_stop(pos, b))


def test_trail_activation_defaults_to_tp1_gain():
    """Leaving trail_activate_gain unset preserves the original behaviour."""
    from uat.strategy.exits import trail_is_active

    p = ExitParams()          # trail_activate_gain is None
    pos = make_position(entry=1.00)
    pos.hc_since_entry = 1.00 + p.tp1_gain - 0.01
    assert not trail_is_active(pos, p)
    pos.hc_since_entry = 1.00 + p.tp1_gain + 0.01
    assert trail_is_active(pos, p)
