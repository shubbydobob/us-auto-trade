"""Broker holdings parsing, position adoption, and the shared-exit inputs.

The payload below is a verbatim capture from the live account on 2026-11-04
holding 7 QTEX, not a hand-written fixture. That matters because two of KIS's
field names mean something other than what they say, and the only way to know
is to look at a real response:

    output2.tot_evlu_pfls_amt  = 7.84  -> market VALUE (7 x $1.12), not P&L
    output2.frcr_pchs_amt1     = 8.47  -> USD (frcr = 외화), not KRW

An earlier version of the balance report read the first as profit and divided
the second by the FX rate, turning an $8.47 cost basis into one cent. These
tests exist so that cannot come back.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import pytest

from uat.broker.kis_client import kis_num
from uat.features.pipeline import compute_symbol_features
from uat.live.context import atr_on, high_water
from uat.live.holdings import as_reconcile_input, parse_holdings, parse_summary
from uat.live.positions import sessions_since
from uat.strategy.sizing import size_position, stop_level

LIVE_PAYLOAD = {
    "output1": [
        {
            "cano": "63375318",
            "acnt_prdt_cd": "01",
            "prdt_type_cd": "512",
            "ovrs_pdno": "QTEX",
            "ovrs_item_name": "QTREX QUANTUM LTD",
            "frcr_evlu_pfls_amt": "-0.630000",
            "evlu_pfls_rt": "-7.44",
            "pchs_avg_pric": "1.2100",
            "ovrs_cblc_qty": "7",
            "ord_psbl_qty": "7",
            "frcr_pchs_amt1": "8.47000",
            "ovrs_stck_evlu_amt": "7.84000000",
            "now_pric2": "1.120000",
            "tr_crcy_cd": "USD",
            "ovrs_excg_cd": "NASD",
            "loan_type_cd": "10",
            "loan_dt": "",
            "expd_dt": "",
        }
    ],
    "output2": {
        "frcr_pchs_amt1": "8.47000",
        "ovrs_rlzt_pfls_amt": "0.00000",
        "ovrs_tot_pfls": "-0.63000",
        "rlzt_erng_rt": "0.00000000",
        "tot_evlu_pfls_amt": "7.84000000",
        "tot_pftrt": "-7.43801653",
        "frcr_buy_amt_smtl1": "0.000000",
        "ovrs_rlzt_pfls_amt2": "0.000000",
    },
    "rt_cd": "0",
}


# ---------------------------------------------------------------- holdings


def test_live_payload_parses_to_one_holding():
    (h,) = parse_holdings(LIVE_PAYLOAD)
    assert (h.symbol, h.qty, h.venue) == ("QTEX", 7, "NASDAQ")
    assert h.avg_px == pytest.approx(1.21)
    assert h.last_px == pytest.approx(1.12)
    assert h.pnl_usd == pytest.approx(-0.63)
    assert h.pnl_pct == pytest.approx(-7.44)
    assert h.cost_basis == pytest.approx(8.47)
    assert h.market_value == pytest.approx(7.84)


def test_nothing_is_reserved_when_the_whole_position_is_sellable():
    (h,) = parse_holdings(LIVE_PAYLOAD)
    assert h.qty_reserved == 0


def test_reserved_shares_reveal_a_live_resting_sell():
    """Held minus sellable is how the exit loop sees its own protective order.

    docs/exit_loop.md leaves this as an open question; the arithmetic has to be
    right before the answer can be measured.
    """
    raw = {"output1": [dict(LIVE_PAYLOAD["output1"][0], ord_psbl_qty="2")]}
    (h,) = parse_holdings(raw)
    assert (h.qty, h.qty_sellable, h.qty_reserved) == (7, 2, 5)


def test_zero_quantity_rows_are_dropped():
    raw = {"output1": [dict(LIVE_PAYLOAD["output1"][0], ovrs_cblc_qty="0")]}
    assert parse_holdings(raw) == []


def test_missing_sellable_field_defaults_to_the_held_quantity():
    row = {k: v for k, v in LIVE_PAYLOAD["output1"][0].items() if k != "ord_psbl_qty"}
    (h,) = parse_holdings({"output1": [row]})
    assert h.qty_sellable == 7


def test_empty_payload_is_not_an_error():
    assert parse_holdings({}) == []
    assert parse_holdings({"output1": None}) == []


@pytest.mark.parametrize(
    "value, expected",
    [("1,234.56", 1234.56), ("-", 0.0), ("", 0.0), ("  7 ", 7.0), ("abc", 0.0)],
)
def test_kis_num_tolerates_commas_dashes_and_junk(value, expected):
    assert kis_num({"x": value}, "x") == pytest.approx(expected)


def test_kis_num_falls_through_to_the_next_candidate_key():
    # Paper and live name the same quantity differently, so callers pass both.
    assert kis_num({"a": "-", "b": "3.5"}, "a", "b") == pytest.approx(3.5)


# ---------------------------------------------------------------- summary


def test_total_evaluation_amount_is_market_value_not_pnl():
    """The regression this file exists for.

    tot_evlu_pfls_amt (7.84) is the position's market value; P&L is
    ovrs_tot_pfls (-0.63). Reading the first as profit turns a losing position
    into a winning one on screen.
    """
    s = parse_summary(LIVE_PAYLOAD)
    assert s.market_value == pytest.approx(7.84)
    assert s.unrealised_pnl == pytest.approx(-0.63)
    assert s.unrealised_pnl < 0 < s.market_value


def test_summary_amounts_are_usd():
    """frcr = 외화. Cost basis must reconcile with quantity x average price."""
    s = parse_summary(LIVE_PAYLOAD)
    (h,) = parse_holdings(LIVE_PAYLOAD)
    assert s.cost_basis == pytest.approx(h.qty * h.avg_px, abs=0.01)
    assert s.return_pct == pytest.approx(-7.438, abs=0.01)


def test_summary_accepts_a_list_shaped_output2():
    raw = {"output2": [LIVE_PAYLOAD["output2"]]}
    assert parse_summary(raw).market_value == pytest.approx(7.84)


def test_reconcile_input_is_keyed_by_symbol():
    assert as_reconcile_input(parse_holdings(LIVE_PAYLOAD)) == {"QTEX": (7, 1.21)}


# ---------------------------------------------------------------- adoption


def test_adopted_stop_matches_what_the_sizer_would_have_chosen():
    """A hand-entered position and a system-entered one must get the same stop.

    `adopt_holding` calls `stop_level` and so does `size_position`; if they ever
    diverge, the same symbol carries two different stops depending on how it
    was bought.
    """
    entry, atr = 1.21, 0.1663
    sized = size_position(
        equity=8.47, entry_px=entry, atr14=atr,
        median_volume_20=2_000_000, addv_20=2_500_000,
        single_position_mode=True,
    )
    assert stop_level(entry, atr) == pytest.approx(sized.stop_px)


def test_stop_is_anchored_to_entry_date_atr_not_todays():
    """Adopting with today's ATR would set a stop the entry never implied."""
    ctx = _ctx_from(
        dts=[date(2026, 11, 2), date(2026, 11, 3)],
        atrs=[0.1663, 0.1594],
    )
    assert atr_on(ctx, date(2026, 11, 2)) == pytest.approx(0.1663)
    assert atr_on(ctx, date(2026, 11, 3)) == pytest.approx(0.1594)
    # A date before any feature row falls back to the newest bar's ATR rather
    # than returning zero, which the caller treats as "refuse to adopt".
    assert atr_on(ctx, date(2026, 1, 1)) == pytest.approx(0.1594)


def test_high_water_replays_only_sessions_after_entry():
    ctx = _ctx_from(
        dts=[date(2026, 11, 2), date(2026, 11, 3)],
        atrs=[0.1663, 0.1594],
        highs=[1.25, 1.17],
        closes=[1.12, 1.12],
    )
    hh, hc, held = high_water(ctx, date(2026, 11, 2))
    # The entry session itself is excluded, exactly as the engine advances the
    # trackers from the bar AFTER entry.
    assert (hh, hc, held) == (pytest.approx(1.17), pytest.approx(1.12), 1)


def test_high_water_is_empty_on_the_entry_session():
    ctx = _ctx_from(dts=[date(2026, 11, 2)], atrs=[0.1663])
    assert high_water(ctx, date(2026, 11, 2)) == (0.0, 0.0, 0)


def test_sessions_since_counts_trading_days_only():
    # 2026-11-02 Mon -> 2026-11-06 Fri is four sessions, weekend excluded.
    assert sessions_since(date(2026, 11, 2), date(2026, 11, 6)) == 4
    assert sessions_since(date(2026, 11, 2), date(2026, 11, 2)) == 0
    assert sessions_since(date(2026, 11, 2), date(2026, 10, 30)) == 0


# ---------------------------------------------------------------- shared inputs


def test_feature_panel_emits_sma20_for_the_thesis_break_exit():
    """exits.py skips thesis-break entirely when bar.sma20 is zero.

    sma20 used to be computed in scripts/run_backtest.py only, so every other
    consumer of the panel -- the live loop, the position view -- silently lost
    an exit worth -$2,908 over 83 trades in the Phase 3 run.
    """
    n = 60
    bars = pd.DataFrame(
        {
            "symbol": ["TEST"] * n,
            "dt": pd.bdate_range("2026-01-05", periods=n).date,
            "open": np.linspace(1.0, 2.0, n),
            "high": np.linspace(1.05, 2.05, n),
            "low": np.linspace(0.95, 1.95, n),
            "close": np.linspace(1.0, 2.0, n),
            "volume": np.full(n, 1_000_000),
            "adj_factor": np.ones(n),
        }
    )
    feats = compute_symbol_features(bars)
    assert "sma20" in feats.columns
    tail = feats["sma20"].iloc[-1]
    assert tail > 0
    # Rising series: the mean of the last 20 closes sits below the last close.
    assert tail < bars["close"].iloc[-1]


# ---------------------------------------------------------------- helpers


@dataclass
class _Ctx:
    """Minimal stand-in for SymbolContext; atr_on/high_water read only these."""

    history: pd.DataFrame
    features: pd.DataFrame
    bar: object


def _ctx_from(dts, atrs, highs=None, closes=None) -> _Ctx:
    highs = highs if highs is not None else [1.0] * len(dts)
    closes = closes if closes is not None else [1.0] * len(dts)
    history = pd.DataFrame({"dt": dts, "high": highs, "close": closes})
    features = pd.DataFrame({"dt": dts, "atr14": atrs})

    class _Bar:
        atr14 = atrs[-1]

    return _Ctx(history=history, features=features, bar=_Bar())
