"""Position sizing with non-overridable caps.

The three liquidity/concentration caps are hard-coded here rather than read from
YAML. Config can tighten risk but must not be able to loosen the constraints
that keep the strategy inside what the market can actually absorb -- a penny
name's own order moving the price is not a hypothetical.

Small-account mode
------------------
At KRW 100,000 (~$72) the 1% rule produces a $2.90 position: six shares of a
$0.50 stock, where the spread alone exceeds the risk budget and no performance
statistic is meaningful. Below `small_account_usd` the engine therefore switches
to single-position full-allocation mode. That account is a plumbing test -- does
the order fill, does re-pegging work, what is the real slippage -- not a
performance test. Above the threshold the 1% rule applies unchanged, with no
code change required.
"""
from __future__ import annotations

import math

from uat.strategy.types import SizingResult

# Hard caps. Deliberately not configurable.
MAX_NOTIONAL_PCT_EQUITY = 0.08
MAX_PARTICIPATION = 0.01
MAX_NOTIONAL_PCT_ADDV = 0.02


def stop_distance(entry_px: float, atr14: float, atr_mult: float, min_pct: float) -> float:
    """Penny stops must be wide; a 2% stop is noise, not risk management."""
    return max(atr_mult * atr14, min_pct * entry_px)


def size_position(
    equity: float,
    entry_px: float,
    atr14: float,
    median_volume_20: float,
    addv_20: float,
    *,
    risk_per_trade: float = 0.01,
    atr_mult: float = 2.5,
    stop_min_pct: float = 0.20,
    small_account_usd: float = 1450.0,
    open_risk_total: float = 0.0,
    max_portfolio_heat: float = 0.05,
    min_fill_fraction: float = 0.5,
    single_position_mode: bool | None = None,
) -> SizingResult:
    """Shares to buy, after every cap. Shared by backtest and live."""
    if entry_px <= 0 or equity <= 0:
        return SizingResult(0, 0.0, 0.0, 0.0, rejected=True, reject_reason="bad_inputs")

    dist = stop_distance(entry_px, atr14, atr_mult, stop_min_pct)
    if dist <= 0:
        return SizingResult(0, 0.0, 0.0, 0.0, rejected=True, reject_reason="zero_stop_distance")
    stop_px = max(0.01, entry_px - dist)

    small = equity < small_account_usd if single_position_mode is None else single_position_mode

    if small:
        # Full allocation to one name; the stop still governs the loss, but the
        # 1% budget is abandoned because it cannot be expressed in whole shares.
        target_notional = equity
        risk_dollars = dist * (target_notional / entry_px)
        binding = "small_account_full_allocation"
    else:
        risk_dollars = equity * risk_per_trade
        headroom = max(0.0, equity * max_portfolio_heat - open_risk_total)
        # Reject rather than shrink. Squeezing a full-size idea into whatever
        # heat is left produces dust positions -- an earlier version averaged
        # $94 notional on a $7,246 account with 12% of capital deployed, so the
        # backtest was measuring an uninvested portfolio. A position that does
        # not fit the risk budget is a position you do not take.
        if headroom < risk_dollars * min_fill_fraction:
            return SizingResult(
                0, 0.0, stop_px, 0.0, rejected=True, reject_reason="portfolio_heat_exhausted"
            )
        risk_dollars = min(risk_dollars, headroom)
        target_notional = (risk_dollars / dist) * entry_px
        binding = "risk_budget"

    # Liquidity caps always apply -- being 20% of a thin name's daily volume is
    # how you discover your own order is the market.
    caps = {
        "participation_cap": max(0.0, median_volume_20) * MAX_PARTICIPATION * entry_px,
        "addv_cap": max(0.0, addv_20) * MAX_NOTIONAL_PCT_ADDV,
    }
    # The 8% concentration cap exists to stop one name dominating a diversified
    # book. In single-position plumbing mode concentration IS the design, so
    # applying it would silently shrink a $72 account's order to $5.76 and
    # defeat the mode entirely.
    if not small:
        caps["equity_cap"] = equity * MAX_NOTIONAL_PCT_EQUITY

    notional = target_notional
    for name, cap in caps.items():
        if cap < notional:
            notional, binding = cap, name

    shares = int(math.floor(notional / entry_px))
    if shares < 1:
        return SizingResult(
            0, 0.0, stop_px, 0.0, binding_cap=binding, rejected=True,
            reject_reason="below_one_share", caps=caps,
        )

    actual_notional = shares * entry_px
    return SizingResult(
        shares=shares,
        notional=actual_notional,
        stop_px=stop_px,
        risk_dollars=dist * shares,
        binding_cap=binding,
        caps=caps,
    )
