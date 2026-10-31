"""Transaction cost model.

This is the single largest determinant of whether the strategy is real. The
Phase 2 signal selects $1.53 median price on $1.15M median ADDV -- names where
the quoted spread alone can exceed 2%, and where one cent of tick is 65bps.

Components, per side:

    spread_bps = max(corwin_schultz * multiplier, tick / price)   -- half charged
    impact_bps = 1e4 * k * sqrt(shares / day_volume)              -- square-root law
    commission = 0.25% (KIS US online, no minimum)
    reg fees   = SEC fee + FINRA TAF, sells only

FX is modelled once at funding, not per trade: KIS lets you hold a standing USD
balance, and converting per round trip would cost roughly 1% and dominate
everything else here.

`cost_multiplier` is the stress knob. The Phase 3 gate is survival at 1.5x --
because the Corwin-Schultz estimator is uncalibrated (see docs/data_biases.md),
passing only at 1.0x would be passing on an assumption.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BPS = 1e-4


@dataclass(frozen=True)
class CostParams:
    commission_per_side: float = 0.0025
    sec_fee_rate: float = 0.0000278
    finra_taf_per_share: float = 0.000166
    finra_taf_cap: float = 8.30
    impact_k: float = 0.4
    cs_spread_multiplier: float = 1.5
    tick_size: float = 0.01
    stop_limit_penalty: float = 0.02
    cost_multiplier: float = 1.0

    @classmethod
    def from_settings(cls, settings) -> "CostParams":
        c = settings.get("costs") or {}
        return cls(**{k: v for k, v in c.items() if k in cls.__dataclass_fields__})


def spread_bps(price: float, cs_spread: float | None, p: CostParams) -> float:
    """Effective half-turn spread in basis points, with a tick-size floor.

    At $1.20 a single cent is 83bps, so the tick floor -- not the statistical
    estimate -- is what binds for the cheapest names.
    """
    if price <= 0:
        return 0.0
    tick_floor = (p.tick_size / price) * 1e4
    estimated = (cs_spread or 0.0) * p.cs_spread_multiplier * 1e4
    return max(estimated, tick_floor)


def impact_bps(shares: float, day_volume: float, p: CostParams) -> float:
    """Square-root market impact. Zero if the order is vanishingly small."""
    if day_volume <= 0 or shares <= 0:
        return 0.0
    return 1e4 * p.impact_k * float(np.sqrt(shares / day_volume))


def slippage_bps(
    price: float, shares: float, day_volume: float, cs_spread: float | None, p: CostParams
) -> float:
    """Half the spread plus impact, scaled by the stress multiplier."""
    half_spread = 0.5 * spread_bps(price, cs_spread, p)
    return (half_spread + impact_bps(shares, day_volume, p)) * p.cost_multiplier


def fill_price(
    side: str,
    reference_px: float,
    shares: float,
    day_volume: float,
    cs_spread: float | None,
    p: CostParams,
) -> float:
    """Reference price moved against you by the slippage estimate."""
    slip = slippage_bps(reference_px, shares, day_volume, cs_spread, p) * BPS
    return reference_px * (1 + slip) if side == "BUY" else reference_px * (1 - slip)


def fees(side: str, shares: float, price: float, p: CostParams) -> float:
    """Commission plus regulatory fees for one execution."""
    notional = shares * price
    total = notional * p.commission_per_side * p.cost_multiplier
    if side == "SELL":
        total += notional * p.sec_fee_rate
        total += min(shares * p.finra_taf_per_share, p.finra_taf_cap)
    return total


def stop_fill_price(stop_px: float, bar_open: float, p: CostParams) -> float:
    """Realised exit price for a stop, given KIS has no stop-market order.

    Two distinct penalties:
      * gap-through -- the bar opened below the stop, so you fill at the open,
        not at the stop level
      * monitored-limit slippage -- even without a gap, a watched limit submitted
        after the level breaks fills worse than the level
    """
    if bar_open <= stop_px:
        return bar_open * (1 - p.stop_limit_penalty * p.cost_multiplier * 0.5)
    return stop_px * (1 - p.stop_limit_penalty * p.cost_multiplier)


def round_trip_cost_pct(
    price: float, shares: float, day_volume: float, cs_spread: float | None, p: CostParams
) -> float:
    """Total round-trip drag as a fraction of notional, for reporting."""
    slip = slippage_bps(price, shares, day_volume, cs_spread, p) * BPS
    comm = p.commission_per_side * p.cost_multiplier * 2
    return 2 * slip + comm
