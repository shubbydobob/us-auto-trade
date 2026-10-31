"""Shared types for the strategy layer.

These structures cross the backtest/live boundary. The backtester fills them
from historical bars and the live loop from broker state, but `exits.py` and
`sizing.py` cannot tell the difference -- which is the invariant that makes the
backtest worth trusting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum


class PositionState(str, Enum):
    OPEN = "OPEN"
    TP1 = "TP1"
    TP2 = "TP2"
    CLOSED = "CLOSED"


class ExitReason(str, Enum):
    TP1 = "tp1"
    TP2 = "tp2"
    TRAIL = "trail"
    STOP = "stop"
    TIME = "time"
    THESIS = "thesis"
    EXHAUSTION = "exhaustion"


@dataclass
class Position:
    symbol: str
    opened_on: date
    entry_px: float
    qty_initial: int
    qty_open: int
    atr_at_entry: float
    stop_px: float
    state: PositionState = PositionState.OPEN
    atr_frozen: float | None = None          # frozen at TP1; see exits.py
    hh_since_entry: float = 0.0              # highest high
    hc_since_entry: float = 0.0              # highest close
    sessions_held: int = 0
    realised_pnl: float = 0.0
    fees_paid: float = 0.0

    @property
    def gain(self) -> float:
        return self.hc_since_entry / self.entry_px - 1 if self.entry_px else 0.0

    def gain_at(self, price: float) -> float:
        return price / self.entry_px - 1 if self.entry_px else 0.0

    @property
    def open_risk(self) -> float:
        """Dollars at risk if the stop is hit from here."""
        return max(0.0, (self.entry_px - self.stop_px) * self.qty_open)


@dataclass
class ExitOrder:
    reason: ExitReason
    qty: int
    limit_px: float
    # True when the level has already been breached and we are chasing it down,
    # which is what costs.stop_fill_price penalises.
    is_stop: bool = False


@dataclass
class Bar:
    """One session for one symbol, as-traded."""
    dt: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    atr14: float = 0.0
    sma20: float = 0.0
    rvol_5: float = 1.0
    cs_spread: float | None = None
    halted: bool = False
    limit_locked: bool = False
    luld_band: float = float("nan")


@dataclass
class SizingResult:
    shares: int
    notional: float
    stop_px: float
    risk_dollars: float
    binding_cap: str = ""
    rejected: bool = False
    reject_reason: str = ""
    caps: dict[str, float] = field(default_factory=dict)
