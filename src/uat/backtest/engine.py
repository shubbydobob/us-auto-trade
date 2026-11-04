"""Event-driven backtester.

Written from scratch rather than on vectorbt/backtrader/zipline: none of them
model LULD halts, participation caps, a limit-only broker, or a three-tranche
exit ladder without so much customisation that you have written the engine
anyway -- and the exit logic must be the *same code object* the live loop calls.

Entry model (daily-bar proxy for the intraday logic)
----------------------------------------------------
yfinance has ~7 days of minute history, so the real VWAP-reclaim / opening-range
trigger cannot be backtested over years (docs/data_biases.md). This proxy is
deliberately conservative:

    skip if open > prev_close * 1.20        -- already chased
    skip if open < prev_close * 0.92        -- thesis broken overnight
    skip if the bar is halted or limit-locked
    limit = min(open * 1.01, prev_close * 1.08)
    fill requires high >= limit
    fill at most 1% of the day's volume; carry the remainder up to 2 sessions

The participation cap alone materially reduces returns versus a naive backtest.
That is the point.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from uat.backtest.costs import CostParams, fees, fill_price, stop_fill_price
from uat.strategy.exits import ExitParams, apply_fill, evaluate_exits, update_trackers
from uat.strategy.sizing import size_position, stop_level
from uat.strategy.types import Bar, ExitReason, Position, PositionState


@dataclass(frozen=True)
class EntryParams:
    skip_open_above: float = 1.20
    skip_open_below: float = 0.92
    limit_over_open: float = 1.01
    limit_cap_over_close: float = 1.08
    participation: float = 0.01
    carry_sessions: int = 2

    @classmethod
    def from_settings(cls, settings) -> "EntryParams":
        e = settings.get("entry") or {}
        return cls(
            skip_open_above=e.get("bt_skip_open_above", 1.20),
            skip_open_below=e.get("bt_skip_open_below", 0.92),
            limit_over_open=e.get("bt_limit_over_open", 1.01),
            limit_cap_over_close=e.get("bt_limit_cap_over_close", 1.08),
            participation=settings.get("risk.max_participation", 0.01),
            carry_sessions=e.get("bt_carry_sessions", 2),
        )


@dataclass(frozen=True)
class PortfolioParams:
    initial_equity: float = 10_000.0
    risk_per_trade: float = 0.01
    atr_mult: float = 2.5
    stop_min_pct: float = 0.20
    small_account_usd: float = 1450.0
    max_concurrent: int = 10
    max_portfolio_heat: float = 0.05
    max_new_per_day: int = 3
    top_k: int = 15
    daily_loss_soft: float = -0.03
    drawdown_halt: float = -0.20
    # The live kill switch latches and needs a manual two-key re-enable, which
    # is correct for real money but turns a backtest into a measurement of
    # however many years preceded the first bad drawdown. Research runs disable
    # it to measure the signal; the report states when it WOULD have fired.
    enforce_drawdown_halt: bool = True

    @classmethod
    def from_settings(cls, settings, initial_equity: float | None = None) -> "PortfolioParams":
        r = settings.get("risk") or {}
        return cls(
            initial_equity=initial_equity or float(r.get("equity_usd_override") or 10_000),
            risk_per_trade=r.get("risk_per_trade", 0.01),
            atr_mult=r.get("stop_atr_mult", 2.5),
            stop_min_pct=r.get("stop_min_pct", 0.20),
            small_account_usd=r.get("small_account_usd", 1450.0),
            max_concurrent=r.get("max_concurrent_positions", 10),
            max_portfolio_heat=r.get("max_portfolio_heat", 0.05),
            max_new_per_day=r.get("max_new_entries_per_day", 3),
            daily_loss_soft=r.get("daily_loss_soft", -0.03),
            drawdown_halt=r.get("drawdown_halt", -0.20),
        )


@dataclass
class Trade:
    symbol: str
    entry_dt: date
    exit_dt: date
    entry_px: float
    exit_px: float
    qty: int
    gross_pnl: float
    fees: float
    net_pnl: float
    return_pct: float
    sessions_held: int
    exit_reason: str
    mae: float
    mfe: float


@dataclass
class PendingEntry:
    symbol: str
    signal_dt: date
    sessions_waited: int = 0


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.DataFrame
    skipped: pd.DataFrame
    params: dict = field(default_factory=dict)


class Backtester:
    def __init__(
        self,
        bars: pd.DataFrame,
        signals: pd.DataFrame,
        cost: CostParams,
        entry: EntryParams,
        exits: ExitParams,
        portfolio: PortfolioParams,
    ):
        """bars: long-format as-traded bars with features + halt flags.
        signals: columns (dt, symbol, score), already ranked."""
        self.cost = cost
        self.entry = entry
        self.exit_params = exits
        self.pp = portfolio

        self.bars = bars.sort_values(["symbol", "dt"]).reset_index(drop=True)
        self._index = {
            (r.symbol, r.dt): i for i, r in enumerate(self.bars.itertuples(index=False))
        }
        self._by_symbol = {
            sym: g.reset_index(drop=True)
            for sym, g in self.bars.groupby("symbol", sort=False)
        }
        self._row_of = {
            sym: {d: i for i, d in enumerate(g["dt"])} for sym, g in self._by_symbol.items()
        }
        self.signals = signals.sort_values(["dt", "score"], ascending=[True, False])
        self.sessions = sorted(self.bars["dt"].unique())

    # ---------------- bar access ----------------

    def _bar(self, symbol: str, dt: date) -> pd.Series | None:
        g = self._by_symbol.get(symbol)
        if g is None:
            return None
        i = self._row_of[symbol].get(dt)
        return None if i is None else g.iloc[i]

    def _prev_bar(self, symbol: str, dt: date) -> pd.Series | None:
        g = self._by_symbol.get(symbol)
        i = self._row_of.get(symbol, {}).get(dt)
        if g is None or i is None or i == 0:
            return None
        return g.iloc[i - 1]

    @staticmethod
    def _to_bar(row: pd.Series) -> Bar:
        return Bar(
            dt=row["dt"],
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
            atr14=float(row.get("atr14", 0.0) or 0.0),
            sma20=float(row.get("sma20", 0.0) or 0.0),
            rvol_5=float(row.get("rvol_5", 1.0) or 1.0),
            cs_spread=row.get("cs_spread_20"),
            halted=bool(row.get("halted", False)),
            limit_locked=bool(row.get("limit_locked", False)),
        )

    # ---------------- main loop ----------------

    def run(self) -> BacktestResult:
        cash = self.pp.initial_equity
        equity = self.pp.initial_equity
        hwm = equity

        positions: dict[str, Position] = {}
        entry_meta: dict[str, dict] = {}
        pending: list[PendingEntry] = []
        trades: list[Trade] = []
        skipped: list[dict] = []
        curve: list[dict] = []
        halted_globally = False
        first_halt_dt: date | None = None

        signals_by_dt = {dt: g for dt, g in self.signals.groupby("dt", sort=False)}

        for dt in self.sessions:
            day_start_equity = equity

            # ---- 1. exits on today's bar, for positions opened earlier ----
            for symbol in list(positions):
                pos = positions[symbol]
                row = self._bar(symbol, dt)
                if row is None:
                    continue
                if pos.opened_on >= dt:
                    continue

                bar = self._to_bar(row)
                update_trackers(pos, bar)
                prev = self._prev_bar(symbol, dt)
                prev_low = float(prev["low"]) if prev is not None else bar.low
                med_vol = float(row.get("medvol_20", 0.0) or 0.0)

                orders = evaluate_exits(
                    pos, bar, prev_low=prev_low, median_volume=med_vol,
                    params=self.exit_params,
                )
                for order in orders:
                    if order.is_stop:
                        px = stop_fill_price(order.limit_px, bar.open, self.cost)
                    else:
                        px = fill_price(
                            "SELL", order.limit_px, order.qty, bar.volume,
                            bar.cs_spread, self.cost,
                        )
                    px = max(px, 0.01)
                    fee = fees("SELL", order.qty, px, self.cost)
                    qty = min(order.qty, pos.qty_open)

                    cash += qty * px - fee
                    pos.fees_paid += fee
                    apply_fill(pos, order, px, self.exit_params)

                    meta = entry_meta[symbol]
                    meta["exit_px"] = px
                    meta["exit_reason"] = order.reason.value
                    meta["exit_dt"] = dt

                meta = entry_meta[symbol]
                meta["mae"] = min(meta["mae"], bar.low / pos.entry_px - 1)
                meta["mfe"] = max(meta["mfe"], bar.high / pos.entry_px - 1)

                if pos.state is PositionState.CLOSED:
                    trades.append(self._close_trade(pos, meta))
                    del positions[symbol]
                    del entry_meta[symbol]

            # ---- 2. mark to market ----
            mtm = 0.0
            for symbol, pos in positions.items():
                row = self._bar(symbol, dt)
                px = float(row["close"]) if row is not None else pos.entry_px
                mtm += pos.qty_open * px
            equity = cash + mtm
            hwm = max(hwm, equity)
            drawdown = equity / hwm - 1

            curve.append(
                {"dt": dt, "cash": cash, "mtm": mtm, "equity": equity,
                 "hwm": hwm, "drawdown": drawdown, "n_positions": len(positions)}
            )

            # ---- 3. circuit breakers ----
            if drawdown <= self.pp.drawdown_halt:
                if first_halt_dt is None:
                    first_halt_dt = dt
                if self.pp.enforce_drawdown_halt:
                    halted_globally = True
            day_pnl = equity / day_start_equity - 1 if day_start_equity else 0.0
            blocked = halted_globally or day_pnl <= self.pp.daily_loss_soft

            # ---- 4. attempt pending entries, then today's new signals ----
            opened_today = 0
            still_pending: list[PendingEntry] = []

            for cand in pending:
                if blocked or opened_today >= self.pp.max_new_per_day:
                    cand.sessions_waited += 1
                    if cand.sessions_waited <= self.entry.carry_sessions:
                        still_pending.append(cand)
                    continue
                ok, reason = self._try_enter(
                    cand.symbol, dt, positions, entry_meta, equity, cash
                )
                if ok:
                    cash -= ok["cost"]
                    opened_today += 1
                else:
                    skipped.append({"dt": dt, "symbol": cand.symbol, "reason": reason})
                    cand.sessions_waited += 1
                    if cand.sessions_waited <= self.entry.carry_sessions:
                        still_pending.append(cand)
            pending = still_pending

            todays = signals_by_dt.get(dt)
            if todays is not None and not blocked:
                for sym in todays["symbol"].head(self.pp.top_k):
                    if opened_today >= self.pp.max_new_per_day:
                        break
                    if sym in positions or any(p.symbol == sym for p in pending):
                        continue
                    pending.append(PendingEntry(symbol=sym, signal_dt=dt))

        return BacktestResult(
            trades=pd.DataFrame([t.__dict__ for t in trades]),
            equity=pd.DataFrame(curve),
            skipped=pd.DataFrame(skipped),
            params={
                "cost_multiplier": self.cost.cost_multiplier,
                "initial_equity": self.pp.initial_equity,
                "top_k": self.pp.top_k,
                "first_drawdown_halt": first_halt_dt,
                "halt_enforced": self.pp.enforce_drawdown_halt,
            },
        )

    # ---------------- entry ----------------

    def _try_enter(self, symbol, dt, positions, entry_meta, equity, cash):
        if len(positions) >= self.pp.max_concurrent:
            return None, "max_concurrent"

        row = self._bar(symbol, dt)
        if row is None:
            return None, "no_bar"
        prev = self._prev_bar(symbol, dt)
        if prev is None:
            return None, "no_prev_bar"

        if bool(row.get("halted", False)) or bool(row.get("limit_locked", False)):
            return None, "halted"

        prev_close = float(prev["close"])
        o, h, v = float(row["open"]), float(row["high"]), float(row["volume"])
        if prev_close <= 0 or v <= 0:
            return None, "bad_bar"
        if o > prev_close * self.entry.skip_open_above:
            return None, "gap_up_chased"
        if o < prev_close * self.entry.skip_open_below:
            return None, "gap_down_thesis_broken"

        limit_px = min(o * self.entry.limit_over_open,
                       prev_close * self.entry.limit_cap_over_close)
        if h < limit_px:
            return None, "limit_not_touched"

        open_risk_total = sum(p.open_risk for p in positions.values())
        sizing = size_position(
            equity=equity,
            entry_px=limit_px,
            atr14=float(row.get("atr14", 0.0) or 0.0),
            median_volume_20=float(row.get("medvol_20", 0.0) or 0.0),
            addv_20=float(row.get("addv_20", 0.0) or 0.0),
            risk_per_trade=self.pp.risk_per_trade,
            atr_mult=self.pp.atr_mult,
            stop_min_pct=self.pp.stop_min_pct,
            small_account_usd=self.pp.small_account_usd,
            open_risk_total=open_risk_total,
            max_portfolio_heat=self.pp.max_portfolio_heat,
        )
        if sizing.rejected or sizing.shares < 1:
            return None, sizing.reject_reason or "sizing_rejected"

        # Participation cap on the actual fill, not just on the target size.
        max_shares_today = int(v * self.entry.participation)
        shares = min(sizing.shares, max_shares_today)
        if shares < 1:
            return None, "participation_cap"

        px = fill_price("BUY", limit_px, shares, v, row.get("cs_spread_20"), self.cost)
        fee = fees("BUY", shares, px, self.cost)
        total_cost = shares * px + fee
        if total_cost > cash:
            shares = int((cash - fee) / px) if px > 0 else 0
            if shares < 1:
                return None, "insufficient_cash"
            total_cost = shares * px + fees("BUY", shares, px, self.cost)

        atr = float(row.get("atr14", 0.0) or 0.0)
        pos = Position(
            symbol=symbol,
            opened_on=dt,
            entry_px=px,
            qty_initial=shares,
            qty_open=shares,
            atr_at_entry=atr,
            # stop_level, not a second formula. This line used to read
            #   max(0.01, px - max(atr_mult * atr, stop_min_pct * px))
            # which skips the 50%-of-price clamp commit 80f60b4 added to
            # sizing.py. On a name whose ATR is 40% of price, 2.5x ATR is a
            # 100%-of-price distance: the stop floored to $0.01 while the
            # position was sized as though it had a 50% stop. Measured on a
            # $1.00/0.40-ATR entry: $0.015 against the sizer's $0.5075.
            stop_px=stop_level(px, atr, self.pp.atr_mult, self.pp.stop_min_pct),
            hh_since_entry=float(row["high"]),
            hc_since_entry=float(row["close"]),
        )
        pos.fees_paid = fee
        positions[symbol] = pos
        entry_meta[symbol] = {
            "entry_dt": dt, "exit_dt": dt, "exit_px": px,
            "exit_reason": "open", "mae": 0.0, "mfe": 0.0,
        }
        return {"cost": total_cost}, ""

    @staticmethod
    def _close_trade(pos: Position, meta: dict) -> Trade:
        gross = pos.realised_pnl
        net = gross - pos.fees_paid
        notional = pos.entry_px * pos.qty_initial
        return Trade(
            symbol=pos.symbol,
            entry_dt=meta["entry_dt"],
            exit_dt=meta["exit_dt"],
            entry_px=pos.entry_px,
            exit_px=meta["exit_px"],
            qty=pos.qty_initial,
            gross_pnl=gross,
            fees=pos.fees_paid,
            net_pnl=net,
            return_pct=net / notional if notional else 0.0,
            sessions_held=pos.sessions_held,
            exit_reason=meta["exit_reason"],
            mae=meta["mae"],
            mfe=meta["mfe"],
        )
