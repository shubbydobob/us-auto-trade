"""Position persistence.

The exit loop must be able to reconstruct its plan after a crash, a reboot, or
simply the next scheduled run. In-memory state is not acceptable: the US
session runs 23:30-06:00 KST, so the machine is usually unattended and a
process that dies mid-session must not lose its stop level.

Two sources of truth, deliberately separated:

* **KIS owns the share count.** Quantity and average price are reconciled from
  the broker on every startup; if the two disagree, the broker wins and the
  discrepancy is logged rather than silently patched.
* **We own the exit plan.** The stop level, the frozen ATR and the high-water
  marks exist nowhere at the broker, so they live in pos.positions and would be
  unrecoverable if lost.
"""
from __future__ import annotations

import json
import uuid
from datetime import date, datetime

import pandas as pd

from uat.logging import get
from uat.storage.db import query, upsert_df
from uat.strategy.types import Position, PositionState

log = get("live.positions")


def new_position_id(symbol: str, opened: date) -> str:
    return f"{symbol}-{opened:%Y%m%d}-{uuid.uuid4().hex[:6]}"


def save(pos: Position, position_id: str | None = None, exchange: str = "") -> str:
    pid = position_id or new_position_id(pos.symbol, pos.opened_on)
    row = pd.DataFrame(
        [
            {
                "position_id": pid,
                "symbol": pos.symbol,
                "opened_at": datetime.combine(pos.opened_on, datetime.min.time()),
                "closed_at": None,
                "qty_open": int(pos.qty_open),
                "qty_initial": int(pos.qty_initial),
                "avg_px": float(pos.entry_px),
                "stop_px": float(pos.stop_px),
                "tp_ladder": json.dumps({"exchange": exchange}),
                "atr_at_entry": float(pos.atr_at_entry),
                "atr_frozen": pos.atr_frozen,
                "hh_since_entry": float(pos.hh_since_entry),
                "hc_since_entry": float(pos.hc_since_entry),
                "state": pos.state.value,
            }
        ]
    )
    upsert_df("pos.positions", row, ["position_id"])
    log.info(
        "position saved",
        extra={"extra_fields": {"id": pid, "symbol": pos.symbol,
                                "qty": pos.qty_open, "stop": pos.stop_px}},
    )
    return pid


def load_open() -> dict[str, Position]:
    """Rebuild open positions from the database, keyed by position_id."""
    df = query("SELECT * FROM pos.positions WHERE state != 'CLOSED' AND qty_open > 0")
    out: dict[str, Position] = {}
    for r in df.itertuples(index=False):
        out[r.position_id] = Position(
            symbol=r.symbol,
            opened_on=pd.Timestamp(r.opened_at).date(),
            entry_px=float(r.avg_px),
            qty_initial=int(r.qty_initial),
            qty_open=int(r.qty_open),
            atr_at_entry=float(r.atr_at_entry or 0.0),
            stop_px=float(r.stop_px or 0.0),
            state=PositionState(r.state),
            atr_frozen=float(r.atr_frozen) if r.atr_frozen is not None else None,
            hh_since_entry=float(r.hh_since_entry or 0.0),
            hc_since_entry=float(r.hc_since_entry or 0.0),
        )
    return out


def record_order(
    order_id: str,
    symbol: str,
    side: str,
    qty: int,
    limit_px: float,
    status: str,
    kis_order_no: str = "",
    reason: str = "entry",
    asof: date | None = None,
) -> None:
    now = datetime.now()
    upsert_df(
        "exec.orders",
        pd.DataFrame(
            [
                {
                    "order_id": order_id,
                    "asof_dt": asof or date.today(),
                    "symbol": symbol,
                    "side": side,
                    "qty": int(qty),
                    "limit_px": float(limit_px),
                    "status": status,
                    "kis_order_no": kis_order_no,
                    "reason": reason,
                    "created_at": now,
                    "updated_at": now,
                }
            ]
        ),
        ["order_id"],
    )


def reconcile(broker_holdings: dict[str, tuple[int, float]]) -> list[str]:
    """Compare stored positions against the broker. Returns mismatch messages.

    The broker is authoritative on quantity. A mismatch halts new entries
    rather than being auto-corrected, because the usual cause is an exit that
    fired without being recorded, and guessing which is worse than stopping.
    """
    issues: list[str] = []
    stored = load_open()
    by_symbol: dict[str, int] = {}
    for p in stored.values():
        by_symbol[p.symbol] = by_symbol.get(p.symbol, 0) + p.qty_open

    for sym, qty in by_symbol.items():
        broker_qty = broker_holdings.get(sym, (0, 0.0))[0]
        if broker_qty != qty:
            issues.append(f"{sym}: stored {qty} shares, broker reports {broker_qty}")
    for sym, (qty, _) in broker_holdings.items():
        if qty > 0 and sym not in by_symbol:
            issues.append(f"{sym}: broker holds {qty} shares with no stored position")

    if issues:
        log.error("reconciliation mismatch", extra={"extra_fields": {"issues": issues}})
    return issues
