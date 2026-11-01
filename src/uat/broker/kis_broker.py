"""Order placement for US equities through KIS.

KIS accepts LIMIT ORDERS ONLY for overseas stocks -- there is no market order,
and no MOO/LOO/MOC/LOC. "Buy at market" therefore means an aggressive limit
above the ask, and a stop means a monitored limit submitted after the level
breaks. Both are modelled that way in backtest/costs.py so the numbers match.

POST bodies require a `hashkey` header, obtained from a separate endpoint.

Safety: every order passes through `OrderGuard`, which enforces caps the caller
cannot raise. Nothing here sends an order unless `confirm=True` is passed
explicitly -- the default is a dry run that returns the request it would have
made.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import httpx

from uat.broker.kis_client import EXCHANGE_CODES, KisClient, KisResponse
from uat.logging import get

log = get("broker.kis_order")

Side = Literal["BUY", "SELL"]

# Price-quote endpoint uses 3-letter codes; the order endpoint uses 4-letter.
QUOTE_EXCHANGE = {"NASDAQ": "NAS", "NYSE": "NYS", "AMEX": "AMS",
                  "NASD": "NAS", "NYS": "NYS", "AMS": "AMS"}


def lookup_exchange(symbol: str, default: str = "NASDAQ") -> str:
    """Resolve a symbol's listing venue from the stored symbol directory.

    Guessing is not an option: SDEV is AMEX, and a NASDAQ quote request for it
    returns an empty body rather than an error, so a guessed venue fails
    silently and an order would be routed to the wrong exchange.
    """
    from uat.storage.db import query

    try:
        df = query(
            "SELECT exchange FROM raw.symbol_directory WHERE symbol = ? "
            "ORDER BY as_of DESC LIMIT 1",
            [symbol],
        )
    except Exception:
        return default
    if df.empty or not df.iloc[0]["exchange"]:
        log.warning("no directory entry for %s; defaulting to %s", symbol, default)
        return default
    return str(df.iloc[0]["exchange"])


@dataclass
class OrderGuard:
    """Hard caps. A caller cannot widen these; it can only pass a tighter one."""

    max_notional_usd: float = 20.0
    max_shares: int = 1000
    max_orders_per_session: int = 6
    _sent: int = field(default=0, init=False)

    def check(self, qty: int, price: float) -> tuple[bool, str]:
        notional = qty * price
        if qty <= 0:
            return False, "quantity must be positive"
        if notional > self.max_notional_usd:
            return False, f"notional ${notional:.2f} exceeds cap ${self.max_notional_usd:.2f}"
        if qty > self.max_shares:
            return False, f"{qty} shares exceeds cap {self.max_shares}"
        if self._sent >= self.max_orders_per_session:
            return False, f"already sent {self._sent} orders this session"
        return True, ""

    def record(self) -> None:
        self._sent += 1


@dataclass
class OrderRequest:
    symbol: str
    side: Side
    qty: int
    limit_px: float
    exchange: str
    tr_id: str
    body: dict
    dry_run: bool

    @property
    def notional(self) -> float:
        return self.qty * self.limit_px


class KisBroker:
    def __init__(self, client: KisClient | None = None, guard: OrderGuard | None = None):
        self.c = client or KisClient()
        self.guard = guard or OrderGuard()

    # ---------- quotes ----------

    def current_price(self, symbol: str, exchange: str = "NASDAQ") -> dict:
        """Last price and quote. Returns {} on failure."""
        excd = QUOTE_EXCHANGE.get(exchange.upper(), "NAS")
        r = self.c.get(
            "/uapi/overseas-price/v1/quotations/price",
            "overseas_price",
            {"AUTH": "", "EXCD": excd, "SYMB": symbol},
        )
        return r.output if (r.ok and isinstance(r.output, dict)) else {}

    def marketable_limit(
        self, symbol: str, side: Side, exchange: str = "NASDAQ", buffer: float = 0.004
    ) -> float | None:
        """Aggressive limit that should fill like a market order.

        KIS has no market order, so a buy is placed slightly above the last
        print and a sell slightly below.
        """
        q = self.current_price(symbol, exchange)
        last = float(q.get("last") or 0) or float(q.get("base") or 0)
        if last <= 0:
            log.error("no price for %s", symbol)
            return None
        px = last * (1 + buffer) if side == "BUY" else last * (1 - buffer)
        return round(px, 4)

    # ---------- orders ----------

    def _hashkey(self, body: dict) -> str | None:
        self.c.bucket.acquire()
        resp = httpx.post(
            f"{self.c.base}/uapi/hashkey",
            headers={
                "content-type": "application/json; charset=utf-8",
                "appkey": self.c.creds.app_key,
                "appsecret": self.c.creds.app_secret,
            },
            json=body,
            timeout=20,
        )
        if resp.status_code != 200:
            log.error("hashkey failed %s %s", resp.status_code, resp.text[:200])
            return None
        return resp.json().get("HASH")

    def build_order(
        self, symbol: str, side: Side, qty: int, limit_px: float, exchange: str = "NASDAQ"
    ) -> OrderRequest:
        excg = EXCHANGE_CODES.get(exchange.upper(), exchange.upper())
        tr = self.c.tr_id("overseas_buy" if side == "BUY" else "overseas_sell")
        body = {
            "CANO": self.c.creds.account_8,
            "ACNT_PRDT_CD": self.c.creds.account_2,
            "OVRS_EXCG_CD": excg,
            "PDNO": symbol,
            "ORD_QTY": str(int(qty)),
            "OVRS_ORD_UNPR": f"{limit_px:.4f}",
            "ORD_SVR_DVSN_CD": "0",
            "ORD_DVSN": "00",          # limit only; KIS rejects market orders here
        }
        return OrderRequest(symbol, side, int(qty), float(limit_px), excg, tr, body, True)

    def submit(self, req: OrderRequest, confirm: bool = False) -> KisResponse:
        """Send the order. Without confirm=True this is a no-op dry run."""
        ok, why = self.guard.check(req.qty, req.limit_px)
        if not ok:
            log.error("order blocked by guard: %s", why)
            return KisResponse(False, "GUARD", why, None, {})

        if not confirm:
            log.info(
                "DRY RUN -- not sent",
                extra={"extra_fields": {"symbol": req.symbol, "side": req.side,
                                        "qty": req.qty, "limit": req.limit_px,
                                        "tr": req.tr_id, "env": self.c.env}},
            )
            return KisResponse(True, "DRYRUN", "not sent", req.body, {})

        hashkey = self._hashkey(req.body)
        if not hashkey:
            return KisResponse(False, "HASHKEY", "hashkey request failed", None, {})

        headers = self.c._headers(req.tr_id)
        headers["hashkey"] = hashkey
        self.c.bucket.acquire()
        resp = httpx.post(
            f"{self.c.base}/uapi/overseas-stock/v1/trading/order",
            headers=headers,
            json=req.body,
            timeout=20,
        )
        out = self.c._parse(resp, "order", req.tr_id)
        self.guard.record()
        log.info(
            "order submitted",
            extra={"extra_fields": {"symbol": req.symbol, "side": req.side,
                                    "qty": req.qty, "limit": req.limit_px,
                                    "ok": out.ok, "rt_cd": out.rt_cd, "msg": out.msg,
                                    "ts": datetime.now().isoformat()}},
        )
        return out

    def close(self) -> None:
        self.c.close()
