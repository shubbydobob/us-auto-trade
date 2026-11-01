"""Rate-limited KIS REST client for US equities.

Two constraints shape this file:

* Overseas endpoints are limited to roughly **1 request per second** (domestic
  gets 20). The token bucket defaults to 0.8/s for headroom, and every call
  goes through it.
* Paper and live differ in BOTH host and `tr_id` prefix (V... vs T...). Every
  difference lives in TR_IDS so flipping KIS_ENV switches the whole system
  atomically rather than leaving one endpoint pointed at the wrong environment.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import httpx

from uat.broker.kis_auth import KisCredentials, get_access_token, load_credentials
from uat.config import load_settings
from uat.logging import get

log = get("broker.kis")

# (live, paper) transaction ids.
TR_IDS: dict[str, tuple[str, str]] = {
    "overseas_balance": ("TTTS3012R", "VTTS3012R"),
    "overseas_buying_power": ("TTTS3007R", "VTTS3007R"),
    "overseas_buy": ("TTTT1002U", "VTTT1002U"),
    "overseas_sell": ("TTTT1006U", "VTTT1006U"),
    "overseas_price": ("HHDFS00000300", "HHDFS00000300"),
}

# KIS exchange codes. The balance endpoint wants NASD as an umbrella for all
# US exchanges; order endpoints want the specific venue.
EXCHANGE_CODES = {"NASDAQ": "NASD", "NYSE": "NYSE", "AMEX": "AMEX"}


class TokenBucket:
    """Blocking token bucket. Overseas KIS allows ~1 req/s; we use 0.8."""

    def __init__(self, rate_per_sec: float):
        self.interval = 1.0 / max(rate_per_sec, 0.01)
        self._lock = threading.Lock()
        self._next_at = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next_at - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self._next_at = now + self.interval


@dataclass
class KisResponse:
    ok: bool
    rt_cd: str
    msg: str
    output: Any
    raw: dict


class KisClient:
    def __init__(self, env: str | None = None):
        self.settings = load_settings(env)
        self.env = self.settings.env
        self.creds: KisCredentials = load_credentials(env)
        self.base = self.settings.get("broker.base_url")
        self.bucket = TokenBucket(float(self.settings.get("broker.rate_limit_per_sec", 0.8)))
        self._token: str | None = None
        self._client = httpx.Client(timeout=20)

    # ---------- plumbing ----------

    def tr_id(self, name: str) -> str:
        live, paper = TR_IDS[name]
        return paper if self.creds.is_paper else live

    def _headers(self, tr: str) -> dict[str, str]:
        if self._token is None:
            self._token = get_access_token(self.env)
        return {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {self._token}",
            "appkey": self.creds.app_key,
            "appsecret": self.creds.app_secret,
            "tr_id": tr,
            "custtype": "P",
        }

    def get(self, path: str, tr_name: str, params: dict) -> KisResponse:
        self.bucket.acquire()
        tr = self.tr_id(tr_name)
        resp = self._client.get(f"{self.base}{path}", headers=self._headers(tr), params=params)
        return self._parse(resp, tr_name, tr)

    def _parse(self, resp: httpx.Response, tr_name: str, tr: str) -> KisResponse:
        if resp.status_code != 200:
            log.error(
                "KIS HTTP error",
                extra={"extra_fields": {"tr": tr, "status": resp.status_code,
                                        "body": resp.text[:300]}},
            )
            return KisResponse(False, str(resp.status_code), resp.text[:300], None, {})
        data = resp.json()
        rt_cd = str(data.get("rt_cd", ""))
        msg = data.get("msg1", "").strip()
        ok = rt_cd == "0"
        if not ok:
            log.error("KIS API error",
                      extra={"extra_fields": {"tr": tr, "rt_cd": rt_cd, "msg": msg}})
        output = data.get("output") or data.get("output1") or data.get("output2")
        return KisResponse(ok, rt_cd, msg, output, data)

    # ---------- account queries ----------

    def overseas_balance(self, exchange: str = "NASD", currency: str = "USD") -> KisResponse:
        """Holdings and account summary for US equities."""
        return self.get(
            "/uapi/overseas-stock/v1/trading/inquire-balance",
            "overseas_balance",
            {
                "CANO": self.creds.account_8,
                "ACNT_PRDT_CD": self.creds.account_2,
                "OVRS_EXCG_CD": exchange,
                "TR_CRCY_CD": currency,
                "CTX_AREA_FK200": "",
                "CTX_AREA_NK200": "",
            },
        )

    def overseas_buying_power(
        self, symbol: str = "AAPL", exchange: str = "NASD", price: float = 1.0
    ) -> KisResponse:
        """Purchasable amount.

        KIS requires a symbol and a price because it answers "how many shares of
        THIS can I buy". The cash figures in the response are what we actually
        want, so any liquid symbol works as a probe.
        """
        return self.get(
            "/uapi/overseas-stock/v1/trading/inquire-psamount",
            "overseas_buying_power",
            {
                "CANO": self.creds.account_8,
                "ACNT_PRDT_CD": self.creds.account_2,
                "OVRS_EXCG_CD": exchange,
                "OVRS_ORD_UNPR": f"{price:.4f}",
                "ITEM_CD": symbol,
            },
        )

    def close(self) -> None:
        self._client.close()
