"""Broker-side holdings: what the account actually owns, right now.

This is the authoritative side of the two-sources-of-truth split described in
`live.positions`: KIS owns quantity and average price, we own the exit plan.
Nothing here invents or infers anything -- it only parses.

Field semantics were read off a live response rather than the documentation,
because two of them are actively misleading:

* `output2.tot_evlu_pfls_amt` is **not** profit and loss despite the name. On a
  7-share QTEX position at $1.12 it returned 7.84, which is the market value.
  Realised/unrealised P&L is `ovrs_tot_pfls` (-0.63) and the percentage is
  `tot_pftrt`. An earlier version of the balance report printed market value
  labelled as P&L.
* `frcr_pchs_amt1` is USD, not KRW -- `frcr` is 외화, foreign currency. The same
  earlier version divided it by the FX rate and reported an $8.47 cost basis as
  one cent.

`ord_psbl_qty` is kept alongside `ovrs_cblc_qty` deliberately. Held minus
sellable is how many shares the broker has reserved against an open sell order,
which is the direct measurement for open question 3 in docs/exit_loop.md (does
a resting protective sell lock up the shares?) and the only way the exit loop
can tell its own resting order is still live.
"""
from __future__ import annotations

from dataclasses import dataclass

from uat.broker.kis_client import KisClient, kis_num
from uat.logging import get

log = get("live.holdings")

# Venue code as it comes back from the balance endpoint -> our directory name.
VENUE_NAMES = {"NASD": "NASDAQ", "NAS": "NASDAQ", "NYSE": "NYSE",
               "NYS": "NYSE", "AMEX": "AMEX", "AMS": "AMEX"}


@dataclass(frozen=True)
class Holding:
    symbol: str
    name: str
    qty: int
    qty_sellable: int
    avg_px: float
    last_px: float
    pnl_usd: float
    pnl_pct: float
    venue: str
    currency: str

    @property
    def cost_basis(self) -> float:
        return self.qty * self.avg_px

    @property
    def market_value(self) -> float:
        return self.qty * self.last_px

    @property
    def qty_reserved(self) -> int:
        """Shares the broker will not let us sell -- i.e. already committed to
        an open sell order. Non-zero means a resting exit is live."""
        return max(0, self.qty - self.qty_sellable)


@dataclass(frozen=True)
class AccountSummary:
    cost_basis: float          # frcr_pchs_amt1, USD
    market_value: float        # tot_evlu_pfls_amt -- value, not P&L; see module docstring
    unrealised_pnl: float      # ovrs_tot_pfls
    realised_pnl: float        # ovrs_rlzt_pfls_amt
    return_pct: float          # tot_pftrt


def parse_holdings(raw: dict) -> list[Holding]:
    """Holdings from an inquire-balance payload. Zero-quantity rows are dropped."""
    out: list[Holding] = []
    for r in raw.get("output1") or []:
        qty = int(kis_num(r, "ovrs_cblc_qty"))
        if qty <= 0:
            continue
        venue_raw = str(r.get("ovrs_excg_cd", "")).strip()
        out.append(
            Holding(
                symbol=str(r.get("ovrs_pdno", "")).strip(),
                name=str(r.get("ovrs_item_name", "")).strip(),
                qty=qty,
                qty_sellable=int(kis_num(r, "ord_psbl_qty", default=float(qty))),
                avg_px=kis_num(r, "pchs_avg_pric"),
                last_px=kis_num(r, "now_pric2"),
                pnl_usd=kis_num(r, "frcr_evlu_pfls_amt"),
                pnl_pct=kis_num(r, "evlu_pfls_rt"),
                venue=VENUE_NAMES.get(venue_raw.upper(), venue_raw),
                currency=str(r.get("tr_crcy_cd", "USD")).strip() or "USD",
            )
        )
    return out


def parse_summary(raw: dict) -> AccountSummary:
    o = raw.get("output2") or {}
    if isinstance(o, list):
        o = o[0] if o else {}
    return AccountSummary(
        cost_basis=kis_num(o, "frcr_pchs_amt1", "frcr_buy_amt_smtl1"),
        market_value=kis_num(o, "tot_evlu_pfls_amt"),
        unrealised_pnl=kis_num(o, "ovrs_tot_pfls"),
        realised_pnl=kis_num(o, "ovrs_rlzt_pfls_amt"),
        return_pct=kis_num(o, "tot_pftrt"),
    )


def fetch(client: KisClient, venues: tuple[str, ...] = ("NASD",)) -> tuple[list[Holding], AccountSummary]:
    """Query the broker for US holdings.

    NASD acts as an umbrella for all US venues on this endpoint, so one request
    is the default -- each extra venue costs a second of the ~1 req/s overseas
    budget. Pass more venues if a holding ever fails to appear; results are
    deduplicated by symbol.
    """
    seen: dict[str, Holding] = {}
    summary = AccountSummary(0.0, 0.0, 0.0, 0.0, 0.0)
    for i, venue in enumerate(venues):
        r = client.overseas_balance(venue)
        if not r.ok:
            log.error("balance query failed",
                      extra={"extra_fields": {"venue": venue, "rt_cd": r.rt_cd, "msg": r.msg}})
            continue
        for h in parse_holdings(r.raw):
            seen.setdefault(h.symbol, h)
        if i == 0:
            summary = parse_summary(r.raw)
    log.info("holdings fetched",
             extra={"extra_fields": {"n": len(seen), "symbols": sorted(seen)}})
    return [seen[k] for k in sorted(seen)], summary


def as_reconcile_input(holdings: list[Holding]) -> dict[str, tuple[int, float]]:
    """Shape `live.positions.reconcile` expects: symbol -> (qty, avg_px)."""
    return {h.symbol: (h.qty, h.avg_px) for h in holdings}
