"""One-shot entry: refresh data, scan, price, confirm, buy, record.

    python scripts/buy_today.py --env live                    # dry run
    python scripts/buy_today.py --env live --symbol QTEX      # pick explicitly
    python scripts/buy_today.py --env live --symbol QTEX --confirm

Signals are computed from the previous close, so the data refresh matters: a
Friday signal acted on Monday has had a weekend of filings behind it. Pass
--skip-refresh only when the bars are already current.

On a fill the position is written to pos.positions with its stop level, so the
exit loop can pick it up. Without that record the stop exists nowhere.
"""
from __future__ import annotations

import argparse
import os
import sys
import uuid
import warnings
from datetime import date, datetime
from pathlib import Path

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uat.broker.kis_auth import MissingCredentials  # noqa: E402
from uat.broker.kis_broker import KisBroker, OrderGuard, lookup_exchange  # noqa: E402
from uat.broker.kis_client import EXCHANGE_CODES, KisClient  # noqa: E402
from uat.clock import ET, is_session, session_open_close  # noqa: E402
from uat.live.positions import record_order, save  # noqa: E402
from uat.live.scanner import affordable, scan  # noqa: E402
from uat.strategy.sizing import size_position  # noqa: E402
from uat.strategy.types import Position  # noqa: E402


def market_open_now() -> tuple[bool, str]:
    now = datetime.now(ET)
    d = now.date()
    if not is_session(d):
        return False, f"{d} is not a trading session"
    o, c = session_open_close(d)
    if now < o:
        return False, f"pre-open; regular session starts {o:%H:%M} ET"
    if now > c:
        return False, f"after close; session ended {c:%H:%M} ET"
    return True, f"open until {c:%H:%M} ET"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=["paper", "live"], default="live")
    ap.add_argument("--symbol", default=None, help="override the scanner's pick")
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--max-notional", type=float, default=20.0)
    ap.add_argument("--skip-refresh", action="store_true")
    ap.add_argument("--allow-closed", action="store_true",
                    help="build the order even though the market is shut")
    ap.add_argument("--confirm", action="store_true")
    args = ap.parse_args()

    os.environ["KIS_ENV"] = args.env

    if not args.skip_refresh:
        from uat.data.ingest_daily import ingest_daily
        from uat.universe.build import build_tradeable_symbols

        print("refreshing bars (last 10 sessions) ...")
        symbols, _ = build_tradeable_symbols(refresh=False)
        rep = ingest_daily(symbols, date.today().replace(day=1), date.today())
        print(f"  {rep['rows']:,} rows, success {rep['success_rate']:.1%}")
        if rep["below_quality_threshold"]:
            print("  [abort] ingest below the data-quality threshold -- not trading")
            sys.exit(1)
        print()

    try:
        client = KisClient(args.env)
    except MissingCredentials as exc:
        print(f"[credentials] {exc}")
        sys.exit(2)
    broker = KisBroker(client, OrderGuard(max_notional_usd=args.max_notional))

    is_open, state = market_open_now()
    print(f"market : {state}")
    if not is_open and args.confirm and not args.allow_closed:
        print("[abort] market is closed; pass --allow-closed to queue anyway")
        broker.close()
        sys.exit(1)

    bp = client.overseas_buying_power("AAPL", "NASD", 1.0)
    cash = float((bp.output or {}).get("ord_psbl_frcr_amt", 0) or 0)
    print(f"cash   : ${cash:,.2f}")
    print()

    cands, _ = scan(cash_usd=cash, k=args.top_k)
    pool = affordable(cands)
    if not pool:
        print("[abort] nothing in the top-%d is affordable" % args.top_k)
        broker.close()
        sys.exit(1)

    if args.symbol:
        match = [c for c in pool if c.symbol == args.symbol.upper()]
        if not match:
            print(f"[abort] {args.symbol} is not in today's affordable candidates")
            print("        affordable: " + ", ".join(c.symbol for c in pool))
            broker.close()
            sys.exit(1)
        pick = match[0]
    else:
        pick = pool[0]

    exchange = lookup_exchange(pick.symbol)
    limit = broker.marketable_limit(pick.symbol, "BUY", exchange)
    if limit is None:
        print(f"[abort] no quote for {pick.symbol}")
        broker.close()
        sys.exit(1)

    excg = EXCHANGE_CODES.get(exchange.upper(), "NASD")
    bp2 = client.overseas_buying_power(pick.symbol, excg, limit)
    venue_cash = float((bp2.output or {}).get("ord_psbl_frcr_amt", cash) or cash)

    sizing = size_position(
        equity=venue_cash,
        entry_px=limit,
        atr14=pick.atr14,
        median_volume_20=pick.medvol_20,
        addv_20=pick.addv_20,
    )
    qty = min(sizing.shares, int(venue_cash // limit))
    if qty < 1:
        print(f"[abort] sizing yields {sizing.shares} shares ({sizing.reject_reason})")
        broker.close()
        sys.exit(1)

    print(f"pick   : rank {pick.rank} {pick.symbol} ({exchange})  score {pick.score:.1f}")
    print(f"limit  : ${limit:.4f}   qty {qty}   notional ${qty*limit:.2f}")
    print(f"stop   : ${sizing.stop_px:.4f}  ({(1-sizing.stop_px/limit)*100:.0f}% away)"
          f"   binding cap: {sizing.binding_cap}")
    print(f"cost   : ~{pick.cost_drag_pct*100:.1f}% round trip "
          f"(${qty*limit*pick.cost_drag_pct:.2f})")
    print()

    req = broker.build_order(pick.symbol, "BUY", qty, limit, exchange)
    if not args.confirm:
        print("DRY RUN -- nothing sent. Add --confirm to place it.")
        print(f"  body: {req.body}")
        broker.close()
        return

    print(">>> SENDING LIVE BUY <<<")
    res = broker.submit(req, confirm=True)
    order_id = f"{pick.symbol}-{uuid.uuid4().hex[:8]}"
    kis_no = str((res.output or {}).get("ODNO", "")) if isinstance(res.output, dict) else ""
    record_order(order_id, pick.symbol, "BUY", qty, limit,
                 "NEW" if res.ok else "REJECTED", kis_no, "entry")
    print(f"  ok={res.ok}  rt_cd={res.rt_cd}  msg={res.msg}  kis_order_no={kis_no}")

    if not res.ok:
        print("  order rejected -- no position recorded")
        broker.close()
        sys.exit(1)

    # Record the intended position immediately. The fill may be partial, and
    # the next reconciliation corrects the quantity from the broker; what must
    # not be lost is the stop level, which exists nowhere else.
    pos = Position(
        symbol=pick.symbol,
        opened_on=date.today(),
        entry_px=limit,
        qty_initial=qty,
        qty_open=qty,
        atr_at_entry=pick.atr14,
        stop_px=sizing.stop_px,
        hh_since_entry=limit,
        hc_since_entry=limit,
    )
    pid = save(pos, exchange=exchange)
    print(f"  position recorded: {pid}")
    print("  NOTE: exit is manual until the exit loop ships (docs/exit_loop.md)")
    broker.close()


if __name__ == "__main__":
    main()
