"""Place one plumbing-test order.

    python scripts/place_order.py --symbol SDEV --cash 8.91            # dry run
    python scripts/place_order.py --symbol SDEV --cash 8.91 --confirm  # sends it

Dry run is the default and prints the exact request body. `--confirm` is the
only thing that actually sends, and OrderGuard still caps notional regardless.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uat.broker.kis_auth import MissingCredentials  # noqa: E402
from uat.broker.kis_broker import KisBroker, OrderGuard, lookup_exchange  # noqa: E402
from uat.broker.kis_client import EXCHANGE_CODES  # noqa: E402
from uat.broker.kis_client import KisClient  # noqa: E402
from uat.clock import ET, current_session_date, is_session, session_open_close  # noqa: E402


def market_state() -> str:
    now = datetime.now(ET)
    d = now.date()
    if not is_session(d):
        return f"CLOSED ({d} is not a trading session; next order would queue)"
    open_t, close_t = session_open_close(d)
    if now < open_t:
        return f"PRE-OPEN (regular session opens {open_t:%H:%M} ET)"
    if now > close_t:
        return f"AFTER CLOSE (session ended {close_t:%H:%M} ET)"
    return f"OPEN (until {close_t:%H:%M} ET)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--exchange", default=None,
                    help="default: resolved from the symbol directory")
    ap.add_argument("--side", choices=["BUY", "SELL"], default="BUY")
    ap.add_argument("--env", choices=["paper", "live"], default="live")
    ap.add_argument("--cash", type=float, default=None, help="budget; default = buying power")
    ap.add_argument("--qty", type=int, default=None, help="explicit share count")
    ap.add_argument("--limit", type=float, default=None, help="explicit limit price")
    ap.add_argument("--max-notional", type=float, default=20.0)
    ap.add_argument("--confirm", action="store_true", help="actually send the order")
    args = ap.parse_args()

    os.environ["KIS_ENV"] = args.env

    try:
        client = KisClient(args.env)
    except MissingCredentials as exc:
        print(f"[credentials] {exc}")
        sys.exit(2)

    broker = KisBroker(client, OrderGuard(max_notional_usd=args.max_notional))
    exchange = args.exchange or lookup_exchange(args.symbol)

    print(f"env       : {args.env}  account {client.creds.account_8}-{client.creds.account_2}")
    print(f"market    : {market_state()}")
    print(f"session   : {current_session_date()}")
    print()

    quote = broker.current_price(args.symbol, exchange)
    last = float(quote.get("last") or 0) or float(quote.get("base") or 0)
    if last <= 0:
        print(f"[quote] no price for {args.symbol} -- cannot size or price the order")
        print(f"        raw: {quote}")
        broker.close()
        sys.exit(1)

    limit = args.limit or broker.marketable_limit(args.symbol, args.side, exchange)
    if limit is None:
        broker.close()
        sys.exit(1)

    if args.qty:
        qty = args.qty
    else:
        cash = args.cash
        if cash is None:
            # Buying power is venue-specific too: asking NASD about an AMEX
            # listing returns a quantity of zero rather than an error.
            excg = EXCHANGE_CODES.get(exchange.upper(), "NASD")
            bp = client.overseas_buying_power(args.symbol, excg, limit)
            cash = float((bp.output or {}).get("ord_psbl_frcr_amt", 0) or 0)
        qty = int(cash // limit)

    print(f"symbol    : {args.symbol} ({exchange})")
    print(f"last      : ${last:.4f}")
    print(f"limit     : ${limit:.4f}   ({(limit/last-1)*100:+.2f}% vs last)")
    print(f"quantity  : {qty}")
    print(f"notional  : ${qty*limit:.2f}")
    print()

    if qty < 1:
        print("[abort] budget buys zero shares")
        broker.close()
        sys.exit(1)

    req = broker.build_order(args.symbol, args.side, qty, limit, exchange)
    print(f"tr_id     : {req.tr_id}   (ORD_DVSN=00 limit; KIS has no market order)")
    print("request body:")
    print(json.dumps(req.body, indent=2))
    print()

    if not args.confirm:
        print("DRY RUN -- nothing was sent. Re-run with --confirm to place it.")
        broker.close()
        return

    print(">>> SENDING LIVE ORDER <<<")
    res = broker.submit(req, confirm=True)
    print(f"  ok={res.ok}  rt_cd={res.rt_cd}  msg={res.msg}")
    if res.output:
        print(f"  output: {res.output}")
    broker.close()


if __name__ == "__main__":
    main()
