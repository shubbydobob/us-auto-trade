"""Show what the strategy would buy, and what this account can actually afford.

    python scripts/scan_today.py --env live

Read-only: it queries buying power and prints a plan. It places no orders.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uat.broker.kis_auth import MissingCredentials  # noqa: E402
from uat.broker.kis_client import KisClient  # noqa: E402
from uat.live.scanner import affordable, scan  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=["paper", "live"], default="live")
    ap.add_argument("--scorer", default="evidence")
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--cash", type=float, default=None,
                    help="override buying power instead of querying KIS")
    args = ap.parse_args()

    os.environ["KIS_ENV"] = args.env

    cash = args.cash
    if cash is None:
        try:
            client = KisClient(args.env)
        except MissingCredentials as exc:
            print(f"[credentials] {exc}")
            sys.exit(2)
        bp = client.overseas_buying_power("AAPL", "NASD", 1.0)
        client.close()
        if not bp.ok:
            print(f"[buying power] FAILED {bp.rt_cd} {bp.msg}")
            sys.exit(1)
        cash = float((bp.output or {}).get("ord_psbl_frcr_amt", 0) or 0)

    print(f"env {args.env}   buying power $ {cash:,.2f}   scorer {args.scorer}")
    print()

    cands, _ = scan(cash_usd=cash, scorer=args.scorer, k=args.top_k)
    if not cands:
        print("no candidates for the latest session")
        return

    print(f"{'#':>3} {'symbol':<8}{'score':>7}{'close':>9}{'ADDV':>10}"
          f"{'shares':>8}{'notional':>10}{'spread':>8}{'cost':>7}  verdict")
    print("-" * 82)
    for c in cands:
        verdict = "buyable" if c.affordable_shares >= 1 else "too expensive"
        if c.affordable_shares >= 1 and c.cost_drag_pct > 0.06:
            verdict = "buyable (cost heavy)"
        print(
            f"{c.rank:>3} {c.symbol:<8}{c.score:>7.1f}{c.close:>9.3f}"
            f"{c.addv_20/1e6:>9.1f}M{c.affordable_shares:>8}"
            f"{c.notional:>10.2f}{c.spread_est*100:>7.1f}%"
            f"{c.cost_drag_pct*100:>6.1f}%  {verdict}"
        )

    buyable = affordable(cands)
    print()
    if not buyable:
        print("Nothing in the top-%d is affordable at $%.2f." % (args.top_k, cash))
        print("The ranking is not the constraint here -- the cash is.")
        return

    pick = buyable[0]
    print(f"=== plumbing-test pick: rank {pick.rank} {pick.symbol} ===")
    print(f"  buy            {pick.affordable_shares} shares @ ~${pick.close:.4f}"
          f"  = ${pick.notional:.2f}")
    stop_dist = max(2.5 * pick.atr14, 0.20 * pick.close)
    print(f"  stop           ${max(0.01, pick.close - stop_dist):.4f}"
          f"  ({stop_dist/pick.close*100:.0f}% away)")
    print(f"  round-trip cost~{pick.cost_drag_pct*100:.1f}% of the position"
          f"  (${pick.notional*pick.cost_drag_pct:.2f})")
    print(f"  breakeven move  +{pick.cost_drag_pct*100:.1f}%")
    print()
    print("  This exists to measure fills and slippage, not returns. A +100%")
    print(f"  winner on ${pick.notional:.2f} is ${pick.notional:.2f}.")


if __name__ == "__main__":
    main()
