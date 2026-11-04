"""Query the KIS account: cash, buying power, and current US holdings.

    KIS_ENV=paper python scripts/check_balance.py
    KIS_ENV=live  python scripts/check_balance.py

Read-only. It places no orders and needs no write permission on the account.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# The Windows console defaults to cp949 here, which cannot encode Korean output
# or symbols like the warning sign and kills the script mid-report.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uat.broker.kis_auth import MissingCredentials  # noqa: E402
from uat.broker.kis_client import KisClient, kis_num as _num  # noqa: E402
from uat.config import load_settings  # noqa: E402
from uat.live import holdings as hold  # noqa: E402

KRW_PER_USD = 1380.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=["paper", "live"], default=None)
    ap.add_argument("--probe-symbol", default="AAPL")
    ap.add_argument("--probe-price", type=float, default=1.00)
    args = ap.parse_args()

    if args.env:
        os.environ["KIS_ENV"] = args.env
    settings = load_settings(args.env)

    try:
        client = KisClient(args.env)
    except MissingCredentials as exc:
        print(f"\n[credentials missing] {exc}\n")
        print("Create .env in the project root with:")
        print("  KIS_ENV=paper")
        print("  KIS_APP_KEY=...")
        print("  KIS_APP_SECRET=...")
        print("  KIS_ACCOUNT_8=<8-digit account number>")
        print("  KIS_ACCOUNT_2=01")
        print("\nIssue the key at https://apiportal.koreainvestment.com "
              "(계좌 개설 후 KIS Developers 신청).")
        sys.exit(2)

    print(f"env           : {settings.env}  ({settings.get('broker.base_url')})")
    print(f"account       : {client.creds.account_8}-{client.creds.account_2}")
    print(f"rate limit    : {settings.get('broker.rate_limit_per_sec')} req/s")
    print()

    bp = client.overseas_buying_power(args.probe_symbol, "NASD", args.probe_price)
    fx = KRW_PER_USD
    if not bp.ok:
        print(f"[buying power] FAILED rt_cd={bp.rt_cd} {bp.msg}")
    else:
        o = bp.output if isinstance(bp.output, dict) else {}
        # ord_psbl_frcr_amt is USD (tr_crcy_cd confirms). exrt is KIS's own
        # rate, which beats hardcoding one.
        usd = _num(o, "ord_psbl_frcr_amt", "ovrs_ord_psbl_amt")
        fx = _num(o, "exrt") or KRW_PER_USD
        max_qty = _num(o, "max_ord_psbl_qty", "ovrs_max_ord_psbl_qty")
        print("=== 매수가능금액 (buying power) ===")
        print(f"  주문가능        : $ {usd:,.2f}   (~KRW {usd*fx:,.0f})")
        print(f"  적용환율        : {fx:,.2f} KRW/USD")
        if max_qty:
            print(f"  {args.probe_symbol} @ ${args.probe_price:.2f} 최대 {max_qty:,.0f}주")
        if not usd:
            print(f"  (no recognised amount field; raw output: {o})")

    print()
    bal = client.overseas_balance()
    if not bal.ok:
        print(f"[balance] FAILED rt_cd={bal.rt_cd} {bal.msg}")
    else:
        rows = hold.parse_holdings(bal.raw)
        summary = hold.parse_summary(bal.raw)
        print("=== 해외주식 잔고 ===")
        if not rows:
            print("  (보유 종목 없음)")
        for h in rows:
            print(f"  {h.symbol:<8} {h.qty:>8,d}주  평단 ${h.avg_px:>8.4f}  "
                  f"현재 ${h.last_px:>8.4f}  "
                  f"평가손익 ${h.pnl_usd:>9,.2f} ({h.pnl_pct:+.2f}%)")
        # output2 is USD, not KRW: `frcr` is 외화. And tot_evlu_pfls_amt is the
        # market VALUE despite the name -- on 7 QTEX at $1.12 it returns 7.84.
        # This block previously divided the USD cost basis by the FX rate and
        # printed market value as P&L, turning $8.47 of cost into one cent.
        # See live/holdings.py for the field-by-field reading.
        if summary.cost_basis or summary.market_value:
            print(f"  ---- 매입원금 $ {summary.cost_basis:,.2f}  "
                  f"평가금액 $ {summary.market_value:,.2f}")
            print(f"       평가손익 $ {summary.unrealised_pnl:+,.2f}  "
                  f"실현손익 $ {summary.realised_pnl:+,.2f}  "
                  f"수익률 {summary.return_pct:+.2f}%")

        if rows:
            print()
            if settings.env == "paper":
                print("  ⚠ 이 모의계좌는 pivot-engine 과 공유됩니다. 위 보유 종목은")
                print("    그쪽 전략이 낸 포지션일 수 있고, 여기서 주문을 내면 간섭합니다.")
            else:
                print("  보유 종목의 스톱/트레일 상태는 이 스크립트가 아니라")
                print("  scripts/positions.py 가 보여줍니다. 거기에 기록이 없는 종목은")
                print("  스톱이 어디에도 존재하지 않는 상태입니다.")

    print()
    print("=== what this account can run ===")
    o = bp.output if (bp.ok and isinstance(bp.output, dict)) else {}
    usd = _num(o, "ord_psbl_frcr_amt", "ovrs_ord_psbl_amt")
    if usd <= 0:
        print("  buying power unavailable -- cannot advise on sizing")
    else:
        small_threshold = float(settings.get("risk.small_account_usd", 1450))
        if usd < small_threshold:
            pos = usd
            mode = "single-position plumbing mode (1% rule is not expressible here)"
        else:
            pos = usd * float(settings.get("risk.risk_per_trade", 0.01)) / 0.22
            mode = "1% risk rule, ~5 concurrent positions"
        print(f"  mode          : {mode}")
        print(f"  position size : ~$ {pos:,.2f}")
        print(f"  threshold     : $ {small_threshold:,.0f} "
              f"(~KRW {small_threshold*fx:,.0f}) switches to the 1% rule")

    client.close()


if __name__ == "__main__":
    main()
