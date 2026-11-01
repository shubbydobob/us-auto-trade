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
from uat.broker.kis_client import KisClient  # noqa: E402
from uat.config import load_settings  # noqa: E402

KRW_PER_USD = 1380.0


def _num(d: dict, *keys) -> float:
    for k in keys:
        if k in d and str(d[k]).strip() not in ("", "-"):
            try:
                return float(str(d[k]).replace(",", ""))
            except ValueError:
                continue
    return 0.0


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
        rows = bal.raw.get("output1") or []
        summary = bal.raw.get("output2") or {}
        if isinstance(summary, list):
            summary = summary[0] if summary else {}
        print("=== 해외주식 잔고 ===")
        if not rows:
            print("  (보유 종목 없음)")
        for r in rows:
            sym = r.get("ovrs_pdno", "?")
            qty = _num(r, "ovrs_cblc_qty")
            avg = _num(r, "pchs_avg_pric")
            now = _num(r, "now_pric2")
            pnl = _num(r, "frcr_evlu_pfls_amt")
            rate = _num(r, "evlu_pfls_rt")
            print(f"  {sym:<8} {qty:>8,.0f}주  평단 ${avg:>8.4f}  현재 ${now:>8.4f}  "
                  f"평가손익 ${pnl:>9,.2f} ({rate:+.2f}%)")
        if summary:
            # output2 is denominated in KRW, not USD -- frcr_pchs_amt1 of
            # 9,701,405 against ~$7.3k of holdings only makes sense as won.
            pnl_krw = _num(summary, "tot_evlu_pfls_amt", "ovrs_tot_pfls")
            cost_krw = _num(summary, "frcr_pchs_amt1", "frcr_buy_amt_smtl1")
            rate = _num(summary, "tot_pftrt")
            print(f"  ---- 매입원금 KRW {cost_krw:,.0f} (~$ {cost_krw/fx:,.2f})")
            print(f"       평가손익 KRW {pnl_krw:,.0f} (~$ {pnl_krw/fx:,.2f})  "
                  f"수익률 {rate:+.2f}%")

        if rows:
            print()
            print("  ⚠ 이 계좌는 pivot-engine 의 모의 운용 계좌입니다. 위 보유 종목은")
            print("    그쪽 전략이 낸 포지션이고, 여기서 주문을 내면 서로 간섭합니다.")

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
