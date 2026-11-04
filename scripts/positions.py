"""What we hold, what the exit plan says, and what the exit loop would do now.

    python scripts/positions.py --env live
    python scripts/positions.py --env live --refresh
    python scripts/positions.py --env live --adopt QTEX --opened 2026-11-03

Read-only unless --adopt is passed. It places no orders in any mode: the exit
loop does not exist yet (docs/exit_loop.md), so this is the instrument panel for
a position that is still exited by hand.

Three things are deliberately kept apart in the output, because collapsing them
is how a stop gets believed that is not there:

  BROKER     quantity, average price, last price -- KIS owns these.
  PLAN       the stop, frozen ATR and high-water marks -- we own these, they
             exist nowhere at the broker, and a holding without one is
             unprotected no matter what the P&L looks like.
  DECISION   what strategy/exits.py returns for the last COMPLETED session,
             from the same functions the backtest calls.

The live-price stop check is reported separately from the DECISION block. It
compares the quoted last against the stop level, which is not the same as a
session's exit decision -- thesis-break and the time stop are end-of-day rules
and an intraday print cannot settle them.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from datetime import date, datetime
from pathlib import Path

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uat.broker.kis_auth import MissingCredentials  # noqa: E402
from uat.broker.kis_broker import KisBroker  # noqa: E402
from uat.broker.kis_client import KisClient  # noqa: E402
from uat.clock import ET, current_session_date  # noqa: E402
from uat.config import load_settings  # noqa: E402
from uat.live import context as ctxmod  # noqa: E402
from uat.live import holdings as hold  # noqa: E402
from uat.live.positions import adopt_holding, load_open, reconcile  # noqa: E402
from uat.strategy.exits import (  # noqa: E402
    ExitParams,
    evaluate_exits,
    trail_is_active,
    trailing_stop,
)


def pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


def refresh_bars(symbols: list[str]) -> None:
    """Top up daily bars for the held names only.

    The session-level exit decision is computed from the last completed daily
    bar, so a stale ingest moves both the stop and the trail. A handful of
    symbols is a cheap fetch; the full universe refresh belongs to the morning
    job.
    """
    from uat.data.ingest_daily import ingest_daily

    rep = ingest_daily(symbols, date.today().replace(day=1), date.today())
    print(f"refresh : {rep['rows']:,} rows for {len(symbols)} symbol(s), "
          f"success {rep['success_rate']:.0%}")


def report_broker(held: list, summary) -> None:
    print("\n=== BROKER (KIS 보유) ===")
    for h in held:
        print(f"  {h.symbol:<6} {h.qty:>4}주  평단 ${h.avg_px:<8.4f} 현재 ${h.last_px:<8.4f}  "
              f"평가 ${h.market_value:>7.2f}  손익 ${h.pnl_usd:>+7.2f} "
              f"({h.pnl_pct:+.2f}%)  {h.venue}")
        if h.qty_reserved:
            # Held minus sellable is the direct answer to open question 3 in
            # docs/exit_loop.md: it means a resting sell is still live.
            print(f"         매도가능 {h.qty_sellable}주 -- {h.qty_reserved}주는 "
                  "미체결 매도주문에 잡혀 있음 (레스팅 보호주문 생존 신호)")
    print(f"  ---- 매입원금 ${summary.cost_basis:,.2f}  "
          f"평가금액 ${summary.market_value:,.2f}  "
          f"평가손익 ${summary.unrealised_pnl:+,.2f} ({summary.return_pct:+.2f}%)  "
          f"실현 ${summary.realised_pnl:+,.2f}")


def report_plan(held: list, broker: KisBroker, params: ExitParams, env: str) -> None:
    stored = load_open()
    by_symbol = {p.symbol: (pid, p) for pid, p in stored.items()}

    print("\n=== PLAN / DECISION ===")
    for h in held:
        entry = by_symbol.get(h.symbol)
        if entry is None:
            print(f"  {h.symbol:<6} 로컬 exit plan 없음 -- 스톱이 어디에도 존재하지 않는다.")
            print(f"         기록: python scripts/positions.py --env {env} "
                  f"--adopt {h.symbol} --opened YYYY-MM-DD")
            continue
        pid, pos = entry

        quote = broker.current_price(h.symbol, h.venue)
        last = float(quote.get("last") or 0) or h.last_px

        level, label = pos.stop_px, "stop"
        if trail_is_active(pos, params):
            trail = trailing_stop(pos, params)
            if trail > level:
                level, label = trail, "trail"

        print(f"  {h.symbol:<6} {pid}")
        print(f"         상태 {pos.state.value}  진입 ${pos.entry_px:.4f} "
              f"({pos.opened_on}, {pos.sessions_held} 세션 경과)  "
              f"수량 {pos.qty_open}/{pos.qty_initial}")
        print(f"         유효 {label} ${level:.4f}  현재 ${last:.4f}  "
              f"여유 {pct(level / last - 1)}  "
              f"스톱 체결시 ${(level - pos.entry_px) * pos.qty_open:+.2f}")
        print(f"         TP1 ${pos.entry_px * (1 + params.tp1_gain):.4f}  "
              f"TP2 ${pos.entry_px * (1 + params.tp2_gain):.4f}  "
              f"타임스톱 {pos.sessions_held}/{params.time_stop_sessions} 세션")
        if last <= level:
            print(f"         >>> 현재가가 {label} 아래. 수동 청산 대상.")

        ctx = ctxmod.build(h.symbol)
        if ctx is None:
            print("         바 데이터 없음 -- 세션 단위 판정 불가 (--refresh)")
            continue
        if not ctx.is_current:
            print(f"         [stale] 최신 바 {ctx.bar.dt}, {ctx.stale_sessions} 세션 "
                  "뒤처짐 -- 아래 판정은 그 날짜 기준이다. --refresh 로 갱신")

        # The shared path: the same functions backtest/engine.py calls. A
        # position object is not mutated here -- this reports, it does not act.
        orders = evaluate_exits(
            pos, ctx.bar,
            prev_low=ctx.prev_low,
            median_volume=ctx.median_volume,
            params=params,
        )
        if not orders:
            print(f"         판정({ctx.bar.dt}) : 청산 신호 없음  "
                  f"[종가 ${ctx.bar.close:.4f} SMA20 ${ctx.bar.sma20:.4f} "
                  f"rvol5 {ctx.bar.rvol_5:.2f} ATR ${ctx.bar.atr14:.4f}]")
        for o in orders:
            print(f"         판정({ctx.bar.dt}) : {o.reason.value.upper()} "
                  f"{o.qty}주 @ ${o.limit_px:.4f}  is_stop={o.is_stop}")


def do_adopt(args, held: list, settings) -> None:
    sym = args.adopt.upper()
    match = [h for h in held if h.symbol == sym]
    if not match:
        print(f"\n[abort] 브로커 잔고에 {sym} 없음. 보유: "
              + ", ".join(h.symbol for h in held))
        sys.exit(1)
    if not args.opened:
        print(f"\n[abort] --opened YYYY-MM-DD 가 필요하다. {sym} 의 진입일은 잔고 조회로는 "
              "알 수 없고(주문체결내역 조회도 수동 진입분에는 빈 응답을 준다), "
              "타임스톱이 그 날짜부터 세션을 센다.")
        sys.exit(1)

    ctx = ctxmod.build(sym)
    if ctx is None:
        print(f"\n[abort] {sym} 바 데이터 없음 -- --refresh 를 먼저 실행")
        sys.exit(1)
    try:
        pid, pos = adopt_holding(
            match[0], opened_on=date.fromisoformat(args.opened), ctx=ctx, settings=settings
        )
    except ValueError as exc:
        print(f"\n[abort] {exc}")
        sys.exit(1)

    print(f"\n기록됨 : {pid}")
    print(f"  진입 ${pos.entry_px:.4f} x {pos.qty_open}주  진입일 {pos.opened_on}")
    print(f"  스톱 ${pos.stop_px:.4f} ({pct(pos.stop_px / pos.entry_px - 1)} from entry)  "
          f"ATR(진입일 기준) ${pos.atr_at_entry:.4f}")
    print(f"  경과 {pos.sessions_held} 세션  고가 ${pos.hh_since_entry:.4f}  "
          f"종가고점 ${pos.hc_since_entry:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=["paper", "live"], default="live")
    ap.add_argument("--refresh", action="store_true",
                    help="ingest recent bars for held symbols before reporting")
    ap.add_argument("--adopt", metavar="SYMBOL", default=None,
                    help="record an exit plan for a broker holding we do not track")
    ap.add_argument("--opened", metavar="YYYY-MM-DD", default=None,
                    help="entry date for --adopt; the broker does not report one")
    ap.add_argument("--venues", default="NASD",
                    help="comma-separated balance venues (default NASD, the US umbrella)")
    args = ap.parse_args()

    os.environ["KIS_ENV"] = args.env
    settings = load_settings(args.env)

    try:
        client = KisClient(args.env)
    except MissingCredentials as exc:
        print(f"[credentials] {exc}")
        sys.exit(2)
    broker = KisBroker(client)

    print(f"env     : {settings.env}   "
          f"account {client.creds.account_8}-{client.creds.account_2}")
    print(f"now     : {datetime.now(ET):%Y-%m-%d %H:%M} ET   "
          f"last session {current_session_date()}")

    venues = tuple(v.strip().upper() for v in args.venues.split(",") if v.strip())
    held, summary = hold.fetch(client, venues)
    if not held:
        print("\n보유 종목 없음.")
        broker.close()
        return

    if args.refresh:
        print()
        refresh_bars([h.symbol for h in held])

    if args.adopt:
        do_adopt(args, held, settings)

    report_broker(held, summary)
    report_plan(held, broker, ExitParams.from_settings(settings), args.env)

    issues = reconcile(hold.as_reconcile_input(held))
    print("\n=== RECONCILE ===")
    if not issues:
        print("  브로커 잔고와 로컬 기록 일치")
    for i in issues:
        print(f"  MISMATCH {i}")
    if issues:
        print("  이 상태에서는 신규 진입을 차단하고 청산만 허용한다 (docs/exit_loop.md).")

    broker.close()


if __name__ == "__main__":
    main()
