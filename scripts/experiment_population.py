"""실험 H: 모집단. 꼬리 의존·깊은 DD·큰 평균패가 같은 원인인지 본다.

C_WIDE 로 게이트는 넘겼지만 거래 가능한 결과가 아니다. 세 증상이 같은 방향을
가리킨다.

    상위 5 트레이드가 P&L 의 154.9%, 제외하면 적자
    최대DD -38.11%, -20% 킬스위치가 2022-01-21 에 발동
    평균패 -46.5%

가설: 모집단이 꼬리에 너무 치우쳐 있다. $1.20 종목에서 1센트 틱은 83bps 이고,
`watch` 티어는 $0.50 까지 내려간다. 거기서는 스톱 폭이 가격의 20~50% 가 되어야
의미가 있고, 그러면 패배가 구조적으로 커진다. 가격·유동성 하한을 올리면 꼬리가
얇아지고 비용도 같이 내려간다.

실험 B 는 "유동성 밴드 실패, 비용은 장애가 아니다"로 이 방향을 이미 기각했다.
그러나 그 측정은 좀비 포지션 버그(18efc6e) 위에서 이뤄졌다 -- 패자가 원가에
동결되는 엔진에서는 패배를 줄이는 변경의 이득이 보이지 않는다. 기각 근거가
무효가 되었으므로 다시 묻는다.

방법: 선택 없음. 사전 확정한 모집단 7개를 비중첩 3개 창에 돌리고, 청산/포트폴리오
설정은 C_WIDE 로 고정한다. 질문은 서수다 -- 변형이 현행 모집단보다 DD 를 줄이는가,
꼬리 의존을 줄이는가, 그러면서 수익을 잃지 않는가.

캐시된 패널은 left join 이라 비게이트 행까지 들고 있으므로, `gated` 플래그만 다시
계산하면 피처 재계산 없이 모집단을 바꿀 수 있다.

    PYTHONPATH=src python scripts/experiment_population.py
"""
from __future__ import annotations

import argparse
import sys
import warnings
from datetime import date
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.experiment_walkforward import LADDERLESS, WINDOWS  # noqa: E402
from scripts.run_backtest import make_signals, prepare_cached, random_signals  # noqa: E402
from uat.backtest.costs import CostParams  # noqa: E402
from uat.backtest.engine import Backtester, EntryParams, PortfolioParams  # noqa: E402
from uat.backtest.metrics import full_report  # noqa: E402
from uat.strategy.exits import ExitParams  # noqa: E402

from dataclasses import replace  # noqa: E402

# 채택된 설정. 모집단만 바꾸고 이쪽은 고정한다.
C_WIDE_EX = {**LADDERLESS, "thesis_enabled": False, "time_stop_sessions": 10_000}
C_WIDE_PP = dict(max_concurrent=25, risk_per_trade=0.004,
                 max_portfolio_heat=0.10, max_new_per_day=8)


def pop_current(df: pd.DataFrame) -> pd.Series:
    return df["gated"]


def pop_price(lo: float) -> callable:
    """가격 하한만 올린다. 틱 비용이 가격에 반비례하므로 가장 직접적인 손잡이다."""
    return lambda df: df["gated"] & (df["close"] >= lo)


def pop_core_only(df: pd.DataFrame) -> pd.Series:
    """watch 티어($0.50~1.00, $7~10)를 버리고 core($1~7)만."""
    return df["tier"].eq("core")


def pop_addv(floor: float) -> callable:
    return lambda df: df["gated"] & (df["addv_20"] >= floor)


def pop_price_addv(lo: float, floor: float) -> callable:
    return lambda df: df["gated"] & (df["close"] >= lo) & (df["addv_20"] >= floor)


def pop_spread(cap: float) -> callable:
    return lambda df: df["gated"] & (df["cs_spread_20"] <= cap)


POPULATIONS = [
    ("P0 현행 ($0.5~10, ADDV 300k)", pop_current),
    ("P1 가격 >= $2", pop_price(2.0)),
    ("P2 가격 >= $3", pop_price(3.0)),
    ("P3 core 만 ($1~7)", pop_core_only),
    ("P4 ADDV >= $1M", pop_addv(1_000_000)),
    ("P5 가격 >= $2 + ADDV >= $1M", pop_price_addv(2.0, 1_000_000)),
    ("P6 스프레드 <= 1.5%", pop_spread(0.015)),
]


def run(panel, signals, settings, equity, cost_mult):
    pp = replace(PortfolioParams.from_settings(settings, initial_equity=equity),
                 enforce_drawdown_halt=False, **C_WIDE_PP)
    ex = replace(ExitParams.from_settings(settings), **C_WIDE_EX)
    cost = replace(CostParams.from_settings(settings), cost_multiplier=cost_mult)
    res = Backtester(panel, signals, cost,
                     EntryParams.from_settings(settings), ex, pp).run()
    rep = full_report(res.trades, res.equity)
    t, e, tl = rep["trades"], rep["equity"], rep["tails"]
    if not t.get("n_trades"):
        return None
    return {
        "n": t["n_trades"], "hit": t["hit_rate"], "payoff": t["payoff_ratio"],
        "avg_loss": t.get("avg_loss", float("nan")),
        "cagr": e["cagr"], "maxdd": e["max_drawdown"],
        "top5": tl.get("top5_share", float("nan")),
        "ok_excl_top5": tl.get("profitable_excl_top5", False),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost", type=float, default=1.5)
    ap.add_argument("--top-k", type=int, default=15)
    args = ap.parse_args()

    equity = 10_000_000 / 1380.0
    full, s = prepare_cached(date(2019, 1, 1), None, None, 7)
    dts = pd.to_datetime(full["dt"])

    print(f"\ncost {args.cost}x  top-{args.top_k}  청산/포트폴리오는 C_WIDE 고정")
    print("모집단만 바꾼다. 선택 없음 -- 사전 확정 7개를 3개 창에 전부 돌린다.")

    table: dict[str, dict] = {}
    for wlabel, start, end in WINDOWS:
        mask = dts >= pd.Timestamp(start)
        if end:
            mask &= dts <= pd.Timestamp(end)
        win = full[mask].copy()

        print()
        print(f"--- {wlabel} ---")
        print(f"{'모집단':<30}{'게이트행':>10}{'n':>6}{'적중':>7}{'손익비':>7}"
              f"{'평균패':>8}{'CAGR':>9}{'maxDD':>8}{'top5':>8}{'top5제외':>9}")
        for label, fn in POPULATIONS:
            w = win.copy()
            w["gated"] = fn(w).fillna(False)
            n_gated = int(w["gated"].sum())
            sig = make_signals(w, "evidence", args.top_k)
            r = run(w, sig, s, equity, args.cost)
            if r is None:
                print(f"{label:<30}{n_gated:>10,}{'-- no trades --':>40}")
                continue
            table.setdefault(label, {})[wlabel] = r
            print(f"{label:<30}{n_gated:>10,}{r['n']:>6}{r['hit'] * 100:>6.1f}%"
                  f"{r['payoff']:>7.2f}{r['avg_loss'] * 100:>7.1f}%"
                  f"{r['cagr'] * 100:>8.2f}%{r['maxdd'] * 100:>7.1f}%"
                  f"{r['top5'] * 100:>7.0f}%{str(r['ok_excl_top5']):>9}")

    print()
    print("=" * 110)
    print("서수 요약 -- 현행(P0) 대비")
    print("=" * 110)
    base = POPULATIONS[0][0]
    print(f"{'모집단':<30}" + "".join(f"{w:>12}" for w, _, _ in WINDOWS)
          + f"{'평균CAGR':>10}{'평균maxDD':>11}{'CAGR승':>8}{'DD승':>7}")
    for label, _ in POPULATIONS:
        if label not in table:
            continue
        cells = "".join(
            f"{table[label].get(w, {}).get('cagr', float('nan')) * 100:>11.2f}%"
            for w, _, _ in WINDOWS
        )
        cagrs = [table[label][w]["cagr"] for w in table[label]]
        dds = [table[label][w]["maxdd"] for w in table[label]]
        cagr_win = sum(
            1 for w in table[label]
            if table[label][w]["cagr"] > table[base].get(w, {}).get("cagr", 9e9)
        )
        dd_win = sum(
            1 for w in table[label]
            if table[label][w]["maxdd"] > table[base].get(w, {}).get("maxdd", -9e9)
        )
        tag_c = "--" if label == base else f"{cagr_win}/{len(WINDOWS)}"
        tag_d = "--" if label == base else f"{dd_win}/{len(WINDOWS)}"
        print(f"{label:<30}{cells}{pd.Series(cagrs).mean() * 100:>9.2f}%"
              f"{pd.Series(dds).mean() * 100:>10.1f}%{tag_c:>8}{tag_d:>7}")

    print()
    print("DD승 = 세 창 중 몇 개에서 현행보다 최대DD 가 얕았는가.")
    print("가설(모집단이 꼬리에 치우쳐 DD·평균패·꼬리의존을 만든다)이 맞다면")
    print("가격/유동성 하한을 올린 변형이 DD 와 top5 를 동시에 낮춰야 한다.")


if __name__ == "__main__":
    main()
