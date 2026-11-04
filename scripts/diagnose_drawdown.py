"""최대DD 지속 1,002일의 원인을 측정한다.

게이트는 통과했지만 최대DD 지속이 1,002일(약 2.7년)이다. 수익률보다 이 숫자가
운영을 끝낼 수 있고, "2.7년 물밑을 견딜 수 있는가"는 백테스트가 답하지 못한다.
답할 수 있는 것은 **왜 그렇게 긴가**이고, 그것이 고칠 수 있는 종류인지다.

가설: 자본 고착. 타임스톱은 15세션 이후 수익률이 +12% 아래일 때만 발동하고,
트레일은 +40% 부터 추적한다. 그 사이 구간(+12% ~ +40%)에 머무는 포지션은 어느
청산에도 걸리지 않고 무기한 보유된다. 진입 차단 사유가 max_concurrent 26,036 건인
것과 합치면, 죽지 않은 포지션이 슬롯을 점유한 채 회복 국면의 기회를 막는 그림이다.

측정하는 것:

1. DD 에피소드 -- 어디서 시작해 언제 저점을 찍고 언제 회복했는지
2. 보유기간 분포 -- 세션 버킷별 트레이드 수, 수익률, 청산 사유
3. 자본-시간 점유 -- 장기 보유 포지션이 슬롯-일을 얼마나 먹는지

    PYTHONPATH=src python scripts/diagnose_drawdown.py
"""
from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_backtest import make_signals, prepare_cached  # noqa: E402
from uat.backtest.costs import CostParams  # noqa: E402
from uat.backtest.engine import Backtester, EntryParams, PortfolioParams  # noqa: E402
from uat.config import load_settings  # noqa: E402
from uat.strategy.exits import ExitParams  # noqa: E402


def dd_episodes(eq: pd.DataFrame, min_depth: float = 0.03) -> pd.DataFrame:
    """고점 대비 하락 구간을 에피소드로 분해한다.

    에피소드는 신고점에서 시작해 다음 신고점에서 끝난다. 회복하지 못한 마지막
    구간은 `recovered=False` 로 남긴다 -- 지속일수를 "아직 진행 중"과 "끝난 것"으로
    섞어 보고하면 최악값이 과소평가된다.
    """
    e = eq.sort_values("dt").reset_index(drop=True)
    v = e["equity"].to_numpy(float)
    dt = pd.to_datetime(e["dt"]).to_numpy()
    peak = np.maximum.accumulate(v)
    under = v < peak

    rows, i, n = [], 0, len(v)
    while i < n:
        if not under[i]:
            i += 1
            continue
        start = i
        while i < n and under[i]:
            i += 1
        seg = slice(start, i)
        trough = start + int(np.argmin(v[seg]))
        depth = v[trough] / peak[start] - 1
        recovered = i < n
        rows.append({
            "peak_dt": pd.Timestamp(dt[max(start - 1, 0)]).date(),
            "trough_dt": pd.Timestamp(dt[trough]).date(),
            "end_dt": pd.Timestamp(dt[min(i, n - 1)]).date(),
            "depth": depth,
            "to_trough_days": (pd.Timestamp(dt[trough])
                               - pd.Timestamp(dt[max(start - 1, 0)])).days,
            "duration_days": (pd.Timestamp(dt[min(i, n - 1)])
                              - pd.Timestamp(dt[max(start - 1, 0)])).days,
            "recovered": recovered,
        })
    df = pd.DataFrame(rows)
    return df[df["depth"] <= -min_depth].sort_values("duration_days",
                                                     ascending=False)


def run_once(panel, signals, settings, equity, cost_mult, ex_kw=None, pp_kw=None):
    pp = replace(PortfolioParams.from_settings(settings, initial_equity=equity),
                 enforce_drawdown_halt=False, **(pp_kw or {}))
    ex = replace(ExitParams.from_settings(settings), **(ex_kw or {}))
    cost = replace(CostParams.from_settings(settings), cost_multiplier=cost_mult)
    return Backtester(panel, signals, cost,
                      EntryParams.from_settings(settings), ex, pp).run()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=15)
    args = ap.parse_args()

    s = load_settings()
    equity = 10_000_000 / 1380.0
    panel, s = prepare_cached(date(2019, 1, 1), None, None, 7)
    signals = make_signals(panel, "evidence", args.top_k)
    res = run_once(panel, signals, s, equity, args.cost)
    tr = res.trades

    print()
    print("=" * 92)
    print("1. 드로다운 에피소드 (깊이 3% 이상, 지속일 내림차순)")
    print("=" * 92)
    ep = dd_episodes(res.equity)
    print(f"  에피소드 {len(ep)}개, 회복 못한 구간 {int((~ep.recovered).sum())}개")
    show = ep.head(8).copy()
    show["depth"] = (show["depth"] * 100).round(1)
    print(show.to_string(index=False))

    print()
    print("=" * 92)
    print("2. 보유기간 버킷별 트레이드 (sessions_held)")
    print("=" * 92)
    buckets = [0, 5, 15, 30, 60, 120, 250, 10_000]
    labels = ["1-5", "6-15", "16-30", "31-60", "61-120", "121-250", "250+"]
    tr = tr.copy()
    tr["bucket"] = pd.cut(tr["sessions_held"], bins=buckets, labels=labels)
    agg = tr.groupby("bucket", observed=True).agg(
        n=("return_pct", "size"),
        mean_ret=("return_pct", "mean"),
        total_pnl=("net_pnl", "sum"),
        hit=("return_pct", lambda x: (x > 0).mean()),
        mean_sessions=("sessions_held", "mean"),
    )
    agg["slot_days"] = tr.groupby("bucket", observed=True)["sessions_held"].sum()
    agg["slot_day_share"] = (agg["slot_days"] / agg["slot_days"].sum() * 100)
    agg["mean_ret"] = (agg["mean_ret"] * 100).round(1)
    agg["hit"] = (agg["hit"] * 100).round(1)
    agg["total_pnl"] = agg["total_pnl"].round(0)
    agg["slot_day_share"] = agg["slot_day_share"].round(1)
    agg["mean_sessions"] = agg["mean_sessions"].round(0)
    print(agg.to_string())

    print()
    print("  슬롯-일 점유 대비 P&L -- 자본을 오래 잡아두는 버킷이 값을 하는가")
    for b in agg.index:
        sd = agg.loc[b, "slot_days"]
        pnl = agg.loc[b, "total_pnl"]
        print(f"    {b:<10} 슬롯-일 {sd:>8,.0f} ({agg.loc[b, 'slot_day_share']:>4.1f}%)"
              f"   P&L ${pnl:>9,.0f}   $/슬롯-일 {pnl / max(sd, 1):>7.3f}")

    print()
    print("=" * 92)
    print("3. 청산 사유 x 보유기간")
    print("=" * 92)
    piv = tr.pivot_table(index="exit_reason", columns="bucket", values="return_pct",
                         aggfunc="size", observed=True).fillna(0).astype(int)
    print(piv.to_string())

    print()
    print("  장기 보유분의 청산 사유별 평균 수익률")
    long_held = tr[tr["sessions_held"] > 60]
    if not long_held.empty:
        g = long_held.groupby("exit_reason")["return_pct"]
        for reason, vals in g:
            print(f"    {reason:<12} n={len(vals):>4}  평균 {vals.mean() * 100:>7.1f}%"
                  f"  중간값 {vals.median() * 100:>7.1f}%")
    print(f"  60세션 초과 보유: {len(long_held)}건 / {len(tr)}건 "
          f"({len(long_held) / max(len(tr), 1) * 100:.1f}%), "
          f"슬롯-일 점유 {long_held['sessions_held'].sum() / tr['sessions_held'].sum() * 100:.1f}%")


if __name__ == "__main__":
    main()
