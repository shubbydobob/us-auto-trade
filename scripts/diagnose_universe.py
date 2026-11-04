"""게이트 통과 뒤에 남은 문제 1·2번을 측정한다.

Phase 3 게이트는 통과했지만 600 트레이드 중 423 건이 2020-2021 에 몰려 있고
2023/2024 는 연 20건뿐이다. 그 상태의 "통과"는 2023/2024 에서 전략이 작동했다는
증거가 아니라 **측정되지 않았다**는 증거다. 그리고 2025 한 해가 +14.0% 로 혼자
크다.

둘 다 선택이 아니라 측정이다. 이 스크립트는 파라미터를 고르지 않는다.

1. 유니버스 게이트가 연도별로 몇 종목-일을 통과시키는지, 떨어진 것은 어느 조건에
   걸렸는지. 트레이드가 없는 해는 전략이 쉰 것이 아니라 모집단이 비었을 수 있다
2. 2025 를 제외하면 수익률이 얼마나 남는지

    PYTHONPATH=src python scripts/diagnose_universe.py
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

# Windows 콘솔 기본 cp949 는 한글 출력을 중간에 죽인다. 다른 스크립트들과 동일.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uat.config import load_settings  # noqa: E402
from uat.data.ingest_daily import load_bars  # noqa: E402
from uat.universe.filters import bar_level_metrics, classify_tier  # noqa: E402

# 게이트 실행(logs/gate_after_adoption.log)의 per-year 표에서 그대로 옮긴 값.
# 추정이 아니라 실측이고, 소수점은 리포트가 출력한 자리를 유지했다.
GATE_YEARLY = {
    2019: 0.002179, 2020: 0.068628, 2021: 0.021184, 2022: 0.039556,
    2023: 0.014888, 2024: 0.037924, 2025: 0.139605, 2026: 0.021774,
}
GATE_TRADES = {2019: 0, 2020: 215, 2021: 208, 2022: 37, 2023: 20,
               2024: 20, 2025: 69, 2026: 31}


def year_exclusion_table() -> None:
    """한 해를 빼면 무엇이 남는지.

    주의: 중간의 한 해를 제거하는 것은 그 해에 전략이 평탄했다고 가정하는 것이고,
    연말을 넘어 유지되는 포지션 상태를 무시한다. 별도 백테스트 실행이 아니라
    연도별 수익률을 곱한 산수이므로 근사다.
    """
    print()
    print("=" * 72)
    print("2. 한 해를 제외하면 무엇이 남는가 (연도별 수익률 연쇄, 근사)")
    print("=" * 72)
    years = sorted(GATE_YEARLY)
    base = 1.0
    for y in years:
        base *= 1 + GATE_YEARLY[y]
    n = len(years)
    print(f"  전체 {years[0]}-{years[-1]}  누적 {(base - 1) * 100:+.2f}%  "
          f"연환산 {((base ** (1 / n)) - 1) * 100:+.2f}%  "
          f"(리포트 실측 CAGR +5.12%, 총수익 +40.71%)")
    print()
    print(f"  {'제외한 해':<12}{'누적':>12}{'연환산':>12}{'그 해 트레이드':>16}")
    for drop in years:
        if GATE_TRADES[drop] == 0:
            continue
        cum = 1.0
        for y in years:
            if y != drop:
                cum *= 1 + GATE_YEARLY[y]
        k = n - 1
        print(f"  {drop:<12}{(cum - 1) * 100:>11.2f}%"
              f"{((cum ** (1 / k)) - 1) * 100:>11.2f}%{GATE_TRADES[drop]:>16}")
    print()
    print("  한 해를 빼도 부호가 뒤집히지 않으면 그 해에 업힌 결과가 아니다.")


def universe_by_year(start: str) -> None:
    s = load_settings()
    print("bars 적재 ...")
    bars = load_bars(start=pd.Timestamp(start).date())
    print(f"  {len(bars):,} bars / {bars.symbol.nunique():,} symbols")

    print("게이트 판정 ...")
    tiered = classify_tier(bar_level_metrics(bars), settings=s)
    tiered["year"] = pd.to_datetime(tiered["dt"]).dt.year
    gated = tiered["tier"].isin(["core", "watch"])

    print()
    print("=" * 96)
    print("1. 유니버스 게이트를 통과한 종목-일, 연도별")
    print("=" * 96)
    print(f"  {'year':<6}{'전체 종목-일':>14}{'통과':>10}{'통과율':>9}"
          f"{'통과 종목수':>12}{'core':>9}{'watch':>9}{'트레이드':>10}")
    for y, g in tiered.groupby("year"):
        ok = g[g["tier"].isin(["core", "watch"])]
        print(f"  {y:<6}{len(g):>14,}{len(ok):>10,}"
              f"{(len(ok) / max(len(g), 1)) * 100:>8.2f}%"
              f"{ok['symbol'].nunique():>12,}"
              f"{(ok['tier'] == 'core').sum():>9,}"
              f"{(ok['tier'] == 'watch').sum():>9,}"
              f"{GATE_TRADES.get(y, 0):>10}")

    print()
    print("탈락 사유, 연도별 비중 (해당 연도 전체 종목-일 대비 %)")
    rej = tiered[~gated]
    tab = (
        rej.pivot_table(index="reject_reason", columns="year",
                        values="symbol", aggfunc="count")
        .fillna(0)
    )
    totals = tiered.groupby("year").size()
    share = (tab / totals * 100).round(1)
    share["평균"] = share.mean(axis=1).round(1)
    print(share.sort_values("평균", ascending=False).to_string())

    print()
    print("가격대 분포 -- 가격 밴드 밖으로 떨어진 것이 어느 쪽인지")
    out_band = rej[rej["reject_reason"] == "outside_price_band"]
    if not out_band.empty:
        below = out_band["close"] < float(s.get("universe.watch_price_min"))
        by_year = out_band.assign(below=below).groupby("year")["below"]
        print(f"  {'year':<6}{'밴드밖 종목-일':>16}{'하단 이탈(<$0.5)':>18}"
              f"{'상단 이탈(>$10)':>18}")
        for y, grp in by_year:
            print(f"  {y:<6}{len(grp):>16,}{int(grp.sum()):>18,}"
                  f"{int((~grp).sum()):>18,}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--skip-universe", action="store_true")
    args = ap.parse_args()

    year_exclusion_table()
    if not args.skip_universe:
        universe_by_year(args.start)


if __name__ == "__main__":
    main()
