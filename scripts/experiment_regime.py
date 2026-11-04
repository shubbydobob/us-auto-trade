"""실험 J: 레짐 필터. 2022 를 고칠 수 있는가.

라벨 교체로 게이트는 1.0x +7.26% 까지 왔지만 거래 가능한 결과가 아니다.
-20% 킬스위치가 2022-05-09 에 발동하고, 2022 한 해가 -18.0% 다. 그리고 실험 I 의
라벨 변형 다섯 개가 **전부** 2022-2023 창에서 음수였다 -- 어떤 라벨도 이 구간을
고치지 못한다.

**가설**: 전략은 롱 온리이고 레짐 필터가 없다. 전제는 "눌린 종목이 다시 튄다"
인데(스코어러가 `dist_from_high_252` 를 low 방향으로 가중한다), 전체가 눌린
국면에서는 "눌렸다"가 선별 정보를 담지 않는다. 2022 는 소형주 약세장이었다.

**측정 방법 -- 새 데이터도, 엔진 변경도 필요 없다.** 패널에 이미 있는
`px_vs_sma200` 로 날짜별 폭(게이트 통과 종목 중 SMA200 위 비율)을 계산하고,
폭이 낮은 날의 시그널을 버린다. 보유 포지션은 건드리지 않는다 -- 레짐 필터는
진입 억제이지 청산 규칙이 아니다.

인과성: 폭은 날짜 d 까지의 데이터로 계산되고 시그널도 d 종가에 발생해 d+1 에
진입한다. 미래 정보가 없다.

레짐 필터는 **하나의 나쁜 기간에 하나의 손잡이를 맞추는** 전형적 과적합 형태다.
그래서 수준 임계값(R1~R3)과 함께 임계값이 없는 추세형(R4, R5)을 같이 돌리고,
서수 질문을 던진다 -- 변형이 **모든** 창에서 R0 를 이기는가. 2022 만 고치고
다른 창을 망가뜨리면 그것은 2022 에 대한 커브 피팅이다.

    PYTHONPATH=src python scripts/experiment_regime.py
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

from scripts.experiment_population import run  # noqa: E402
from scripts.run_backtest import make_signals, prepare_cached, random_signals  # noqa: E402

# 연도 단위로도 보기 위해 창을 5개로 쓴다. 2022 를 따로 떼어 놓는 것이 핵심이다.
WINDOWS5 = [
    ("2020", "2020-01-01", "2020-12-31"),
    ("2021", "2021-01-01", "2021-12-31"),
    ("2022", "2022-01-01", "2022-12-31"),
    ("2023", "2023-01-01", "2023-12-31"),
    ("2024-26", "2024-01-01", None),
]


def breadth_series(panel: pd.DataFrame) -> pd.Series:
    """날짜별 폭: 게이트 통과 종목 중 SMA200 위 비율.

    `above_sma200` 은 패널 컬럼이 아니고 `px_vs_sma200`(= close/sma200 - 1)이
    있으므로 그 부호로 센다. 같은 값이다.
    """
    g = panel[panel["gated"] & panel["px_vs_sma200"].notna()]
    return g.groupby("dt")["px_vs_sma200"].apply(lambda x: (x > 0).mean())


def filt_none(b: pd.Series) -> pd.Series:
    return pd.Series(True, index=b.index)


def filt_level(lo: float):
    return lambda b: b >= lo


def filt_above_own_sma(window: int):
    """임계값 없음 -- 폭이 자기 이동평균 위인가. 수준이 아니라 추세를 본다."""
    return lambda b: b > b.rolling(window, min_periods=window // 2).mean()


REGIMES = [
    ("R0 필터 없음", filt_none),
    ("R1 폭 >= 20%", filt_level(0.20)),
    ("R2 폭 >= 30%", filt_level(0.30)),
    ("R3 폭 >= 40%", filt_level(0.40)),
    ("R4 폭 > 자기 50일 평균", filt_above_own_sma(50)),
    ("R5 폭 > 자기 100일 평균", filt_above_own_sma(100)),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost", type=float, default=1.5)
    ap.add_argument("--top-k", type=int, default=15)
    args = ap.parse_args()

    equity = 10_000_000 / 1380.0
    full, s = prepare_cached(date(2019, 1, 1), None, None, 7)
    dts = pd.to_datetime(full["dt"])

    # 폭은 전 구간에서 한 번 계산한다. 창별로 계산하면 창 시작부에서 롤링
    # 평균이 비어 필터가 임의로 꺼진다.
    b = breadth_series(full)
    print(f"\ncost {args.cost}x  top-{args.top_k}  청산/포트폴리오 C_WIDE, 라벨 L4")
    print(f"폭(SMA200 위 비율): 중간값 {b.median() * 100:.1f}%  "
          f"최저 {b.min() * 100:.1f}%  최고 {b.max() * 100:.1f}%")
    by_year = b.groupby(pd.to_datetime(b.index).year).median()
    print("  연도별 중간값: " + "  ".join(f"{y} {v * 100:.0f}%"
                                          for y, v in by_year.items()))

    table: dict[str, dict] = {}
    for label, make in REGIMES:
        allow = make(b)
        ok_dates = set(pd.Index(b.index)[allow.fillna(False)])
        print()
        print(f"=== {label}  (진입 허용 세션 {len(ok_dates):,} / {len(b):,} "
              f"= {len(ok_dates) / len(b) * 100:.0f}%) ===")
        for wl, a, bb in WINDOWS5:
            m = dts >= pd.Timestamp(a)
            if bb:
                m &= dts <= pd.Timestamp(bb)
            win = full[m]
            sig = make_signals(win, "evidence", args.top_k)
            sig = sig[sig["dt"].isin(ok_dates)]
            r = run(win, sig, s, equity, args.cost) if len(sig) else None
            table.setdefault(label, {})[wl] = r
            if r:
                print(f"  {wl:<9} n={r['n']:>4}  적중 {r['hit'] * 100:>5.1f}%  "
                      f"손익비 {r['payoff']:>5.2f}  CAGR {r['cagr'] * 100:>7.2f}%  "
                      f"maxDD {r['maxdd'] * 100:>6.1f}%")
            else:
                print(f"  {wl:<9} 트레이드 없음")

    print()
    print("=" * 104)
    print("서수 요약 -- R0 대비. 2022 만 고치고 다른 창을 망가뜨리면 커브 피팅이다")
    print("=" * 104)
    base = REGIMES[0][0]
    print(f"{'레짐':<24}" + "".join(f"{w:>11}" for w, _, _ in WINDOWS5)
          + f"{'평균':>9}{'R0대비':>8}{'평균DD':>9}")
    for label, _ in REGIMES:
        if label not in table:
            continue
        cells = "".join(
            f"{(table[label].get(w) or {}).get('cagr', float('nan')) * 100:>10.2f}%"
            for w, _, _ in WINDOWS5
        )
        cagrs = [v["cagr"] for v in table[label].values() if v]
        dds = [v["maxdd"] for v in table[label].values() if v]
        wins = sum(
            1 for w in table[label]
            if table[label][w] and (table.get(base, {}).get(w) or {})
            and table[label][w]["cagr"] > table[base][w]["cagr"]
        )
        tag = "--" if label == base else f"{wins}/{len(WINDOWS5)}"
        print(f"{label:<24}{cells}{pd.Series(cagrs).mean() * 100:>8.2f}%{tag:>8}"
              f"{pd.Series(dds).mean() * 100:>8.1f}%")

    print()
    for label, _ in REGIMES:
        if label == base or label not in table:
            continue
        wins = sum(
            1 for w in table[label]
            if table[label][w] and (table.get(base, {}).get(w) or {})
            and table[label][w]["cagr"] > table[base][w]["cagr"]
        )
        r22 = (table[label].get("2022") or {}).get("cagr")
        b22 = (table[base].get("2022") or {}).get("cagr")
        fix = ("2022 개선" if r22 is not None and b22 is not None and r22 > b22
               else "2022 개선 못함")
        if wins == len(WINDOWS5):
            print(f"{label}: 모든 창에서 R0 를 이긴다. 구조적 결과다. ({fix})")
        elif wins == 0:
            print(f"{label}: 어느 창에서도 이기지 못한다. 기각. ({fix})")
        else:
            print(f"{label}: {wins}/{len(WINDOWS5)} -- 기간 의존. ({fix})")


if __name__ == "__main__":
    main()
