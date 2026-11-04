"""실험 I: 라벨 지평선. 시그널이 예측하는 것과 전략이 수확하는 것을 맞춘다.

청산·슬롯 레이어는 끝났고 모집단 가설은 기각됐다. 남은 불일치는 하나다.

    라벨   +60% / -20% / 20세션 삼중배리어
    청산   스톱/트레일만 (C_WIDE) -- 사실상 무한 보유, 실측 평균 ~58세션

시그널은 "20세션 안에 +60% 가는가"를 맞히도록 가중치가 유도됐는데, 실제로 돈을
버는 경로는 "크게 가는 종목을 트레일로 끝까지 따라가는가"다. 20세션 지평선은
채택된 청산이 수확하는 구간보다 짧다.

라벨을 바꾸면 lift 테이블과 가중치가 전부 다시 계산되므로 시그널 자체가 달라진다.
Phase 2 로 되돌아가는 작업이고, 지금까지의 검증은 이 축에 대해서는 리셋된다.

방법 -- 사람이 피처를 고르지 않는다:

1. 라벨 변형별로 `build_labels` 로 전 종목 라벨을 다시 계산
2. **TRAIN(2019~2023-11, 30일 엠바고) 에서만** lift 테이블을 돌려
   `walkforward_check.derive_weights` 로 가중치를 자동 유도. 사람의 선택이 없다
3. 그 가중치로 각 창을 채점해 시그널을 만들고 C_WIDE 로 백테스트
4. 서수 질문: 변형이 L0 를 **모든** 창에서 이기는가

변형은 사전 확정이고 전부 보고한다. 선택은 없다.

    PYTHONPATH=src python scripts/experiment_label.py
    PYTHONPATH=src python scripts/experiment_label.py --only L0 L1
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.experiment_population import C_WIDE_EX, C_WIDE_PP  # noqa: E402
from scripts.experiment_walkforward import WINDOWS  # noqa: E402
from scripts.run_backtest import prepare_cached, random_signals  # noqa: E402
from scripts.walkforward_check import derive_weights  # noqa: E402
from uat.backtest.costs import CostParams  # noqa: E402
from uat.backtest.engine import Backtester, EntryParams, PortfolioParams  # noqa: E402
from uat.backtest.metrics import full_report  # noqa: E402
from uat.labeling.triple_barrier import build_labels  # noqa: E402
from uat.model.rules import SCORERS, score, top_k  # noqa: E402
from uat.strategy.exits import ExitParams  # noqa: E402

TRAIN_END = pd.Timestamp("2023-12-01")
# 라벨 지평선이 train/test 경계를 넘어 겹치지 않도록 엠바고를 둔다. 가장 긴
# 지평선(60세션)이 약 3개월이므로 100일로 잡는다.
EMBARGO_DAYS = 100

# (라벨, horizon, upper, lower). 전부 사전 확정.
LABELS = [
    ("L0 현행 20세션 +60/-20", 20, 0.60, -0.20),
    ("L1 60세션 +60/-20", 60, 0.60, -0.20),
    ("L2 60세션 +100/-20", 60, 1.00, -0.20),
    ("L3 20세션 +30/-20", 20, 0.30, -0.20),
    ("L4 60세션 +60/-35", 60, 0.60, -0.35),
]

# 가중치 유도에 쓰는 후보 피처는 패널에 실제로 있는 것만. walkforward_check 의
# CANDIDATES 는 패널이 담지 않는 피처도 포함하므로, SCORERS 전체의 합집합을
# 후보로 쓴다 -- 사람이 고르는 것이 아니라 "쓸 수 있는 전부"다.
CANDIDATE_POOL = sorted({f for w in SCORERS.values() for f, _, _ in w})


def run_bt(panel, signals, settings, equity, cost_mult):
    pp = replace(PortfolioParams.from_settings(settings, initial_equity=equity),
                 enforce_drawdown_halt=False, **C_WIDE_PP)
    ex = replace(ExitParams.from_settings(settings), **C_WIDE_EX)
    cost = replace(CostParams.from_settings(settings), cost_multiplier=cost_mult)
    res = Backtester(panel, signals, cost,
                     EntryParams.from_settings(settings), ex, pp).run()
    rep = full_report(res.trades, res.equity)
    t, e = rep["trades"], rep["equity"]
    if not t.get("n_trades"):
        return None
    return {"n": t["n_trades"], "hit": t["hit_rate"], "payoff": t["payoff_ratio"],
            "cagr": e["cagr"], "maxdd": e["max_drawdown"]}


def signals_from(panel: pd.DataFrame, weights, k: int) -> pd.DataFrame:
    pool = panel[panel["gated"]]
    picks = top_k(score(pool, weights), k=k)
    return picks[["dt", "symbol", "score"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost", type=float, default=1.5)
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--only", nargs="*", default=None,
                    help="라벨 접두어로 일부만 실행 (예: L0 L1)")
    args = ap.parse_args()

    equity = 10_000_000 / 1380.0
    full, s = prepare_cached(date(2019, 1, 1), None, None, 7)
    dts = pd.to_datetime(full["dt"])
    bars = full[["symbol", "dt", "open", "high", "low", "close"]].copy()

    print(f"\ncost {args.cost}x  top-{args.top_k}  청산/포트폴리오 C_WIDE 고정")
    print(f"가중치는 TRAIN(< {TRAIN_END.date()} - {EMBARGO_DAYS}일 엠바고)에서만 "
          "자동 유도한다. 사람이 피처를 고르지 않는다.")
    from scripts.walkforward_check import CANDIDATES
    avail = [c for c in CANDIDATES if c in full.columns]
    print(f"후보 피처 {len(avail)}/{len(CANDIDATES)}개가 패널에 존재: "
          + ", ".join(avail))

    table: dict[str, dict] = {}
    chosen: dict[str, list] = {}

    for label, horizon, upper, lower in LABELS:
        if args.only and not any(label.startswith(p) for p in args.only):
            continue
        t0 = time.time()
        print()
        print(f"=== {label}  (horizon={horizon}, upper={upper:+.0%}, "
              f"lower={lower:+.0%}) ===")
        lab = build_labels(bars, horizon=horizon, upper=upper, lower=lower)
        print(f"  라벨 {len(lab):,}행  {time.time() - t0:.0f}s  "
              f"제외 {lab.excluded.mean() * 100:.1f}%")

        panel = full.merge(lab[["symbol", "dt", "y_explode", "excluded", "mfe", "mae"]],
                           on=["symbol", "dt"], how="left", suffixes=("", "_lab"))
        usable = panel[panel["gated"] & ~panel["excluded"].fillna(True)]
        base = usable["y_explode"].mean()
        print(f"  기저 확률 {base * 100:.2f}%  (게이트·미제외 {len(usable):,}행)")

        embargo = TRAIN_END - pd.Timedelta(days=EMBARGO_DAYS)
        train = usable[pd.to_datetime(usable["dt"]) < embargo]
        # derive_weights 는 walkforward_check.CANDIDATES 중 패널에 있는 것만
        # 쓴다. 사람이 고르는 것이 아니라 train lift 가 1.25 를 넘는 것만 뽑는다.
        weights = derive_weights(train)
        if not weights:
            print("  가중치 유도 실패 (lift 1.25 를 넘는 피처 없음) -- 건너뜀")
            continue
        chosen[label] = weights
        print("  TRAIN 유도 가중치:")
        for f, d, w in weights:
            print(f"    {f:<22} {d:<5} {w:5.1f}")

        for wl, a, b in WINDOWS:
            m = dts >= pd.Timestamp(a)
            if b:
                m &= dts <= pd.Timestamp(b)
            win = panel[m]
            r = run_bt(win, signals_from(win, weights, args.top_k), s, equity,
                       args.cost)
            table.setdefault(label, {})[wl] = r
            if r:
                print(f"    {wl:<12} n={r['n']:>5}  적중 {r['hit'] * 100:>5.1f}%  "
                      f"손익비 {r['payoff']:>5.2f}  CAGR {r['cagr'] * 100:>7.2f}%  "
                      f"maxDD {r['maxdd'] * 100:>6.1f}%")
            else:
                print(f"    {wl:<12} 트레이드 없음")
        print(f"  소요 {time.time() - t0:.0f}s")

    print()
    print("=" * 96)
    print("서수 요약 -- L0 대비. 모든 창에서 이겨야 구조적 결과다")
    print("=" * 96)
    base_label = LABELS[0][0]
    print(f"{'라벨':<26}" + "".join(f"{w:>14}" for w, _, _ in WINDOWS)
          + f"{'평균':>9}{'L0대비':>9}")
    for label, *_ in LABELS:
        if label not in table:
            continue
        cells = "".join(
            f"{(table[label].get(w) or {}).get('cagr', float('nan')) * 100:>13.2f}%"
            for w, _, _ in WINDOWS
        )
        cagrs = [v["cagr"] for v in table[label].values() if v]
        wins = sum(
            1 for w in table[label]
            if table[label][w] and (table.get(base_label, {}).get(w) or {})
            and table[label][w]["cagr"] > table[base_label][w]["cagr"]
        )
        tag = "--" if label == base_label else f"{wins}/{len(WINDOWS)}"
        print(f"{label:<26}{cells}{pd.Series(cagrs).mean() * 100:>8.2f}%{tag:>9}")

    print()
    for label, *_ in LABELS:
        if label == base_label or label not in table:
            continue
        wins = sum(
            1 for w in table[label]
            if table[label][w] and (table.get(base_label, {}).get(w) or {})
            and table[label][w]["cagr"] > table[base_label][w]["cagr"]
        )
        if wins == len(WINDOWS):
            print(f"{label}: 모든 창에서 L0 를 이긴다. 구조적 결과다.")
        elif wins == 0:
            print(f"{label}: 어느 창에서도 이기지 못한다. 기각.")
        else:
            print(f"{label}: {wins}/{len(WINDOWS)} -- 기간 의존, 구조적 결과가 아니다.")


if __name__ == "__main__":
    main()
