"""Phase 3 gate: full backtest with cost stress and a shuffled control.

    python scripts/run_backtest.py --start 2019-01-01 --capital 10000000 \
        --scorer evidence --stress 1.0 1.5 2.0 --control

GATE: the strategy must still beat the shuffled control at a 1.5x cost
multiplier. The Corwin-Schultz spread estimator is uncalibrated, so passing
only at 1.0x would be passing on an assumption rather than a measurement.
"""
from __future__ import annotations

import argparse
import json
import warnings
from datetime import date

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from uat.backtest.costs import CostParams  # noqa: E402
from uat.backtest.engine import (  # noqa: E402
    Backtester,
    EntryParams,
    PortfolioParams,
)
from uat.backtest.halts import flag_halts  # noqa: E402
from uat.backtest.metrics import (  # noqa: E402
    exit_reason_breakdown,
    format_report,
    full_report,
    return_distribution,
    yearly_breakdown,
)
from uat.config import load_settings, project_root  # noqa: E402
from uat.data.ingest_daily import load_bars  # noqa: E402
from uat.features.pipeline import build_features  # noqa: E402
from uat.model.rules import SCORERS, score, top_k  # noqa: E402
from uat.strategy.exits import ExitParams  # noqa: E402
from uat.universe.filters import bar_level_metrics, classify_tier  # noqa: E402

KRW_PER_USD = 1380.0

# Bump when the panel's columns or their computation change, so a cached panel
# from before the change is never silently reused. sma20 moving into the
# feature pipeline is exactly the kind of change that has to invalidate it.
FEATURE_PANEL_VERSION = 3


def prepare(start: date, end: date | None, max_symbols: int | None, seed: int):
    """Bars + features + universe gate + halt flags, ready for the engine."""
    s = load_settings()
    print("loading bars ...")
    bars = load_bars(start=start, end=end)
    print(f"  {len(bars):,} bars / {bars.symbol.nunique():,} symbols")

    print("universe gate ...")
    metrics = bar_level_metrics(bars)
    tiered = classify_tier(metrics, settings=s)
    # `keep` 은 "한 세션이라도 거래 가능했던 종목"이라 게이트 통과 행에서 뽑는다.
    # `gate` 는 전 행을 들고 간다 -- cs_spread_20 은 체결가 모델이 쓰므로 티어를
    # 이탈한 세션의 청산에도 필요하고, tier 는 gated 플래그의 근거다.
    gated_rows = tiered.loc[tiered["tier"].isin(["core", "watch"])]
    keep = gated_rows["symbol"].unique().tolist()
    gate = tiered[["symbol", "dt", "tier", "cs_spread_20"]]
    print(f"  {len(keep):,} symbols with >=1 tradeable session")

    if max_symbols and len(keep) > max_symbols:
        rng = np.random.default_rng(seed)
        keep = sorted(rng.choice(keep, max_symbols, replace=False).tolist())
        print(f"  sampled to {len(keep):,}")

    bars = bars[bars.symbol.isin(keep)].copy()

    print("features ...")
    feats = build_features(bars)

    print("halt flags ...")
    bars = flag_halts(bars)

    cols = [
        "symbol", "dt", "atr14", "medvol_20", "addv_20", "rvol_5",
        "mfi_14", "ret_20", "sma20",
    ] + [f for w in SCORERS.values() for f, _, _ in w]
    cols = list(dict.fromkeys(c for c in cols if c in feats.columns))
    # LEFT join. 과거에는 inner join 이어서 게이트를 통과하지 못한 세션의 행이
    # 패널에서 사라졌고, 보유 포지션의 종목이 티어를 이탈하면 engine._bar() 가
    # None 을 돌려주어 그 포지션이 다시 평가되지 않았다. 슬롯을 영구 점유하고,
    # mtm 은 진입가로 고정되어 손실이 등가곡선에서 사라지고, trades 에도 잡히지
    # 않아 통계에서도 사라졌다. 종목의 89.8% 가 첫 통과일~마지막 통과일 사이에
    # 결손 구간을 갖고(최장 결손 중간값 112세션), 2020년 이후 25슬롯 중 평균
    # 24.9개가 그런 포지션으로 채워져 있었다.
    #
    # 게이트는 진입 필터다. 청산은 티어와 무관하게 평가되어야 한다.
    panel = bars.merge(feats[cols], on=["symbol", "dt"], how="left")
    panel = panel.merge(gate, on=["symbol", "dt"], how="left")
    panel["gated"] = panel["tier"].isin(["core", "watch"])
    panel = panel.sort_values(["symbol", "dt"])

    # sma20 now comes from the feature pipeline, where the live path reads it
    # too. It used to be rolled here, AFTER the inner join against `gate` --
    # which means the 20-period window walked only the rows that survived
    # gating. A symbol drops in and out of the core/watch tier day to day, so
    # that window was 20 scattered sessions, not the last 20. The thesis-break
    # exit fires off this number; expect its timing to shift versus the Phase 3
    # run, which has to be re-run rather than compared across the change.

    print(f"  engine panel: {len(panel):,} rows")
    return panel, s


def prepare_cached(start: date, end: date | None, max_symbols: int | None,
                   seed: int, refresh: bool = False):
    """`prepare` with the panel memoised to parquet.

    The feature step dominates the runtime -- 8.6M bars through
    compute_symbol_features takes tens of minutes -- and a parameter sweep
    re-derives the identical panel on every iteration. Caching it is what makes
    an experiment loop possible at all. The key covers every input that changes
    the panel; anything else (exits, sizing, costs) is applied by the engine
    afterwards and must NOT be part of the key.
    """
    key = (f"panel_{start}_{end or 'open'}_{max_symbols or 'all'}_{seed}"
           f"_v{FEATURE_PANEL_VERSION}.parquet")
    path = project_root() / "reports" / "cache" / key
    if path.exists() and not refresh:
        print(f"panel cache hit: {path.name}")
        return pd.read_parquet(path), load_settings()
    panel, s = prepare(start, end, max_symbols, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(path, index=False)
    print(f"panel cached: {path.name}")
    return panel, s


def make_signals(panel: pd.DataFrame, which: str, k: int) -> pd.DataFrame:
    """게이트 통과 행에서만 랭킹한다.

    패널이 left join 으로 바뀐 뒤 비게이트 행도 들어오므로, 그대로 스코어링하면
    교차섹션 백분위의 모집단이 달라져 랭킹 자체가 바뀐다. 진입 대상은 게이트
    통과 종목뿐이므로 여기서 걸러야 한다.
    """
    pool = panel[panel["gated"]] if "gated" in panel.columns else panel
    scored = score(pool, SCORERS[which])
    picks = top_k(scored, k=k)
    return picks[["dt", "symbol", "score"]]


def random_signals(panel: pd.DataFrame, k: int, seed: int) -> pd.DataFrame:
    """Shuffled control: random picks from the same universe on the same days."""
    rng = np.random.default_rng(seed)
    pairs = panel[["dt", "symbol"]]
    out = pairs.loc[
        pairs.groupby("dt", group_keys=False).apply(
            lambda g: g.sample(min(k, len(g)),
                               random_state=int(rng.integers(1e9))).index.to_series(),
            include_groups=False,
        )
    ].copy()
    out["score"] = 1.0
    return out[["dt", "symbol", "score"]]


def run_one(panel, signals, settings, equity_usd, cost_mult, enforce_halt=False):
    cost = CostParams.from_settings(settings)
    cost = CostParams(**{**cost.__dict__, "cost_multiplier": cost_mult})
    pp = PortfolioParams.from_settings(settings, initial_equity=equity_usd)
    pp = PortfolioParams(**{**pp.__dict__, "enforce_drawdown_halt": enforce_halt})
    bt = Backtester(
        bars=panel,
        signals=signals,
        cost=cost,
        entry=EntryParams.from_settings(settings),
        exits=ExitParams.from_settings(settings),
        portfolio=pp,
    )
    return bt.run()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--capital", type=float, default=10_000_000, help="KRW")
    ap.add_argument("--scorer", choices=list(SCORERS), default="evidence")
    ap.add_argument("--stress", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--max-symbols", type=int, default=None)
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--enforce-halt", action="store_true",
                    help="latch the -20%% kill switch (live behaviour); off for research")
    args = ap.parse_args()

    equity_usd = args.capital / KRW_PER_USD
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) if args.end else None

    panel, settings = prepare(start, end, args.max_symbols, args.seed)
    signals = make_signals(panel, args.scorer, args.top_k)
    print(f"  signals: {len(signals):,} ({signals.dt.nunique():,} sessions)")
    print()
    print(f"capital: KRW {args.capital:,.0f} = ${equity_usd:,.0f}   scorer: {args.scorer}")
    print("=" * 76)

    out_dir = project_root() / "reports"
    out_dir.mkdir(exist_ok=True)
    summary_rows = []

    for mult in args.stress:
        print()
        print(f"--- COST MULTIPLIER {mult:.1f}x ---")
        res = run_one(panel, signals, settings, equity_usd, mult, args.enforce_halt)
        rep = full_report(res.trades, res.equity)
        print(format_report(rep, res.trades, res.equity))
        halt_dt = res.params.get('first_drawdown_halt')
        if halt_dt:
            state = 'ENFORCED (trading stopped)' if args.enforce_halt else 'would have fired'
            print(f"{'-20% kill switch':<24}{str(halt_dt):>10}  {state}")

        row = {"cost_multiplier": mult, "variant": args.scorer}
        row.update({f"eq_{k}": v for k, v in rep["equity"].items()})
        row.update({f"tr_{k}": v for k, v in rep["trades"].items()})
        row.update({f"tail_{k}": v for k, v in rep["tails"].items()})
        summary_rows.append(row)

        if mult == 1.0:
            res.trades.to_csv(out_dir / f"trades_{args.scorer}.csv", index=False)
            res.equity.to_csv(out_dir / f"equity_{args.scorer}.csv", index=False)
            print()
            print("per-year:")
            print(yearly_breakdown(res.trades, res.equity).to_string(index=False))
            print()
            print("exit reasons:")
            print(exit_reason_breakdown(res.trades).to_string(index=False))
            print()
            print("trade return distribution:")
            print(return_distribution(res.trades).to_string(index=False))
            if not res.skipped.empty:
                print()
                print("entry skips:")
                print(res.skipped["reason"].value_counts().head(8).to_string())

    if args.control:
        print()
        print("=" * 76)
        print("SHUFFLED CONTROL (random top-15, same universe/days, same exits)")
        print("=" * 76)
        for mult in args.stress:
            ctrl_sig = random_signals(panel, args.top_k, args.seed)
            res = run_one(panel, ctrl_sig, settings, equity_usd, mult, args.enforce_halt)
            rep = full_report(res.trades, res.equity)
            print(f"\n--- control @ {mult:.1f}x ---")
            print(format_report(rep, res.trades, res.equity))
            row = {"cost_multiplier": mult, "variant": "control"}
            row.update({f"eq_{k}": v for k, v in rep["equity"].items()})
            row.update({f"tr_{k}": v for k, v in rep["trades"].items()})
            row.update({f"tail_{k}": v for k, v in rep["tails"].items()})
            summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "backtest_summary.csv", index=False)

    print()
    print("=" * 76)
    print("PHASE 3 GATE")
    print("=" * 76)
    strat = summary[summary.variant == args.scorer]
    ctrl = summary[summary.variant == "control"]
    for mult in args.stress:
        s_row = strat[strat.cost_multiplier == mult]
        if s_row.empty:
            continue
        s_cagr = s_row.iloc[0].get("eq_cagr", np.nan)
        line = f"  {mult:.1f}x costs: strategy CAGR {s_cagr*100:>7.2f}%"
        if not ctrl.empty:
            c_row = ctrl[ctrl.cost_multiplier == mult]
            if not c_row.empty:
                c_cagr = c_row.iloc[0].get("eq_cagr", np.nan)
                beats = s_cagr > c_cagr and s_cagr > 0
                line += (f"  | control {c_cagr*100:>7.2f}%  -> "
                         f"{'PASS' if beats else 'FAIL'}")
        print(line)
    print()
    print(f"written: {out_dir/'backtest_summary.csv'}")


if __name__ == "__main__":
    main()
