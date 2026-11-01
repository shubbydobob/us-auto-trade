"""Experiment C: deployment and capture, searched on train only.

Two known leaks in the portfolio layer, neither of them about stock selection:

1. DEPLOYMENT -- only ~7% of capital is at work. Risking 1% per trade behind a
   ~22% stop gives a 4.5% position, so five slots cap deployment near 22% and
   fills fall short of even that.
2. CAPTURE -- the trail keeps only ~31% of maximum excursion. A name that runs
   +300% returns +90%.

Protocol, because the earlier top-K search looked great in-sample and died out
of sample: every variant is scored on 2019-2023 ONLY. The single best is then
run once on 2024-2026. The out-of-sample number is the only one that counts and
it is reported for every variant anyway, so the degradation is visible.
"""
from __future__ import annotations

import argparse
import warnings
from dataclasses import replace
from datetime import date

import pandas as pd

warnings.filterwarnings("ignore")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_backtest import make_signals, prepare, random_signals  # noqa: E402
from uat.backtest.costs import CostParams  # noqa: E402
from uat.backtest.engine import Backtester, EntryParams, PortfolioParams  # noqa: E402
from uat.backtest.metrics import full_report  # noqa: E402
from uat.strategy.exits import ExitParams  # noqa: E402

TRAIN_END = pd.Timestamp("2023-12-01")
TEST_START = pd.Timestamp("2024-01-01")

# (label, slots, risk_per_trade, new_entries_per_day)
# More slots means more lottery tickets, which suits a strategy where ~6% of
# trades carry the result. Bigger bets per name is the opposite wager.
DEPLOYMENT = [
    ("D0 current  5x1%", 5, 0.01, 3),
    ("D1 10 slots x1%", 10, 0.01, 4),
    ("D2 15 slots x1%", 15, 0.01, 5),
    ("D3  5 slots x2%", 5, 0.02, 3),
    ("D4 10 slots x2%", 10, 0.02, 4),
]

# (label, tp1_gain, tp1_frac, tp2_gain, tp2_frac, trail_pct, trail_tight)
# E1 removes the scale-out entirely: taking a third off at +30% caps the very
# trades that pay for everything else.
# trail_activate_gain is set explicitly on the no-ladder variants; without it
# the trail would never engage and "let winners run" would silently become
# "hold to the hard stop" (the bug that made E1-E3 identical).
CAPTURE = [
    ("E0 current 30/80 trail25", 0.30, 1 / 3, 0.80, 1 / 3, 0.25, 0.18, None),
    ("E1 no TP, trail 25 @+15%", 9.99, 0.0, 9.99, 0.0, 0.25, 0.18, 0.15),
    ("E2 no TP, trail 20 @+15%", 9.99, 0.0, 9.99, 0.0, 0.20, 0.15, 0.15),
    ("E3 no TP, trail 35 @+15%", 9.99, 0.0, 9.99, 0.0, 0.35, 0.25, 0.15),
    ("E4 no TP, trail 25 @+40%", 9.99, 0.0, 9.99, 0.0, 0.25, 0.18, 0.40),
    ("E5 TP 50/150 trail 30", 0.50, 1 / 3, 1.50, 1 / 3, 0.30, 0.22, None),
    ("E6 TP 80 only, trail 30", 0.80, 1 / 3, 9.99, 0.0, 0.30, 0.22, None),
]


def run(panel, signals, settings, equity, cost_mult, pp_kw=None, ex_kw=None):
    pp = PortfolioParams.from_settings(settings, initial_equity=equity)
    pp = replace(pp, enforce_drawdown_halt=False, **(pp_kw or {}))
    ex = replace(ExitParams.from_settings(settings), **(ex_kw or {}))
    cost = replace(CostParams.from_settings(settings), cost_multiplier=cost_mult)
    bt = Backtester(panel, signals, cost, EntryParams.from_settings(settings), ex, pp)
    res = bt.run()
    rep = full_report(res.trades, res.equity)
    t, e = rep["trades"], rep["equity"]
    if not t.get("n_trades"):
        return None
    deploy = res.equity.mtm.mean() / res.equity.equity.mean() * 100
    # Fraction of the run-up kept, averaged over trades that actually ran up.
    # Summing across all trades (the earlier version) let losers cancel winners
    # and produced meaningless near-zero values.
    ran = res.trades[res.trades.mfe >= 0.20]
    capture = (ran.return_pct / ran.mfe).mean() if len(ran) else float("nan")
    return {
        "trades": t["n_trades"], "hit": t["hit_rate"], "payoff": t["payoff_ratio"],
        "expect": t["expectancy"], "cagr": e["cagr"], "maxdd": e["max_drawdown"],
        "deploy": deploy, "capture": capture,
    }


def header(title):
    print()
    print(title)
    print(f"{'variant':<26}{'trades':>8}{'hit':>7}{'payoff':>8}{'deploy':>8}"
          f"{'capture':>9}{'CAGR tr':>9}{'CAGR te':>9}{'maxDD te':>10}")
    print("-" * 94)


def row(label, tr, te):
    if tr is None:
        print(f"{label:<26}{'-- no trades on train --':>40}")
        return
    te_cagr = f"{te['cagr']*100:>8.2f}%" if te else "      n/a"
    te_dd = f"{te['maxdd']*100:>9.1f}%" if te else "       n/a"
    print(f"{label:<26}{tr['trades']:>8}{tr['hit']*100:>6.1f}%{tr['payoff']:>8.2f}"
          f"{tr['deploy']:>7.0f}%{tr['capture']*100:>8.0f}%{tr['cagr']*100:>8.2f}%{te_cagr}{te_dd}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost", type=float, default=1.5)
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--max-symbols", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    equity = 10_000_000 / 1380.0
    full, s = prepare(date(2019, 1, 1), None, args.max_symbols, args.seed)
    dts = pd.to_datetime(full["dt"])
    train, test = full[dts < TRAIN_END], full[dts >= TEST_START]

    sig_tr = make_signals(train, "evidence", args.top_k)
    sig_te = make_signals(test, "evidence", args.top_k)

    print(f"\ncost {args.cost}x  top-{args.top_k}  "
          f"train {len(train):,} rows / test {len(test):,} rows")

    header("STEP 1 -- DEPLOYMENT (exits held at current settings)")
    best_dep, best_score = None, -9e9
    for label, slots, risk, newday in DEPLOYMENT:
        kw = dict(max_concurrent=slots, risk_per_trade=risk,
                  max_portfolio_heat=slots * risk, max_new_per_day=newday)
        tr = run(train, sig_tr, s, equity, args.cost, pp_kw=kw)
        te = run(test, sig_te, s, equity, args.cost, pp_kw=kw)
        row(label, tr, te)
        if tr and tr["cagr"] > best_score:
            best_dep, best_score = kw, tr["cagr"]
    print(f"  train-best deployment: {best_dep}")

    header("STEP 2 -- CAPTURE (using the train-best deployment above)")
    best_cap, best_score = None, -9e9
    for label, t1, f1, t2, f2, trail, tight, act in CAPTURE:
        kw = dict(tp1_gain=t1, tp1_fraction=f1, tp2_gain=t2, tp2_fraction=f2,
                  trail_pct_of_high=trail, trail_pct_tight=tight,
                  trail_activate_gain=act)
        tr = run(train, sig_tr, s, equity, args.cost, pp_kw=best_dep, ex_kw=kw)
        te = run(test, sig_te, s, equity, args.cost, pp_kw=best_dep, ex_kw=kw)
        row(label, tr, te)
        if tr and tr["cagr"] > best_score:
            best_cap, best_score = kw, tr["cagr"]
    print(f"  train-best capture: {best_cap}")

    print()
    print("=" * 94)
    print("FINAL -- train-selected configuration, evaluated out of sample")
    print("=" * 94)
    for cm in (1.0, 1.5, 2.0):
        te = run(test, sig_te, s, equity, cm, pp_kw=best_dep, ex_kw=best_cap)
        ct = run(test, random_signals(test, args.top_k, args.seed), s, equity, cm,
                 pp_kw=best_dep, ex_kw=best_cap)
        if not te:
            continue
        verdict = "PASS" if (te["cagr"] > 0 and ct and te["cagr"] > ct["cagr"]) else "FAIL"
        print(f"  cost {cm:.1f}x : CAGR {te['cagr']*100:>7.2f}%  "
              f"maxDD {te['maxdd']*100:>6.1f}%  hit {te['hit']*100:>4.1f}%  "
              f"capture {te['capture']*100:>3.0f}%  | control "
              f"{ct['cagr']*100 if ct else float('nan'):>7.2f}%  -> {verdict}")


if __name__ == "__main__":
    main()
