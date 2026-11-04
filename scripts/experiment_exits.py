"""Experiment E: where the losses actually come from, searched on train only.

The corrected sma20 (commit 51ede05) turned thesis-break into the dominant
exit: 103 of 199 trades at an average -12.2%, totalling -$1,922 against a trail
that made +$2,017. The gate failure -- hit rate 28.6% with payoff 1.90 where
break-even needs ~2.5 -- is therefore concentrated in ONE rule, and the earlier
analysis that cleared it was computed with the broken moving average.

Three questions, in dependency order:

1. THESIS   Is cutting at -12% better or worse than letting the stop decide?
            A rule that fires on half the book deserves its own answer.
2. TIME     41 trades at -14.7% exit on the 15-session clock. Is the clock
            cutting duds or interrupting slow winners?
3. BASIS    Entry is a limit at open x 1.01. With losses dominating, a cheaper
            basis moves every trade at once -- at the cost of unfilled signals.

Protocol, because this project already caught itself overfitting once (top-3
looked like +1.27%/trade in sample and -6.17% CAGR out of sample): every
variant is SELECTED on 2019-2023 and the 2024-2026 column is printed for all of
them so the degradation is visible rather than discovered later. The final
block re-runs the train-selected configuration out of sample against a shuffled
control.

    PYTHONPATH=src python scripts/experiment_exits.py --cost 1.5
    PYTHONPATH=src python scripts/experiment_exits.py --step thesis --quick
"""
from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_backtest import (  # noqa: E402
    make_signals,
    prepare_cached,
    random_signals,
)
from uat.backtest.costs import CostParams  # noqa: E402
from uat.backtest.engine import Backtester, EntryParams, PortfolioParams  # noqa: E402
from uat.backtest.metrics import full_report  # noqa: E402
from uat.strategy.exits import ExitParams  # noqa: E402

TRAIN_END = pd.Timestamp("2023-12-01")
TEST_START = pd.Timestamp("2024-01-01")

# ---------------------------------------------------------------- variants

# Experiment C (commit 01898e1) searched the ladder on train and picked this:
# no take-profit tranches, trail 25% engaging once peak gain passes +40%. Out
# of sample it ran -0.42% / -0.09% / +1.14% at 1.0/1.5/2.0x against a control
# of -10% to -18%. It was never written into config/base.yaml, so every gate
# run since -- including the two re-runs today -- measured the OLD ladder.
#
# The reason it works is mechanical rather than fitted: the signal is trained
# on a +60%/20-day barrier, and selling a third at +30% caps exactly the move
# being predicted. So this is the baseline the loss structure should be studied
# on top of, not the shipped default.
LADDER = [
    ("E0 shipped TP 30/80 trail25", {}),
    ("E4 no TP, trail25 from +40%", dict(
        tp1_gain=9.99, tp1_fraction=0.0, tp2_gain=9.99, tp2_fraction=0.0,
        trail_pct_of_high=0.25, trail_pct_tight=0.18, trail_activate_gain=0.40,
    )),
    ("E6 TP80 only, trail30", dict(
        tp1_gain=0.80, tp1_fraction=1 / 3, tp2_gain=9.99, tp2_fraction=0.0,
        trail_pct_of_high=0.30, trail_pct_tight=0.22,
    )),
    ("E7 no TP, trail30 from +30%", dict(
        tp1_gain=9.99, tp1_fraction=0.0, tp2_gain=9.99, tp2_fraction=0.0,
        trail_pct_of_high=0.30, trail_pct_tight=0.22, trail_activate_gain=0.30,
    )),
]

# T1 is the one that matters most: it removes the rule entirely, which is the
# only way to see what those 103 positions would have done. Everything between
# T2 and T7 is the same rule made harder to trigger.
THESIS = [
    ("T0 current (close<entry, rvol<.8)", {}),
    ("T1 OFF entirely", dict(thesis_enabled=False)),
    ("T2 buffer -7%", dict(thesis_loss_buffer=0.07)),
    ("T3 buffer -12%", dict(thesis_loss_buffer=0.12)),
    ("T4 rvol<0.6", dict(thesis_max_rvol=0.60)),
    ("T5 rvol<0.5 + buffer -7%", dict(thesis_max_rvol=0.50, thesis_loss_buffer=0.07)),
    ("T6 wait 5 sessions", dict(thesis_min_sessions=5)),
    ("T7 wait 10 + buffer -7%", dict(thesis_min_sessions=10, thesis_loss_buffer=0.07)),
]

TIME = [
    ("M0 current 15s / +12%", {}),
    ("M1 25 sessions", dict(time_stop_sessions=25)),
    ("M2 40 sessions", dict(time_stop_sessions=40)),
    ("M3 10 sessions", dict(time_stop_sessions=10)),
    ("M4 15s, only if losing", dict(time_stop_min_gain=0.0)),
    ("M5 OFF", dict(time_stop_sessions=10_000)),
]

# limit_over_open below 1.0 bids under the open: a better basis on everything
# that fills, and `limit_not_touched` on everything that gaps away.
BASIS = [
    ("A0 current open x1.01", {}),
    ("A1 at the open x1.00", dict(limit_over_open=1.00)),
    ("A2 pullback x0.99", dict(limit_over_open=0.99)),
    ("A3 pullback x0.98", dict(limit_over_open=0.98)),
    ("A4 pullback x0.96", dict(limit_over_open=0.96)),
    ("A5 chase x1.03", dict(limit_over_open=1.03)),
]


def run(panel, signals, settings, equity, cost_mult,
        pp_kw=None, ex_kw=None, en_kw=None):
    """One engine run. Returns the numbers that decide, plus the exit mix."""
    pp = replace(
        PortfolioParams.from_settings(settings, initial_equity=equity),
        enforce_drawdown_halt=False, **(pp_kw or {}),
    )
    ex = replace(ExitParams.from_settings(settings), **(ex_kw or {}))
    en = replace(EntryParams.from_settings(settings), **(en_kw or {}))
    cost = replace(CostParams.from_settings(settings), cost_multiplier=cost_mult)

    res = Backtester(panel, signals, cost, en, ex, pp).run()
    rep = full_report(res.trades, res.equity)
    t, e = rep["trades"], rep["equity"]
    if not t.get("n_trades"):
        return None

    mix = {}
    if not res.trades.empty:
        g = res.trades.groupby("exit_reason")
        mix = {k: (int(n), float(p)) for k, (n, p)
               in pd.DataFrame({"n": g.size(), "pnl": g.net_pnl.sum()}).iterrows()}
    return {
        "trades": t["n_trades"], "hit": t["hit_rate"], "payoff": t["payoff_ratio"],
        "expect": t["expectancy"], "cagr": e["cagr"], "maxdd": e["max_drawdown"],
        "mix": mix, "trades_df": res.trades,
    }


def header(title: str) -> None:
    print()
    print(title)
    print(f"{'variant':<34}{'trades':>7}{'hit':>7}{'payoff':>8}{'expect':>9}"
          f"{'CAGR tr':>9}{'CAGR te':>9}  exit mix (train)")
    print("-" * 118)


def row(label: str, tr, te) -> None:
    if tr is None:
        print(f"{label:<34}{'-- no trades on train --':>40}")
        return
    te_cagr = f"{te['cagr'] * 100:>8.2f}%" if te else "      n/a"
    mix = "  ".join(
        f"{k}:{n}/${p:,.0f}" for k, (n, p) in
        sorted(tr["mix"].items(), key=lambda kv: kv[1][1])
    )
    print(f"{label:<34}{tr['trades']:>7}{tr['hit'] * 100:>6.1f}%{tr['payoff']:>8.2f}"
          f"{tr['expect'] * 100:>8.2f}%{tr['cagr'] * 100:>8.2f}%{te_cagr}  {mix}")


def sweep(title, variants, panel_tr, sig_tr, panel_te, sig_te, s, equity, cost,
          *, kind, base_ex=None, base_en=None):
    """Run a block of variants, print it, return the train-best kwargs."""
    header(title)
    best, best_cagr = None, -9e9
    for label, kw in variants:
        ex = {**(base_ex or {}), **(kw if kind == "exit" else {})}
        en = {**(base_en or {}), **(kw if kind == "entry" else {})}
        tr = run(panel_tr, sig_tr, s, equity, cost, ex_kw=ex, en_kw=en)
        te = run(panel_te, sig_te, s, equity, cost, ex_kw=ex, en_kw=en)
        row(label, tr, te)
        if tr and tr["cagr"] > best_cagr:
            best, best_cagr = kw, tr["cagr"]
    print(f"  train-best: {best or 'baseline'}  (CAGR {best_cagr * 100:.2f}%)")
    return best or {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost", type=float, default=1.5,
                    help="cost multiplier; 1.5 is the gate level")
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--max-symbols", type=int, default=1200)
    ap.add_argument("--quick", action="store_true",
                    help="300 symbols -- shape only, not a decision")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--step", choices=["ladder", "thesis", "time", "basis", "all"],
                    default="all")
    args = ap.parse_args()

    max_symbols = 300 if args.quick else args.max_symbols
    equity = 10_000_000 / 1380.0

    full, s = prepare_cached(date(2019, 1, 1), None, max_symbols, args.seed)
    dts = pd.to_datetime(full["dt"])
    train, test = full[dts < TRAIN_END], full[dts >= TEST_START]
    sig_tr = make_signals(train, "evidence", args.top_k)
    sig_te = make_signals(test, "evidence", args.top_k)

    print(f"\ncost {args.cost}x  top-{args.top_k}  symbols {max_symbols}  "
          f"train {len(train):,} rows / test {len(test):,} rows")
    print("selection happens on TRAIN only; the test column is printed for "
          "every variant so degradation is visible, not discovered later.")

    ex_best: dict = {}
    en_best: dict = {}

    if args.step in ("ladder", "all"):
        ex_best = sweep(
            "STEP 0 -- LADDER (re-establish experiment C's result on this panel)",
            LADDER, train, sig_tr, test, sig_te, s, equity, args.cost, kind="exit",
        )

    if args.step in ("thesis", "all"):
        ex_best = {**ex_best, **sweep(
            "STEP 1 -- THESIS BREAK (on top of the train-best ladder)",
            THESIS, train, sig_tr, test, sig_te, s, equity, args.cost,
            kind="exit", base_ex=ex_best,
        )}

    if args.step in ("time", "all"):
        ex_best = {**ex_best, **sweep(
            "STEP 2 -- TIME STOP (on top of the train-best thesis setting)",
            TIME, train, sig_tr, test, sig_te, s, equity, args.cost,
            kind="exit", base_ex=ex_best,
        )}

    if args.step in ("basis", "all"):
        en_best = sweep(
            "STEP 3 -- ENTRY BASIS (on top of the train-best exits)",
            BASIS, train, sig_tr, test, sig_te, s, equity, args.cost,
            kind="entry", base_ex=ex_best,
        )

    print()
    print("=" * 118)
    print(f"FINAL -- train-selected config out of sample   exits={ex_best}  entry={en_best}")
    print("=" * 118)
    for cm in (1.0, 1.5, 2.0):
        te = run(test, sig_te, s, equity, cm, ex_kw=ex_best, en_kw=en_best)
        ct = run(test, random_signals(test, args.top_k, args.seed), s, equity, cm,
                 ex_kw=ex_best, en_kw=en_best)
        if not te:
            print(f"  cost {cm:.1f}x : no trades out of sample")
            continue
        verdict = "PASS" if (te["cagr"] > 0 and ct and te["cagr"] > ct["cagr"]) else "FAIL"
        print(f"  cost {cm:.1f}x : CAGR {te['cagr'] * 100:>7.2f}%  "
              f"maxDD {te['maxdd'] * 100:>6.1f}%  hit {te['hit'] * 100:>4.1f}%  "
              f"payoff {te['payoff']:>4.2f}  | control "
              f"{ct['cagr'] * 100 if ct else float('nan'):>7.2f}%  -> {verdict}")


if __name__ == "__main__":
    main()
