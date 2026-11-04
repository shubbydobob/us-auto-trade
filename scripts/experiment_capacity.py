"""Experiment F: the slot/holding-period interaction, with a sample-size gate.

Experiment E ended on a nominal PASS that is not one. Switching off both early
exits (thesis-break and the time stop) lifted payoff from 1.92 to 2.61 and
collapsed the trade count from 202 to 28, because five slots at ~4.5% each stay
occupied for years once nothing recycles them. The out-of-sample "+1.45%" rests
on 15 trail winners against 13 stop losers.

Worse, across every variant in that sweep the train ranking did not predict the
test ranking at all (E4 train -4.87% / test +4.86%; E7 train -3.92% / test
-9.21%; T1 train -0.80% / test +11.36%). At that sample size the knob space is
noise, and selecting on train CAGR selects luck.

So this script tests the one thing experiment C left open rather than turning
more knobs. C found that adding slots was monotonically worse -- but it tested
that with the OLD ladder and both early exits ON, where extra slots only admit
weaker signals. With nothing recycling capital the binding constraint moves to
slot count, and the two have never been varied together.

    holding rule  x  slot count     -- a 4x4 grid, not a chain of best-picks

Two guards against repeating experiment E's mistake:

* MIN_TRADES. A cell with fewer than this many train trades is reported but
  cannot be selected. 28 trades is not a measurement.
* Seed stability. The symbol sample is drawn three times; a cell whose sign
  flips across seeds is marked unstable. A result that depends on which 1,200
  of 2,657 symbols were drawn is not a result.

    PYTHONPATH=src python scripts/experiment_capacity.py --cost 1.5
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

from scripts.experiment_exits import run  # noqa: E402
from scripts.run_backtest import make_signals, prepare_cached, random_signals  # noqa: E402

TRAIN_END = pd.Timestamp("2023-12-01")
TEST_START = pd.Timestamp("2024-01-01")

# Below this many train trades a cell is unmeasurable, not promising.
MIN_TRADES = 80

# The ladderless base. Not selected here -- it is held fixed with the mechanical
# justification that a +60%/20-day label should not be sold a third of the way
# into the move.
LADDERLESS = dict(
    tp1_gain=9.99, tp1_fraction=0.0, tp2_gain=9.99, tp2_fraction=0.0,
    trail_pct_of_high=0.25, trail_pct_tight=0.18, trail_activate_gain=0.40,
)

# How capital gets recycled out of positions that are not working.
HOLDING = [
    ("H0 thesis+time (shipped)", {}),
    ("H1 time only", dict(thesis_enabled=False)),
    ("H2 thesis only", dict(time_stop_sessions=10_000)),
    ("H3 neither -- stop/trail only", dict(thesis_enabled=False,
                                           time_stop_sessions=10_000)),
]

# (label, slots, risk_per_trade, new_per_day). Heat is slots x risk so the
# sizer is never the thing throttling the book -- that is what made experiment
# C's wider books fill with dust.
SLOTS = [
    ("S5  x1.0%", 5, 0.010, 3),
    ("S10 x1.0%", 10, 0.010, 5),
    ("S15 x0.7%", 15, 0.007, 5),
    ("S25 x0.4%", 25, 0.004, 8),
]


def cell(panel_tr, sig_tr, panel_te, sig_te, s, equity, cost, ex_kw, pp_kw):
    tr = run(panel_tr, sig_tr, s, equity, cost, pp_kw=pp_kw, ex_kw=ex_kw)
    te = run(panel_te, sig_te, s, equity, cost, pp_kw=pp_kw, ex_kw=ex_kw)
    return tr, te


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost", type=float, default=1.5)
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--max-symbols", type=int, default=1200)
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 23])
    args = ap.parse_args()

    equity = 10_000_000 / 1380.0
    print(f"cost {args.cost}x  top-{args.top_k}  symbols {args.max_symbols}  "
          f"seeds {args.seeds}   min train trades to be selectable: {MIN_TRADES}")

    # seed -> (train panel, train signals, test panel, test signals, settings)
    data = {}
    for seed in args.seeds:
        full, s = prepare_cached(date(2019, 1, 1), None, args.max_symbols, seed)
        dts = pd.to_datetime(full["dt"])
        tr_p, te_p = full[dts < TRAIN_END], full[dts >= TEST_START]
        data[seed] = (tr_p, make_signals(tr_p, "evidence", args.top_k),
                      te_p, make_signals(te_p, "evidence", args.top_k), s)

    print()
    print(f"{'cell':<34}{'n tr':>6}{'hit':>7}{'payoff':>8}{'exp':>8}"
          f"{'CAGR tr':>9}{'CAGR te':>9}{'te by seed':>26}  flags")
    print("-" * 124)

    results = []
    for h_label, h_kw in HOLDING:
        for s_label, slots, risk, newday in SLOTS:
            ex_kw = {**LADDERLESS, **h_kw}
            pp_kw = dict(max_concurrent=slots, risk_per_trade=risk,
                         max_portfolio_heat=slots * risk, max_new_per_day=newday)

            per_seed_tr, per_seed_te, n_tr = [], [], []
            for seed, (tr_p, tr_s, te_p, te_s, st) in data.items():
                tr, te = cell(tr_p, tr_s, te_p, te_s, st, equity, args.cost,
                              ex_kw, pp_kw)
                per_seed_tr.append(tr["cagr"] if tr else float("nan"))
                per_seed_te.append(te["cagr"] if te else float("nan"))
                n_tr.append(tr["trades"] if tr else 0)
                if seed == args.seeds[0]:
                    first = (tr, te)

            tr, te = first
            mean_tr = pd.Series(per_seed_tr).mean()
            mean_te = pd.Series(per_seed_te).mean()
            median_n = int(pd.Series(n_tr).median())

            flags = []
            if median_n < MIN_TRADES:
                flags.append("THIN")
            signs = {s > 0 for s in per_seed_te if s == s}
            if len(signs) > 1:
                flags.append("UNSTABLE")

            by_seed = " ".join(f"{v * 100:+5.1f}" for v in per_seed_te)
            label = f"{h_label} / {s_label}"
            print(f"{label:<34}{median_n:>6}"
                  f"{(tr['hit'] * 100 if tr else 0):>6.1f}%"
                  f"{(tr['payoff'] if tr else 0):>8.2f}"
                  f"{(tr['expect'] * 100 if tr else 0):>7.1f}%"
                  f"{mean_tr * 100:>8.2f}%{mean_te * 100:>8.2f}%"
                  f"{by_seed:>26}  {','.join(flags)}")

            results.append({
                "label": label, "ex": ex_kw, "pp": pp_kw, "n": median_n,
                "cagr_tr": mean_tr, "cagr_te": mean_te,
                "thin": median_n < MIN_TRADES, "unstable": len(signs) > 1,
            })

    df = pd.DataFrame(results)
    # BOTH guards bind on selection, not just the trade floor. The first version
    # of this script excluded THIN but let an UNSTABLE cell win, and it picked
    # H3/S15 whose holdout ran +15.4 / +6.3 / -1.9 across seeds -- a sign flip
    # decided by which 1,200 symbols were drawn. That is the same mistake
    # experiment E made, one layer up.
    eligible = df[~df.thin & ~df.unstable]
    print()
    print(f"cells passing BOTH guards (n >= {MIN_TRADES}, seed-stable): "
          f"{len(eligible)} of {len(df)}   "
          f"[thin: {int(df.thin.sum())}, unstable: {int(df.unstable.sum())}]")
    if eligible.empty:
        print("No cell is measurable AND stable. The honest answer is that this "
              "grid cannot be decided at this sample size -- widen the universe "
              "or the period before choosing anything.")
        return

    ranked = eligible.sort_values("cagr_tr", ascending=False)
    print("eligible cells by TRAIN CAGR (selection basis):")
    for r in ranked.itertuples():
        print(f"  {r.label:<40} train {r.cagr_tr * 100:+6.2f}%   "
              f"test {r.cagr_te * 100:+6.2f}%   n~{r.n}")
    best = ranked.iloc[0]
    print(f"train-selected: {best.label}")

    print()
    print("=" * 124)
    print("FINAL -- that cell out of sample, per cost level, against the control")
    print("=" * 124)
    seed0 = args.seeds[0]
    _, _, te_p, te_s, st = data[seed0]
    for cm in (1.0, 1.5, 2.0):
        te = run(te_p, te_s, st, equity, cm, pp_kw=best.pp, ex_kw=best.ex)
        ct = run(te_p, random_signals(te_p, args.top_k, seed0), st, equity, cm,
                 pp_kw=best.pp, ex_kw=best.ex)
        if not te:
            print(f"  cost {cm:.1f}x : no trades")
            continue
        verdict = "PASS" if (te["cagr"] > 0 and ct and te["cagr"] > ct["cagr"]) else "FAIL"
        print(f"  cost {cm:.1f}x : CAGR {te['cagr'] * 100:>7.2f}%  "
              f"maxDD {te['maxdd'] * 100:>6.1f}%  n {te['trades']:>4}  "
              f"hit {te['hit'] * 100:>4.1f}%  payoff {te['payoff']:>4.2f}  | control "
              f"{ct['cagr'] * 100 if ct else float('nan'):>7.2f}%  -> {verdict}")
    print()
    print("A PASS here is still one cell out of 16 on one seed. It is a reason "
          "to run the full universe, not a reason to trade.")


if __name__ == "__main__":
    main()
