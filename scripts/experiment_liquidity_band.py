"""Experiment B: does the same signal work on more liquid, higher-priced names?

The Phase 3 signal selects $1.53 median price on $1.15M median daily dollar
volume, where the quoted spread is 2-3% and a one-cent tick is 65bps. The
hypothesis is that the edge is real but the execution venue is wrong.

Bands are fixed in advance and all of them are reported, in-sample AND
out-of-sample, so the result cannot be a search for the band that happens to
win. The Phase 3 top-K experiment already demonstrated how easily that fools
you: top-3 looked profitable in-sample and evaporated out of sample.
"""
from __future__ import annotations

import argparse
import warnings
from datetime import date

import pandas as pd

warnings.filterwarnings("ignore")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_backtest import make_signals, prepare, random_signals, run_one  # noqa: E402
from uat.backtest.metrics import full_report  # noqa: E402

# (label, min_price, max_price, min_addv) -- fixed before seeing any result.
BANDS = [
    ("baseline  $0.5-10 / 0.3M", 0.50, 10.0, 300_000),
    ("B1  $3-10   / 5M", 3.00, 10.0, 5_000_000),
    ("B2  $2-10   / 2M", 2.00, 10.0, 2_000_000),
    ("B3  $5-20   / 10M", 5.00, 20.0, 10_000_000),
]


def apply_band(panel: pd.DataFrame, lo: float, hi: float, min_addv: float) -> pd.DataFrame:
    """Restrict tradeable sessions by as-traded price and dollar volume.

    Rows outside the band are dropped entirely, so they can be neither scored
    nor held -- the same treatment the universe gate gives them.
    """
    keep = (
        panel["close"].between(lo, hi)
        & (panel["addv_20"] >= min_addv)
    )
    return panel[keep].copy()


def evaluate(panel, settings, equity, cost_mult, k, seed):
    rows = {}
    for label, lo, hi, min_addv in BANDS:
        sub = apply_band(panel, lo, hi, min_addv)
        if sub["dt"].nunique() < 60 or len(sub) < 2000:
            rows[label] = None
            continue

        sig = make_signals(sub, "evidence", k)
        res = run_one(sub, sig, settings, equity, cost_mult, False)
        rep = full_report(res.trades, res.equity)

        ctrl_sig = random_signals(sub, k, seed)
        cres = run_one(sub, ctrl_sig, settings, equity, cost_mult, False)
        crep = full_report(cres.trades, cres.equity)

        t, e = rep["trades"], rep["equity"]
        ct, ce = crep["trades"], crep["equity"]
        rows[label] = {
            "symbols": sub["symbol"].nunique(),
            "rows": len(sub),
            "med_price": sub["close"].median(),
            "med_addv": sub["addv_20"].median(),
            "n_trades": t.get("n_trades", 0),
            "hit": t.get("hit_rate", float("nan")),
            "avg_win": t.get("avg_win", float("nan")),
            "avg_loss": t.get("avg_loss", float("nan")),
            "payoff": t.get("payoff_ratio", float("nan")),
            "expect": t.get("expectancy", float("nan")),
            "cagr": e.get("cagr", float("nan")),
            "maxdd": e.get("max_drawdown", float("nan")),
            "ctrl_cagr": ce.get("cagr", float("nan")),
            "ctrl_trades": ct.get("n_trades", 0),
        }
    return rows


def show(title, rows):
    print()
    print(title)
    print(
        f"{'band':<26}{'syms':>6}{'px':>7}{'ADDV':>9}{'trades':>8}"
        f"{'hit':>7}{'payoff':>8}{'expect':>9}{'CAGR':>9}{'ctrl':>9}{'maxDD':>8}"
    )
    print("-" * 106)
    for label, r in rows.items():
        if r is None:
            print(f"{label:<26}{'-- too few rows after filtering --':>60}")
            continue
        print(
            f"{label:<26}{r['symbols']:>6}{r['med_price']:>7.2f}"
            f"{r['med_addv']/1e6:>8.1f}M{r['n_trades']:>8}"
            f"{r['hit']*100:>6.1f}%{r['payoff']:>8.2f}{r['expect']*100:>8.2f}%"
            f"{r['cagr']*100:>8.2f}%{r['ctrl_cagr']*100:>8.2f}%{r['maxdd']*100:>7.1f}%"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost", type=float, default=1.5)
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--max-symbols", type=int, default=1200)
    ap.add_argument("--capital", type=float, default=10_000_000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    equity = args.capital / 1380.0

    print(f"capital KRW {args.capital:,.0f} (${equity:,.0f})  "
          f"cost {args.cost}x  top-{args.top_k}")

    full, settings = prepare(date(2019, 1, 1), None, args.max_symbols, args.seed)

    train = full[pd.to_datetime(full["dt"]) < pd.Timestamp("2023-12-01")]
    test = full[pd.to_datetime(full["dt"]) >= pd.Timestamp("2024-01-01")]

    show("IN-SAMPLE  2019-01 .. 2023-11", evaluate(train, settings, equity,
                                                   args.cost, args.top_k, args.seed))
    show("OUT-OF-SAMPLE  2024-01 .. present", evaluate(test, settings, equity,
                                                       args.cost, args.top_k, args.seed))

    print()
    print("A band only counts if it is positive AND beats its control in BOTH windows.")


if __name__ == "__main__":
    main()
