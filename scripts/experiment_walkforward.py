"""Experiment G: fixed configs across rolling windows -- no selection at all.

Experiments E and F both ended on a nominal PASS, and both PASSes came from
picking a winner out of a grid on one train window and reading one holdout.
Experiment F's structural claim replicated across two symbol universes, but its
cell-level ranking did not: on the full universe H1/S15 has train -0.70% and
holdout +19.20%, so train rank would have rejected the best cell. And the
shuffled control on 2024-2026 came in at +0.02%, meaning that holdout is a
tailwind, not a hard test.

So this script removes selection entirely. Three FIXED configurations, decided
in advance and never re-chosen:

    C_SHIPPED   5 slots x 1.0%, thesis + time   -- what config/base.yaml does
    C_WIDE      25 slots x 0.4%, stop/trail only -- experiment F's H3/S25
    C_MID       25 slots x 0.4%, thesis off, time kept -- F's H1/S25

evaluated on consecutive, non-overlapping windows. The only question asked is
ordinal and the same in every window: does C_WIDE beat C_SHIPPED? A structural
claim that survives every period does not depend on which holdout was drawn; a
number that only appears in 2024-2026 does.

Nothing here is selected, so nothing here can be overfitted by this script. It
can only confirm or refute.

    PYTHONPATH=src python scripts/experiment_walkforward.py --cost 1.5
"""
from __future__ import annotations

import argparse
import sys
import warnings
from datetime import date
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.experiment_exits import run  # noqa: E402
from scripts.run_backtest import make_signals, prepare_cached, random_signals  # noqa: E402

LADDERLESS = dict(
    tp1_gain=9.99, tp1_fraction=0.0, tp2_gain=9.99, tp2_fraction=0.0,
    trail_pct_of_high=0.25, trail_pct_tight=0.18, trail_activate_gain=0.40,
)

# (label, exit kwargs, portfolio kwargs). Fixed in advance.
CONFIGS = [
    ("C_SHIPPED  5x1.0% thesis+time ladder",
     {},
     dict(max_concurrent=5, risk_per_trade=0.010,
          max_portfolio_heat=0.05, max_new_per_day=3)),
    ("C_MID     25x0.4% time only",
     {**LADDERLESS, "thesis_enabled": False},
     dict(max_concurrent=25, risk_per_trade=0.004,
          max_portfolio_heat=0.10, max_new_per_day=8)),
    ("C_WIDE    25x0.4% stop/trail only",
     {**LADDERLESS, "thesis_enabled": False, "time_stop_sessions": 10_000},
     dict(max_concurrent=25, risk_per_trade=0.004,
          max_portfolio_heat=0.10, max_new_per_day=8)),
]

# Consecutive, non-overlapping. 2019 is short because the feature warm-up eats
# the first ~250 sessions, so it is folded into the 2020-2021 window.
WINDOWS = [
    ("2020-2021", "2020-01-01", "2021-12-31"),
    ("2022-2023", "2022-01-01", "2023-12-31"),
    ("2024-2026", "2024-01-01", None),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost", type=float, default=1.5)
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--max-symbols", type=int, default=0,
                    help="0 = full universe (uses the cached full panel)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    equity = 10_000_000 / 1380.0
    full, s = prepare_cached(date(2019, 1, 1), None, args.max_symbols or None,
                             args.seed)
    dts = pd.to_datetime(full["dt"])

    print(f"\ncost {args.cost}x  top-{args.top_k}  "
          f"symbols {'all' if not args.max_symbols else args.max_symbols}")
    print("No configuration is selected here. Three fixed configs, every "
          "window, same ordinal question.")

    table: dict[str, dict[str, float]] = {}
    for wlabel, start, end in WINDOWS:
        mask = dts >= pd.Timestamp(start)
        if end:
            mask &= dts <= pd.Timestamp(end)
        panel = full[mask]
        sig = make_signals(panel, "evidence", args.top_k)
        ctl = random_signals(panel, args.top_k, args.seed)

        print()
        print(f"--- {wlabel}   ({len(panel):,} panel rows) ---")
        print(f"{'config':<40}{'n':>6}{'hit':>7}{'payoff':>8}{'exp':>8}{'CAGR':>9}"
              f"{'maxDD':>8}")
        row_ctl = run(panel, ctl, s, equity, args.cost,
                      pp_kw=CONFIGS[0][2], ex_kw=CONFIGS[0][1])
        for label, ex_kw, pp_kw in CONFIGS:
            r = run(panel, sig, s, equity, args.cost, pp_kw=pp_kw, ex_kw=ex_kw)
            if not r:
                print(f"{label:<40}{'-- no trades --':>30}")
                continue
            table.setdefault(label, {})[wlabel] = r["cagr"]
            print(f"{label:<40}{r['trades']:>6}{r['hit'] * 100:>6.1f}%"
                  f"{r['payoff']:>8.2f}{r['expect'] * 100:>7.1f}%"
                  f"{r['cagr'] * 100:>8.2f}%{r['maxdd'] * 100:>7.1f}%")
        if row_ctl:
            print(f"{'(shuffled control, shipped config)':<40}"
                  f"{row_ctl['trades']:>6}{row_ctl['hit'] * 100:>6.1f}%"
                  f"{row_ctl['payoff']:>8.2f}{row_ctl['expect'] * 100:>7.1f}%"
                  f"{row_ctl['cagr'] * 100:>8.2f}%{row_ctl['maxdd'] * 100:>7.1f}%")
            table.setdefault("CONTROL", {})[wlabel] = row_ctl["cagr"]

    print()
    print("=" * 96)
    print("ORDINAL SUMMARY -- CAGR by window, and how often each config wins")
    print("=" * 96)
    wins = {}
    shipped = CONFIGS[0][0]
    print(f"{'config':<40}" + "".join(f"{w:>14}" for w, _, _ in WINDOWS)
          + f"{'beats shipped':>16}")
    for label in list(dict.fromkeys([c[0] for c in CONFIGS] + ["CONTROL"])):
        if label not in table:
            continue
        cells = "".join(f"{table[label].get(w, float('nan')) * 100:>13.2f}%"
                        for w, _, _ in WINDOWS)
        n_win = sum(
            1 for w, _, _ in WINDOWS
            if label in table and shipped in table
            and table[label].get(w, -9e9) > table[shipped].get(w, 9e9)
        )
        wins[label] = n_win
        tag = "--" if label == shipped else f"{n_win} / {len(WINDOWS)}"
        print(f"{label:<40}{cells}{tag:>16}")

    print()
    for label, n in wins.items():
        if label in (shipped, "CONTROL"):
            continue
        if n == len(WINDOWS):
            print(f"{label}: beats the shipped config in EVERY window. The "
                  "structural claim does not depend on which holdout was drawn.")
        elif n == 0:
            print(f"{label}: never beats the shipped config. Refuted.")
        else:
            print(f"{label}: beats the shipped config in {n} of {len(WINDOWS)} "
                  "windows -- period-dependent, so not a structural result.")


if __name__ == "__main__":
    main()
