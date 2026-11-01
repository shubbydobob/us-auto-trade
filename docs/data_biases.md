# Known data biases (v1, yfinance)

Every number this system produces before the Polygon migration carries the
biases below. They are listed with their measured magnitude where measurable and
their likely direction where not. Read any backtest result against this page.

## 1. Survivorship bias — present, direction: optimistic

yfinance returns nothing for delisted tickers. Reproduced directly:

```
yf.download(['SNTI','BTAI','ATNF'], period='1y')
  ATNF -> "possibly delisted; no price data found"   (not retrievable at all)
  BTAI -> recent window entirely NaN
```

This matters more here than for most strategies: a large share of penny-stock
runners explode and are then delisted or reverse-split into nothing. Those names
are exactly the population the strategy targets, and they are missing from the
sample. Backtest hit rates and returns are therefore biased **upward**, and the
left tail is under-represented.

**Measured universe churn:** of 4,729 KIS-tradeable common stocks in today's
directory, the 2019-01-01 backfill returned data for 4,685 (99.1%). That high
rate is itself the bias — it is the survival rate *conditional on still being
listed today*, not the survival rate of the 2019 universe.

### Why Wayback only partially helps

The plan proposed backfilling historical `nasdaqtraded.txt` snapshots from the
Internet Archive. Checking the CDX API for 2019-2026 returns **18 snapshots over
seven years**, unevenly spaced, with **2020-2022 almost entirely absent** — the
window containing the 2021 meme-stock squeeze, which is the single most
important regime for this strategy.

Wayback is therefore usable for coarsely identifying names that existed and
vanished, but it cannot produce a month-by-month point-in-time universe. The
real fix is a vendor with delisting history (Polygon.io), which is why
`PriceSource.symbol_meta(asof=...)` already carries an as-of parameter.

## 2. Split back-adjustment — fixed, with a regression test

Yahoo back-adjusts OHLC for splits **even when `auto_adjust=False`**. Verified
on SNTI, which did a 1:10 reverse split on 2024-07-18:

| date | Yahoo close | Yahoo volume | actually traded at |
|---|---|---|---|
| 2024-07-17 | $2.84 | 41,990 | **$0.284 / 419,900 shares** |
| 2024-07-18 | $2.55 | 90,300 | $2.55 / 90,300 (post-split, unchanged) |

Uncorrected, a stock below the $1.00 universe floor is admitted into the middle
of the core price band, and every volume feature is scaled by 10x. This single
bug is capable of manufacturing a plausible equity curve out of nothing.

**Mitigation:** `data/yf_source.reconstruct_as_traded` inverts the adjustment on
ingest and stores as-traded OHLCV plus `adj_factor`. Price band, dollar volume
and fill simulation use as-traded prices; technical features divide by
`adj_factor` to recover the continuous series. Dollar volume is invariant under
the transform, which `tests/test_adjustment.py` asserts, and
`tests/test_no_lookahead.py` AST-scans the source tree for `auto_adjust=True`.

## 3. Minute-bar history — hard limit, blocks intraday backtesting

yfinance serves roughly the last 7 days of 1-minute bars (verified: `period=5d,
interval=1m` returns 1,234 rows for SNTI; earlier windows return nothing).

**Consequence:** the intraday entry layer (VWAP reclaim / opening-range
breakout) **cannot be backtested over any meaningful history.** The multi-year
backtest uses the daily-bar entry proxy in `backtest/engine.py`; the actual
minute logic can only be validated forward, in paper trading. This is the
structural reason the 모의투자 stage cannot be skipped.

## 4. Point-in-time universe — forward-only from today

`dim.universe_snapshot` is written every session from now on, so the universe is
genuinely point-in-time going forward. Historical universe membership is
reconstructed from *today's* directory plus bar availability, which means a
company that changed exchange, ticker or listing status between 2019 and now is
classified by its current state. Direction of bias: unknown, probably small
relative to (1).

## 5. Fundamentals / float — not yet wired

SEC SIC codes, foreign-filer status and shares outstanding are modelled in
`ref.*` with `filed_date` for point-in-time joins, but are not yet populated.
Until they are, SPAC exclusion relies on name heuristics
(`universe/filters.py`) rather than SIC 6770, and there is no float or
short-interest feature.

## 6. Corwin-Schultz spread estimate — uncalibrated

`cs_spread_20` is estimated from daily high/low and is known to be noisy for
illiquid issues. `costs.cs_spread_multiplier` defaults to 1.5 and has **not**
been calibrated against real quoted spreads. Until it is, the cost model's
absolute level is an assumption, not a measurement — which is why the Phase 3
gate requires surviving a 1.5x cost stress rather than passing at 1.0x.

---

## What the biases mean for interpreting results

The Phase 2 lift numbers (see `reports/lift_summary.csv`) are computed on a
survivorship-biased sample. Their *relative* ordering across features is more
trustworthy than their absolute level, because the bias applies to all features
in the panel roughly equally. The absolute hit rates should be treated as an
upper bound.

Critically, the bias direction works **against** the original "quiet
accumulation" thesis: quiet names that exploded and then died are the ones most
likely to be missing. So the finding that compression does not predict
explosions is measured on a sample that, if anything, should have favoured it —
but a Polygon re-run remains the proper test.

---

# Phase 3 result: the gate FAILS

Full backtest, 2019-01-01 to present, 600-symbol sample, KRW 10,000,000
(~$7,246), top-15 daily signals, exits and sizing as specified.

| cost multiplier | strategy CAGR | shuffled control | gap |
|---|---|---|---|
| 1.0x | **-3.71%** | -5.30% | +1.6pp |
| 1.5x | **-3.84%** | -7.49% | +3.7pp |

The signal is real: it beats a random pick from the same universe on the same
days at every cost level, and the gap *widens* as costs rise because the signal
selects names that pay more per win (avg win 29.2% vs 16.9%). But both are
negative. The gate requires beating the control **and** being profitable, so
this is a FAIL.

## Why it loses

Hit rate 28.7% with a payoff ratio of 2.29. Break-even at that hit rate needs
2.48. The strategy is close but short, and the shortfall is roughly the size of
the transaction costs.

Exit attribution (1.0x costs, 579 trades):

| exit | n | avg return | total P&L |
|---|---|---|---|
| trail | 83 | +45.3% | **+$3,317** |
| thesis break | 305 | -11.1% | **-$2,908** |
| stop | 56 | -17.5% | -$1,693 |
| time | 118 | -3.4% | -$476 |

The winners work. There are simply too many small losers for them to carry.

The thesis-break exit was the first suspect, but the data clears it: those
trades have a median MFE of +2.8% and **none** ever reached +30%, so it is
cutting genuine duds at -11% rather than at the -20% hard stop. Removing it
would make things worse.

## Two things that were ruled out

**Portfolio saturation.** Exposure sits at 99.9% and 22,703 entries were
blocked by `max_concurrent`, so the book is permanently full and may never
reach the best signals. In-sample, tightening to the top 3 names per day turned
expectancy positive (+1.27%/trade at 1.5x costs) with a clean monotonic
improvement in hit rate and average win as K fell -- exactly the signature of a
ranking with real power.

**It did not survive out of sample.** K was searched on 2019-2026; re-tested on
the held-out 2024-2026 window the advantage disappeared:

| variant | CAGR (2024-2026, 1.5x) |
|---|---|
| evidence, top-3 | -6.17% |
| evidence, top-15 | -10.86% |
| thesis, top-3 | -14.65% |
| shuffled control, top-3 | -16.09% |

What *does* replicate out of sample is the ordering: evidence beats thesis beats
random, by 9-10 percentage points of CAGR. The ranking is genuine; no setting of
K makes it profitable.

This is recorded rather than tuned away. Searching K on the full sample and
adopting the winner is precisely the overfitting this project is meant to
detect, and the out-of-sample test caught it.

---

# Experiment B: liquidity band — negative, and a sizing bug found

## The bug (invalidated the first Phase 3 numbers)

The first Phase 3 run deployed only **12% of capital**: average position $94 on
a $7,246 account, median notional $28.68, minimum $0.52.

Cause was a config contradiction. `max_concurrent_positions: 10` at
`risk_per_trade: 1%` implies 10% portfolio heat, but `max_portfolio_heat` was
5%. The sizer resolved the conflict by *shrinking* each new position into
whatever heat remained, so positions six through ten were dust. The backtest was
measuring an uninvested portfolio.

Fixed two ways, both now covered by tests:
* `size_position` rejects when less than half a full-size risk budget remains,
  rather than shrinking to dust.
* `max_concurrent_positions` lowered to 5 so `n * risk <= heat` holds, with
  `test_risk_budget_settings_are_internally_consistent` guarding it.

## The experiment

Four price/liquidity bands fixed in advance, every one reported in both
windows so the result cannot be a search for the band that wins.

| band | IS CAGR | OOS CAGR | payoff IS→OOS | OOS maxDD |
|---|---|---|---|---|
| baseline $0.5-10 / 0.3M | -4.29% | -6.67% | 1.98 → 1.58 | -22.1% |
| B1 $3-10 / 5M | -2.83% | **-1.93%** | 1.52 → 1.35 | **-8.3%** |
| B2 $2-10 / 2M | -1.59% | -4.81% | 2.03 → 1.34 | -13.9% |
| B3 $5-20 / 10M | -2.35% | -4.89% | 1.11 → 1.35 | -13.5% |

Moving up the liquidity ladder cuts the drawdown hard (-22% → -8%) and halves
the loss, but the payoff ratio falls with it (1.98 → 1.52). Less is saved in
spread than is lost from the tail. **No band is profitable in either window.**

## The decisive test: zero transaction costs

If costs were the binding constraint, removing them entirely should produce a
profitable strategy. They are not.

| band | risk/trade | capital deployed | expectancy | CAGR |
|---|---|---|---|---|
| baseline | 1% | 8% | +8.19% | **+2.05%** |
| baseline | 3% | 13% | +2.02% | **-4.49%** |
| B1 | 1% | 10% | +0.27% | -0.15% |
| B1 | 3% | 27% | -1.94% | -2.37% |

With free trading the best case is about +2% a year on 8% deployed capital, and
**raising risk per trade makes it worse** — the losers scale with the winners,
which is what "no exploitable edge" looks like. A real edge gets better when
you lever it.

Note also the structural ceiling: risking 1% per trade with a ~22% stop gives a
4.5% position, so five slots cap deployment near 22% no matter how good the
signal is. Even a working signal could not move this account much at these
settings.

## Verdict

The signal has genuine *ranking* power — it beats a random pick from the same
universe on the same days in every window, at every cost level, by 2-12
percentage points of CAGR. It does not have *absolute* edge. Costs are not the
obstacle; the edge is simply too small to survive contact with any trading
system built on it.

---

# Universe filter bug: the ADR exclusion removed the target population

## What happened

`filter_listings` excluded any security whose name contained "Ordinary Shares",
on the stated rationale that a foreign issuer's volume is driven by its
home-market session and corrupts RVOL.

That rationale is correct for a genuine dual-listed ADR. It does **not** apply
to a Cayman or BVI holdco whose only listing is on NASDAQ -- such a company has
no home market. The filter removed 905 symbols on a reason that did not apply
to most of them.

Found by checking three names the user was watching spike on 2026-10-30:

| ticker | move | verdict |
|---|---|---|
| ULH $17.42 | +29.7% | correctly excluded: 60k median volume, and not a penny stock |
| GDYN $9.49 | +40.8% | passed the universe, failed the hard gate on `mfi_14` 86.3 vs 85 |
| SLGB $0.21 | +48.8% | excluded as "Class A Ordinary Shares" |

SLGB is the thesis working exactly as described: volume went 58,000 -> 39,334,800
shares (678x) on 2026-10-29 while the price moved -5.1%, then +48.8% the next
session. The system never saw it.

## Magnitude

| group | symbols | P(+50% in 20d) | P(+100% in 20d) |
|---|---|---|---|
| excluded "ordinary shares" | 436 | **24.59%** | **11.87%** |
| retained universe | 2,232 | 8.01% | 2.31% |

The excluded group explodes 3-5x more often than the universe every earlier
result was computed on.

## Are those moves tradeable?

The obvious objection is that these are pump-and-dumps that spike and collapse
in a day. Measured over all +100% episodes:

| | ordinary shares | rest |
|---|---|---|
| median peak gain | +303% | +241% |
| median days to peak | **14** | 15 |
| peaked on day 1 | 2% | 1% |
| median day-1 open gap | 0.0% | 0.0% |
| gain still available after the day-1 open | +283% | +217% |
| 5 days after the peak | **-26%** | -22% |

The moves run for about two weeks, so there is time to participate. The
objection is half right though: the give-back after the peak is sharp, and the
trailing stop captures only ~31% of the maximum excursion.

Caveat that cuts the other way: survivorship bias is **worse** for this group.
Chinese small-cap shells delist far more often, so the 4,827 episodes measured
come only from names still listed today. The +303% is optimistic.

## Effect on the result

Universe 4,729 -> 5,608 symbols. Re-running everything:

| | before fix | after fix |
|---|---|---|
| CAGR 2019-2026 @1.0x | -3.71% | **+0.47%** |
| CAGR 2019-2026 @1.5x | -3.84% | -0.45% |
| hit rate | 28.7% | **39.6%** |
| max drawdown | -26.1% | **-9.9%** |
| gap over shuffled control | +1.6pp | **+8.4pp** |

Split by window:

| window | signal @1.0x | signal @1.5x | control @1.0x |
|---|---|---|---|
| 2019-2023 (in-sample) | +1.51% | +0.92% | -6.40% |
| 2024-2026 (out-of-sample, 95 trades) | **-3.74%** | -3.98% | -12.44% |

The in-sample profit cannot be trusted: `EVIDENCE_WEIGHTS` were read off a lift
table computed over the whole period. The honest number is the out-of-sample
-3.74%, on a thin 95 trades.

Cost ceiling on the corrected universe:

| cost | expectancy/trade | CAGR | capital deployed |
|---|---|---|---|
| 0.0x | +4.68% | +3.17% | 7% |
| 1.0x | +1.43% | +0.47% | 7% |
| 1.5x | +0.02% | -0.45% | 8% |
| 2.0x | +0.89% | -0.93% | 5% |

Costs are now genuinely close to binding -- expectancy crosses zero right
around 1.5x, where it previously was negative even at 0x. Deployment remains
near 7% because risking 1% per trade behind a ~22% stop caps position size at
roughly 4.5% of equity.

**Status: still FAIL on the out-of-sample gate, but the failure is now narrow
rather than structural.** The strategy went from "no edge at any cost level" to
"breaks even around 1.0-1.5x costs in-sample, negative out-of-sample on a small
sample".
