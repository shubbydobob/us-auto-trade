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
