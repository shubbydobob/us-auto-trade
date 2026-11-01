# Exit loop — design

Status: **design only, not implemented.** Entry ships first; exits are manual
until this is built.

## What makes this hard here

Four constraints, all measured rather than assumed:

| constraint | value | consequence |
|---|---|---|
| No stop-market order | KIS overseas is `ORD_DVSN=00` limit only | A stop is a *monitored* level. Nothing protects the position while the process is down. |
| Overseas REST rate limit | ~1 req/s (domestic gets 20) | Polling prices for N positions costs N seconds per cycle. Useless as a stop mechanism. |
| Websocket symbol cap | 41 per account | Fine for ≤40 positions, but the cap is shared with entry monitoring. |
| Session hours in KST | 23:30–06:00 (EST) / 22:30–05:00 (EDT) | The loop runs overnight unattended and crosses KST midnight. DST shifts it twice a year. |

The first two together are the real problem: **the thing standing between a
position and a -40% gap is a Python process on a home PC.** Every design
decision below follows from taking that seriously rather than pretending the
backtest's instant fills exist.

## Architecture

```
job_intraday  (23:25 KST start, runs to session close)
│
├─ startup
│   ├─ load pos.positions           -> the exit plan (stop, frozen ATR, HWMs)
│   ├─ query KIS balance            -> authoritative quantities
│   ├─ reconcile()                  -> mismatch halts NEW entries, never exits
│   └─ place resting protective sells for every open position
│
├─ websocket HDFSCNT0  (<=40 symbols)
│   └─ trade prints -> synthesise 1-minute bars in memory
│
├─ every completed minute, per position
│   ├─ update_trackers(pos, bar)            [shared with backtest]
│   ├─ evaluate_exits(pos, bar, ...)        [shared with backtest]
│   └─ any order -> re-peg / submit -> persist
│
└─ shutdown (session close or crash)
    └─ resting sells remain live at the broker
```

### Resting protective sells — the core decision

A monitored stop evaluated by our own process is worthless when the process
dies, and this one runs overnight on a desktop. So **the moment an entry
fills, a resting limit sell is placed at the stop level and left at the
broker.** It survives a crash, a reboot and a power cut.

Trade-off, stated plainly: a resting limit sell at the stop will *not* fill in
a fast gap-down — the price trades through it. That is precisely the penalty
`backtest/costs.stop_fill_price` already models (fill at the open, plus a
penalty). The resting order is not a guarantee; it is strictly better than
nothing, and the backtest already assumes it performs badly.

The loop then *amends* that resting order as the trail rises: cancel, re-place
higher. At ~1 req/s and a handful of positions this is affordable, because the
trail only moves on new highs, not every minute.

### Why websocket, not polling

Polling 10 positions at 1 req/s means a 10-second blind spot per cycle, and
the rate limiter is shared with order submission — so placing an exit would
*delay* the next price check. The websocket pushes trade prints, costs no REST
quota, and the loop synthesises its own minute bars from them.

Fallback when the websocket drops: poll at reduced frequency, and log the
degradation loudly. A silent fallback to polling is how you discover afterwards
that stops were being evaluated every 30 seconds.

## Shared code is the invariant

`evaluate_exits`, `update_trackers`, `apply_fill` and `trailing_stop` are
imported from `strategy/exits.py` — the same objects `backtest/engine.py`
calls. The live loop supplies a `Bar` built from websocket prints instead of
from history; nothing else differs.

`tests/test_exits_shared.py` (to be written with the implementation) replays an
identical bar sequence through both paths and asserts identical exit orders.
Without that test the backtest numbers describe a different strategy than the
one running.

## Order lifecycle

```
submit limit -> poll status (inquire-ccnl) -> FILLED | PARTIAL | unfilled
                                                 │
unfilled after 3s ───────────────────────────────┘
  re-peg: cancel, re-place at ask * (1 + 0.004)
  max 3 re-pegs, cumulative drift capped at 1.5%
  beyond that: abandon (entry) or escalate to a wider limit (exit)
```

Entry and exit differ on abandonment. An unfilled **entry** is simply dropped —
there is always another candidate. An unfilled **exit** cannot be dropped, so
after the re-peg budget it widens progressively toward the bid. Getting out at
a bad price beats not getting out.

## Scheduling

Three Windows Task Scheduler entries, all taking an ET `--session-date` and
deriving their windows from `exchange-calendars`. No local wall-clock
arithmetic: US hours land at 23:30–06:00 KST under EST and 22:30–05:00 under
EDT, so the job spans KST midnight and shifts twice a year.

| job | ET | KST (EST) | does |
|---|---|---|---|
| `job_premarket` | 09:00 | 23:00 | refresh token, reconcile, arm entry plans |
| `job_intraday` | 09:30–16:00 | 23:30–06:00 | websocket loop, exits, entries |
| `job_eod_reconcile` | 16:30 | 06:30 | compare local state to KIS, snapshot equity |

## Safety

Reusing what already exists rather than inventing a second set of rules:

- `OrderGuard` caps notional and order count per session; callers cannot widen it.
- Reconciliation mismatch halts **new entries only**. Exits always run — refusing
  to exit because the books disagree is the worst possible failure.
- Kill switch via `data/KILL` and `sys.flags`; the file is the break-glass path
  for when the app itself is down.
- Data-quality breaker: if the morning ingest falls below
  `risk.min_universe_ingest_rate`, no new entries. Exits unaffected.

## Open questions to resolve during implementation

1. **Does 모의투자 support the overseas websocket?** Paper endpoint coverage is
   narrower than live. If not, `paper_broker.py` simulates fills against live
   quotes instead.
2. **Partial fills on 7 shares.** With a position this small a partial fill
   leaves 2-3 shares; confirm the TP ladder's `round()` does not emit a 0-share
   order.
3. **Does a resting sell block buying power?** If KIS reserves cash against an
   open sell, that is irrelevant for exits but matters once multiple positions
   are open.

## What the plumbing test should measure

The point of buying 7 shares of QTEX is not the P&L. It is these numbers, and
each one feeds back into the backtest:

| measurement | feeds |
|---|---|
| fill price vs the limit submitted | `costs.slippage_bps` calibration |
| time-to-fill, and how often a re-peg was needed | the re-peg budget |
| realised spread vs the Corwin-Schultz estimate | `costs.cs_spread_multiplier`, currently an uncalibrated 1.5 |
| whether a resting sell survives overnight | the whole resting-order premise |

If measured slippage exceeds 1.5x the model, the cost assumption is wrong and
the backtest has to be re-run before anything scales up.
