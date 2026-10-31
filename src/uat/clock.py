"""NYSE session calendar and DST-safe KST<->ET conversion.

US market hours land at 23:30-06:00 KST under EST and 22:30-05:00 KST under EDT,
so the intraday job spans KST midnight and the shift moves twice a year. Every
job therefore takes an ET session_date and derives its windows from the exchange
calendar. Never do local wall-clock arithmetic to find a session boundary.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd

ET = ZoneInfo("America/New_York")
KST = ZoneInfo("Asia/Seoul")


@lru_cache(maxsize=1)
def _cal():
    return xcals.get_calendar("XNYS")


def is_session(d: date) -> bool:
    return _cal().is_session(pd.Timestamp(d))


def sessions(start: date, end: date) -> list[date]:
    idx = _cal().sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))
    return [ts.date() for ts in idx]


def previous_session(d: date) -> date:
    return _cal().previous_session(pd.Timestamp(d)).date()


def next_session(d: date) -> date:
    return _cal().next_session(pd.Timestamp(d)).date()


def shift_sessions(d: date, n: int) -> date:
    """Move n sessions forward (n>0) or backward (n<0) from a session date."""
    fn = next_session if n > 0 else previous_session
    for _ in range(abs(n)):
        d = fn(d)
    return d


def session_open_close(d: date) -> tuple[datetime, datetime]:
    """Session open/close as ET-aware datetimes, honouring early closes."""
    cal = _cal()
    ts = pd.Timestamp(d)
    return (
        cal.session_open(ts).tz_convert(ET).to_pydatetime(),
        cal.session_close(ts).tz_convert(ET).to_pydatetime(),
    )


def is_early_close(d: date) -> bool:
    _, close = session_open_close(d)
    return close.time() < time(16, 0)


def et_time_on(d: date, hhmm: str) -> datetime:
    """Build an ET-aware datetime for 'HH:MM' on a session date."""
    h, m = (int(x) for x in hhmm.split(":"))
    return datetime.combine(d, time(h, m), tzinfo=ET)


def to_kst(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        raise ValueError("naive datetime: attach ET before converting to KST")
    return dt.astimezone(KST)


def current_session_date(now: datetime | None = None) -> date:
    """The ET session date in effect right now.

    Before the open, this is still the previous session -- the EOD scan at 17:30 ET
    and the intraday loop that starts the next morning must agree on which date
    they mean, and local KST date would disagree with ET for most of the session.
    """
    now = (now or datetime.now(ET)).astimezone(ET)
    d = now.date()
    if is_session(d) and now.time() >= time(9, 30):
        return d
    return previous_session(d) if is_session(d) else previous_session(d + timedelta(days=1))
