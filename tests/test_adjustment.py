"""The split back-adjustment trap.

Yahoo multiplies pre-reverse-split prices by the split ratio. SNTI did a 1:10
reverse split on 2024-07-18: Yahoo reports 2024-07-17 as $2.84 on 41,990 shares,
but the stock actually traded at $0.284 on 419,900 shares. Uncorrected, a name
below the $1.00 universe floor would be admitted right into the core band.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from uat.data.yf_source import reconstruct_as_traded, split_factor


def _frame(dates, close, volume):
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    return pd.DataFrame(
        {
            "Open": close,
            "High": [c * 1.05 for c in close],
            "Low": [c * 0.95 for c in close],
            "Close": close,
            "Volume": volume,
        },
        index=idx,
    )


def test_split_factor_applies_only_to_bars_before_the_split():
    idx = pd.DatetimeIndex(pd.to_datetime(["2024-07-17", "2024-07-18", "2024-07-19"]))
    splits = pd.Series([0.1], index=pd.DatetimeIndex(pd.to_datetime(["2024-07-18"])))
    f = split_factor(idx, splits)
    assert f.tolist() == [0.1, 1.0, 1.0]


def test_reverse_split_reconstruction_recovers_as_traded_prices():
    # Yahoo's displayed view of a 1:10 reverse split on 2024-07-18.
    df = _frame(["2024-07-17", "2024-07-18"], [2.84, 2.55], [41990, 90300])
    splits = pd.Series([0.1], index=pd.DatetimeIndex(pd.to_datetime(["2024-07-18"])))

    out = reconstruct_as_traded(df, splits)

    assert out["Close"].iloc[0] == pytest.approx(0.284)
    assert out["Volume"].iloc[0] == pytest.approx(419900)
    # The post-split bar is already as-traded and must be untouched.
    assert out["Close"].iloc[1] == pytest.approx(2.55)
    assert out["Volume"].iloc[1] == pytest.approx(90300)


def test_forward_split_reconstruction():
    # 4:1 forward split: Yahoo divides earlier prices by 4 and multiplies volume.
    df = _frame(["2022-07-21", "2022-07-22"], [36.0, 35.0], [4_000_000, 3_000_000])
    splits = pd.Series([4.0], index=pd.DatetimeIndex(pd.to_datetime(["2022-07-22"])))

    out = reconstruct_as_traded(df, splits)

    assert out["Close"].iloc[0] == pytest.approx(144.0)
    assert out["Volume"].iloc[0] == pytest.approx(1_000_000)


def test_dollar_volume_is_invariant_under_reconstruction():
    df = _frame(["2024-07-15", "2024-07-16", "2024-07-17"], [3.15, 3.10, 2.84],
                [42770, 69200, 41990])
    splits = pd.Series([0.1], index=pd.DatetimeIndex(pd.to_datetime(["2024-07-18"])))

    out = reconstruct_as_traded(df, splits)

    before = df["Close"] * df["Volume"]
    after = out["Close"] * out["Volume"]
    np.testing.assert_allclose(before.values, after.values, rtol=1e-9)


def test_no_splits_is_identity():
    df = _frame(["2026-01-05", "2026-01-06"], [1.50, 1.55], [100_000, 120_000])
    out = reconstruct_as_traded(df, pd.Series(dtype=float))

    np.testing.assert_allclose(out["Close"].values, df["Close"].values)
    np.testing.assert_allclose(out["Volume"].values, df["Volume"].values)
    assert (out["adj_factor"] == 1.0).all()


def test_adjusted_price_round_trips_back_to_the_continuous_series():
    """close / adj_factor must reproduce Yahoo's continuous (displayed) series,
    which is what the technical features consume."""
    df = _frame(["2024-07-17", "2024-07-18"], [2.84, 2.55], [41990, 90300])
    splits = pd.Series([0.1], index=pd.DatetimeIndex(pd.to_datetime(["2024-07-18"])))

    out = reconstruct_as_traded(df, splits)
    continuous = out["Close"] / out["adj_factor"]

    np.testing.assert_allclose(continuous.values, df["Close"].values, rtol=1e-9)
