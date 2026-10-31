"""The future-deletion invariance test -- the cheapest defence in the repo.

Compute features on full history, delete every bar after T, recompute, and
assert the value at T is unchanged. This single test catches center=True,
shift(-n), bfill(), full-sample z-scores and adjusted-price leakage at once.

Also greps the source tree for auto_adjust=True, which would silently inject
future reverse-splits into historical prices (see data/yf_source.py).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from uat.features.pipeline import compute_symbol_features
from uat.labeling.triple_barrier import label_symbol

SRC = Path(__file__).resolve().parents[1] / "src"


def synthetic_bars(n: int = 400, seed: int = 11) -> pd.DataFrame:
    """A deterministic series with a quiet base and then a spike."""
    rng = np.random.default_rng(seed)
    # Segment lengths scale with n so the helper works for short frames too.
    n_base = max(1, int(n * 0.625))
    n_creep = max(1, int(n * 0.125))
    n_spike = max(1, int(n * 0.10))
    n_fade = max(1, n - n_base - n_creep - n_spike)
    base = np.concatenate(
        [
            np.full(n_base, 1.5) + rng.normal(0, 0.03, n_base),               # quiet base
            np.linspace(1.5, 2.2, n_creep) + rng.normal(0, 0.05, n_creep),    # creep
            np.linspace(2.2, 9.0, n_spike) + rng.normal(0, 0.3, n_spike),     # spike
            np.linspace(9.0, 4.0, n_fade) + rng.normal(0, 0.2, n_fade),       # collapse
        ]
    )[:n]
    close = np.maximum(base, 0.05)
    n = len(close)
    high = close * (1 + np.abs(rng.normal(0, 0.03, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.03, n)))
    open_ = (high + low) / 2
    volume = np.concatenate(
        [
            rng.integers(80_000, 150_000, n_base),
            rng.integers(200_000, 500_000, n_creep),
            rng.integers(2_000_000, 9_000_000, n_spike),
            rng.integers(400_000, 1_500_000, n_fade),
        ]
    )[:n]
    return pd.DataFrame(
        {
            "symbol": "TEST",
            "dt": pd.bdate_range("2023-01-02", periods=n).date,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "adj_factor": 1.0,
        }
    )


@pytest.mark.parametrize("cut", [300, 350, 380])
def test_features_are_invariant_to_deleting_the_future(cut: int):
    bars = synthetic_bars()

    full = compute_symbol_features(bars).set_index("dt")
    truncated = compute_symbol_features(bars.iloc[: cut + 1].reset_index(drop=True)).set_index("dt")

    t = bars["dt"].iloc[cut]
    numeric = [c for c in full.columns if c != "symbol"]

    leaked = []
    for col in numeric:
        a, b = full.loc[t, col], truncated.loc[t, col]
        if pd.isna(a) and pd.isna(b):
            continue
        if pd.isna(a) != pd.isna(b) or not np.isclose(float(a), float(b), rtol=1e-9, atol=1e-12):
            leaked.append((col, a, b))

    assert not leaked, f"features leak future information at T={t}: {leaked}"


def test_labels_use_only_forward_data_and_anchor_to_next_open():
    bars = synthetic_bars(n=100)
    labels = label_symbol(bars, horizon=5, upper=0.20, lower=-0.10)

    # Entry price must be the NEXT bar's open, never the current close.
    for i in range(len(bars) - 1):
        if labels["excluded"].iloc[i]:
            continue
        assert labels["entry_px"].iloc[i] == pytest.approx(bars["open"].iloc[i + 1])

    # The final bar has no forward window and must be excluded, not labelled.
    assert labels["excluded"].iloc[-1]
    assert pd.isna(labels["y_explode"].iloc[-1])


def test_label_respects_barrier_ordering():
    """Target before stop -> 1; stop before target -> 0; same bar -> 0."""
    def frame(highs, lows):
        n = len(highs)
        return pd.DataFrame(
            {
                "symbol": "X",
                "dt": pd.bdate_range("2026-01-05", periods=n).date,
                "open": [1.0] * n,
                "high": highs,
                "low": lows,
                "close": [1.0] * n,
                "volume": [1000] * n,
                "adj_factor": 1.0,
            }
        )

    up_first = label_symbol(frame([1, 1, 1.5, 0.5], [1, 1, 1.0, 0.5]), horizon=3,
                            upper=0.20, lower=-0.10)
    assert up_first["y_explode"].iloc[0] == 1.0

    down_first = label_symbol(frame([1, 1, 1.0, 1.5], [1, 1, 0.5, 1.0]), horizon=3,
                              upper=0.20, lower=-0.10)
    assert down_first["y_explode"].iloc[0] == 0.0

    # Both barriers on the same bar: daily data cannot resolve the order, so
    # assume the adverse one.
    ambiguous = label_symbol(frame([1, 1, 1.5, 1.0], [1, 1, 0.5, 1.0]), horizon=3,
                             upper=0.20, lower=-0.10)
    assert ambiguous["y_explode"].iloc[0] == 0.0
    assert ambiguous["exit_reason"].iloc[0] == "stop_ambiguous"


def test_gap_up_through_the_barrier_is_excluded_not_labelled_positive():
    bars = pd.DataFrame(
        {
            "symbol": "X",
            "dt": pd.bdate_range("2026-01-05", periods=3).date,
            "open": [1.0, 2.0, 2.0],   # T+1 opens +100%, above a +60% barrier
            "high": [1.0, 2.2, 2.2],
            "low": [1.0, 1.9, 1.9],
            "close": [1.0, 2.0, 2.0],
            "volume": [1000, 1000, 1000],
            "adj_factor": 1.0,
        }
    )
    labels = label_symbol(bars, horizon=2, upper=0.60, lower=-0.20)
    assert labels["excluded"].iloc[0]
    assert labels["exclude_reason"].iloc[0] == "gap_up_unachievable"
    assert pd.isna(labels["y_explode"].iloc[0])


def test_source_tree_never_enables_auto_adjust():
    """Parse the AST rather than grepping text.

    A plain substring search also fires on the docstrings that explain why
    auto_adjust must stay False, so match real keyword arguments only.
    """
    import ast

    offenders = []
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for kw in node.keywords:
                if (
                    kw.arg == "auto_adjust"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value is True
                ):
                    offenders.append(f"{path.relative_to(SRC)}:{node.lineno}")

    assert not offenders, (
        "auto_adjust=True injects future reverse-splits into historical prices "
        f"(see data/yf_source.py); found at {offenders}"
    )
