"""Performance metrics.

Reporting the mean of a fat-tailed strategy is malpractice. The metrics that
decide anything here are the distribution ones:

* the full trade-return distribution, not its average
* P&L with the best five trades removed -- if that kills the strategy, the
  strategy is unproven rather than profitable
* per-year breakdown, because the edge concentrates in squeeze regimes
* block-bootstrap CI resampling MONTHS, not trades (trades are not independent)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _safe(x: float) -> float:
    return float(x) if np.isfinite(x) else float("nan")


def equity_metrics(equity: pd.DataFrame) -> dict:
    if equity.empty:
        return {}
    eq = equity["equity"].to_numpy(float)
    n = len(eq)
    years = n / TRADING_DAYS
    total_return = eq[-1] / eq[0] - 1
    cagr = (eq[-1] / eq[0]) ** (1 / years) - 1 if years > 0 and eq[0] > 0 else np.nan

    rets = pd.Series(eq).pct_change().dropna()
    vol = rets.std() * np.sqrt(TRADING_DAYS)
    downside = rets[rets < 0].std() * np.sqrt(TRADING_DAYS)
    sharpe = (rets.mean() * TRADING_DAYS) / vol if vol > 0 else np.nan
    sortino = (rets.mean() * TRADING_DAYS) / downside if downside > 0 else np.nan

    dd = equity["drawdown"].to_numpy(float)
    max_dd = dd.min() if len(dd) else np.nan

    # Longest stretch below the previous high-water mark.
    underwater, longest, cur = dd < -1e-9, 0, 0
    for flag in underwater:
        cur = cur + 1 if flag else 0
        longest = max(longest, cur)

    exposure = (equity["n_positions"] > 0).mean()
    return {
        "total_return": _safe(total_return),
        "cagr": _safe(cagr),
        "vol_annual": _safe(vol),
        "sharpe": _safe(sharpe),
        "sortino": _safe(sortino),
        "max_drawdown": _safe(max_dd),
        "max_dd_duration_days": int(longest),
        "calmar": _safe(cagr / abs(max_dd)) if max_dd and max_dd < 0 else np.nan,
        "exposure_pct_days": _safe(exposure),
        "avg_positions": _safe(equity["n_positions"].mean()),
        "final_equity": _safe(eq[-1]),
    }


def trade_metrics(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"n_trades": 0}
    r = trades["return_pct"]
    wins, losses = r[r > 0], r[r <= 0]
    gross_win, gross_loss = wins.sum(), abs(losses.sum())
    return {
        "n_trades": int(len(trades)),
        "hit_rate": _safe((r > 0).mean()),
        "avg_return": _safe(r.mean()),
        "median_return": _safe(r.median()),
        "avg_win": _safe(wins.mean()) if len(wins) else 0.0,
        "avg_loss": _safe(losses.mean()) if len(losses) else 0.0,
        "payoff_ratio": _safe(wins.mean() / abs(losses.mean()))
        if len(wins) and len(losses) and losses.mean() != 0
        else np.nan,
        "profit_factor": _safe(gross_win / gross_loss) if gross_loss > 0 else np.nan,
        "expectancy": _safe(r.mean()),
        "avg_sessions_held": _safe(trades["sessions_held"].mean()),
        "total_fees": _safe(trades["fees"].sum()),
        "net_pnl": _safe(trades["net_pnl"].sum()),
    }


def tail_dependence(trades: pd.DataFrame, top_n: int = 5) -> dict:
    """How much of the result comes from a handful of trades."""
    if trades.empty:
        return {}
    pnl = trades["net_pnl"].sort_values(ascending=False)
    total = pnl.sum()
    top = pnl.head(top_n).sum()
    return {
        "total_net_pnl": _safe(total),
        f"top{top_n}_pnl": _safe(top),
        f"top{top_n}_share": _safe(top / total) if total != 0 else np.nan,
        f"pnl_excl_top{top_n}": _safe(total - top),
        f"profitable_excl_top{top_n}": bool(total - top > 0),
        "top10_share": _safe(pnl.head(10).sum() / total) if total != 0 else np.nan,
    }


def return_distribution(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    r = trades["return_pct"]
    qs = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    return pd.DataFrame(
        {"quantile": qs, "return_pct": [_safe(r.quantile(q)) for q in qs]}
    )


def yearly_breakdown(trades: pd.DataFrame, equity: pd.DataFrame) -> pd.DataFrame:
    """Per-year results. Always lead with this, never with the aggregate."""
    if equity.empty:
        return pd.DataFrame()
    eq = equity.copy()
    eq["year"] = pd.to_datetime(eq["dt"]).dt.year
    rows = []
    for year, g in eq.groupby("year"):
        start, end = g["equity"].iloc[0], g["equity"].iloc[-1]
        yr_trades = (
            trades[pd.to_datetime(trades["exit_dt"]).dt.year == year]
            if not trades.empty
            else pd.DataFrame()
        )
        rows.append(
            {
                "year": int(year),
                "return": _safe(end / start - 1) if start else np.nan,
                "max_dd": _safe(g["drawdown"].min()),
                "n_trades": int(len(yr_trades)),
                "hit_rate": _safe((yr_trades["return_pct"] > 0).mean())
                if len(yr_trades)
                else np.nan,
                "best_trade": _safe(yr_trades["return_pct"].max()) if len(yr_trades) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def exit_reason_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    return (
        trades.groupby("exit_reason")
        .agg(n=("return_pct", "size"), avg_return=("return_pct", "mean"),
             total_pnl=("net_pnl", "sum"))
        .sort_values("total_pnl", ascending=False)
        .reset_index()
    )


def block_bootstrap_cagr(
    equity: pd.DataFrame, n_boot: int = 1000, seed: int = 7
) -> dict:
    """CAGR confidence interval, resampling calendar months.

    Resampling individual trades would assume independence that does not hold:
    one squeeze window produces many correlated winners.
    """
    if equity.empty or len(equity) < TRADING_DAYS // 2:
        return {}
    eq = equity.copy()
    eq["dt"] = pd.to_datetime(eq["dt"])
    eq["ret"] = eq["equity"].pct_change().fillna(0.0)
    months = [g["ret"].to_numpy() for _, g in eq.groupby(eq["dt"].dt.to_period("M"))]
    if len(months) < 6:
        return {}

    rng = np.random.default_rng(seed)
    n_months = len(months)
    years = len(eq) / TRADING_DAYS
    cagrs = []
    for _ in range(n_boot):
        picked = rng.integers(0, n_months, n_months)
        series = np.concatenate([months[i] for i in picked])
        growth = np.prod(1 + series)
        if growth <= 0:
            cagrs.append(-1.0)
            continue
        cagrs.append(growth ** (1 / years) - 1)
    arr = np.array(cagrs)
    return {
        "cagr_median": _safe(np.median(arr)),
        "cagr_p05": _safe(np.percentile(arr, 5)),
        "cagr_p95": _safe(np.percentile(arr, 95)),
        "prob_positive": _safe((arr > 0).mean()),
        "n_boot": n_boot,
    }


def full_report(trades: pd.DataFrame, equity: pd.DataFrame) -> dict:
    return {
        "equity": equity_metrics(equity),
        "trades": trade_metrics(trades),
        "tails": tail_dependence(trades),
        "bootstrap": block_bootstrap_cagr(equity),
    }


def format_report(report: dict, trades: pd.DataFrame, equity: pd.DataFrame) -> str:
    e, t, tl, b = (report.get(k, {}) for k in ("equity", "trades", "tails", "bootstrap"))
    L = []
    L.append(f"{'CAGR':<24}{e.get('cagr', float('nan'))*100:>10.2f}%")
    L.append(f"{'total return':<24}{e.get('total_return', float('nan'))*100:>10.2f}%")
    L.append(f"{'max drawdown':<24}{e.get('max_drawdown', float('nan'))*100:>10.2f}%")
    L.append(f"{'max DD duration (days)':<24}{e.get('max_dd_duration_days', 0):>10d}")
    L.append(f"{'Sharpe':<24}{e.get('sharpe', float('nan')):>10.2f}")
    L.append(f"{'Sortino':<24}{e.get('sortino', float('nan')):>10.2f}")
    L.append(f"{'exposure (% days)':<24}{e.get('exposure_pct_days', float('nan'))*100:>10.1f}%")
    L.append("")
    L.append(f"{'trades':<24}{t.get('n_trades', 0):>10d}")
    L.append(f"{'hit rate':<24}{t.get('hit_rate', float('nan'))*100:>10.1f}%")
    L.append(f"{'avg win':<24}{t.get('avg_win', float('nan'))*100:>10.2f}%")
    L.append(f"{'avg loss':<24}{t.get('avg_loss', float('nan'))*100:>10.2f}%")
    L.append(f"{'payoff ratio':<24}{t.get('payoff_ratio', float('nan')):>10.2f}")
    L.append(f"{'profit factor':<24}{t.get('profit_factor', float('nan')):>10.2f}")
    L.append(f"{'expectancy / trade':<24}{t.get('expectancy', float('nan'))*100:>10.2f}%")
    L.append(f"{'total fees paid':<24}${t.get('total_fees', 0):>9,.0f}")
    if tl:
        L.append("")
        L.append(f"{'top-5 share of P&L':<24}{tl.get('top5_share', float('nan'))*100:>10.1f}%")
        L.append(f"{'P&L excl. top 5':<24}${tl.get('pnl_excl_top5', 0):>9,.0f}")
        L.append(f"{'  still profitable?':<24}{str(tl.get('profitable_excl_top5')):>10}")
    if b:
        L.append("")
        L.append(
            f"{'CAGR 90% CI':<24}[{b['cagr_p05']*100:>6.1f}%, {b['cagr_p95']*100:>6.1f}%]"
            f"   P(CAGR>0) = {b['prob_positive']*100:.0f}%"
        )
    return "\n".join(L)
