"""
Compute standard trading metrics from a portfolio value series.

All metrics are comparable across RL agents and baselines using the same interface.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(portfolio_history: list[float], trading_days_per_year: int = 252) -> dict:
    """
    Compute performance metrics from a portfolio value series.

    Args:
        portfolio_history: list of portfolio values at each timestep (including t=0)
        trading_days_per_year: 252 for traditional equities; BTC is 365 but
            we use 252 for comparability with equity papers including the source paper.

    Returns dict with:
        cumulative_return    — total % return
        annualized_return    — CAGR
        sharpe_ratio         — annualised Sharpe (rf=0)
        sortino_ratio        — annualised Sortino (rf=0)
        max_drawdown         — peak-to-trough as fraction
        calmar_ratio         — annualised_return / abs(max_drawdown)
        win_rate             — fraction of days with positive return
        final_value          — ending portfolio value
    """
    if len(portfolio_history) < 2:
        return {k: np.nan for k in [
            "cumulative_return", "annualized_return", "sharpe_ratio",
            "sortino_ratio", "max_drawdown", "calmar_ratio", "win_rate", "final_value",
        ]}

    pv = np.array(portfolio_history, dtype=float)
    daily_returns = np.diff(pv) / pv[:-1]
    n_days = len(daily_returns)

    cumulative_return = (pv[-1] / pv[0] - 1.0) * 100.0
    years = n_days / trading_days_per_year
    annualized_return = ((pv[-1] / pv[0]) ** (1.0 / max(years, 1e-9)) - 1.0) * 100.0

    mean_ret = np.mean(daily_returns)
    std_ret = np.std(daily_returns, ddof=1)
    sharpe = (mean_ret / std_ret * np.sqrt(trading_days_per_year)) if std_ret > 0 else np.nan

    neg = daily_returns[daily_returns < 0]
    downside_std = np.std(neg, ddof=1) if len(neg) > 1 else 0.0
    sortino = (mean_ret / downside_std * np.sqrt(trading_days_per_year)) if downside_std > 0 else np.nan

    running_max = np.maximum.accumulate(pv)
    drawdown = (pv - running_max) / running_max
    max_drawdown = float(drawdown.min())

    calmar = (annualized_return / 100.0) / abs(max_drawdown) if max_drawdown < 0 else np.nan
    win_rate = float(np.mean(daily_returns > 0))

    return {
        "cumulative_return": round(cumulative_return, 4),
        "annualized_return": round(annualized_return, 4),
        "sharpe_ratio": round(sharpe, 4) if not np.isnan(sharpe) else np.nan,
        "sortino_ratio": round(sortino, 4) if not np.isnan(sortino) else np.nan,
        "max_drawdown": round(max_drawdown * 100.0, 4),
        "calmar_ratio": round(calmar, 4) if not np.isnan(calmar) else np.nan,
        "win_rate": round(win_rate * 100.0, 2),
        "final_value": round(pv[-1], 2),
    }


def print_metrics(label: str, metrics: dict) -> None:
    print(f"\n{'─'*50}")
    print(f"  {label}")
    print(f"{'─'*50}")
    for k, v in metrics.items():
        print(f"  {k:<22} {v}")


if __name__ == "__main__":
    # smoke test with a synthetic series
    import random
    random.seed(42)
    pv = [10_000.0]
    for _ in range(365):
        pv.append(pv[-1] * (1 + random.gauss(0.001, 0.02)))
    m = compute_metrics(pv)
    print_metrics("Synthetic random walk", m)
