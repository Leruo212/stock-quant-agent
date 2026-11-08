"""Performance statistics. Same definitions as the thesis (eq. 3.3 - 3.6), plus Calmar
and exposure, which the thesis omits and which matter for judging a long-only book."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def annual_return(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    total = equity.iloc[-1] / equity.iloc[0]
    years = len(equity) / TRADING_DAYS
    if years <= 0 or total <= 0:
        return 0.0
    return float(total ** (1 / years) - 1)


def annual_vol(returns: pd.Series) -> float:
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe(returns: pd.Series, risk_free: float = 0.02) -> float:
    excess = returns - risk_free / TRADING_DAYS
    sd = excess.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(excess.mean() / sd * np.sqrt(TRADING_DAYS))


def sortino(returns: pd.Series, risk_free: float = 0.02) -> float:
    excess = returns - risk_free / TRADING_DAYS
    downside = excess[excess < 0]
    dd = downside.std(ddof=1)
    if dd == 0 or np.isnan(dd):
        return 0.0
    return float(excess.mean() / dd * np.sqrt(TRADING_DAYS))


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def calmar(equity: pd.Series) -> float:
    mdd = abs(max_drawdown(equity))
    return float(annual_return(equity) / mdd) if mdd > 0 else 0.0


def summarize(result: "BacktestResult", risk_free: float = 0.02) -> dict:
    ret = result.returns
    summary = {
        "cumulative_return": round(float(result.equity.iloc[-1] / result.equity.iloc[0] - 1), 4),
        "annual_return": round(annual_return(result.equity), 4),
        "annual_vol": round(annual_vol(ret), 4),
        "sharpe": round(sharpe(ret, risk_free), 3),
        "sortino": round(sortino(ret, risk_free), 3),
        "max_drawdown": round(max_drawdown(result.equity), 4),
        "calmar": round(calmar(result.equity), 3),
        "win_rate_daily": round(float((ret > 0).mean()), 4),
        "exposure": round(float((result.weights > 0).any(axis=1).mean()), 4),
        "avg_positions": round(float((result.weights > 0).sum(axis=1).mean()), 1),
        "turnover_per_year": round(float(result.turnover.sum() / max(len(ret) / TRADING_DAYS, 1e-9)), 2),
        "cost_drag": round(float(result.costs.sum()), 4),
        "n_days": int(len(ret)),
    }
    if result.benchmark_equity is not None and len(result.benchmark_equity) > 1:
        summary["benchmark_annual_return"] = round(annual_return(result.benchmark_equity), 4)
        summary["benchmark_max_drawdown"] = round(max_drawdown(result.benchmark_equity), 4)
        summary["excess_annual_return"] = round(
            summary["annual_return"] - summary["benchmark_annual_return"], 4
        )
    return summary
