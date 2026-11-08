"""Event-driven momentum backtester.

The rule from the thesis, section 3.3.4: every ``rebalance_days`` trading days, rank
the universe by the return over the *previous* window and hold the names with positive
momentum, equal weight, long only. Positions are cleared at the next rebalance when
momentum has turned negative.

Four things the original implementation left out and this one does not:

* the full ten-parameter signal surface from table 3-3 (trend filter, RSI ceiling,
  volatility ceiling, sentiment proxy, per-name weight cap) — a tuning report is only
  meaningful if every knob actually touches the equity curve;
* intraperiod stop-loss / take-profit, which is where most of the drawdown control
  comes from during a 7-day hold;
* transaction cost and slippage on realised turnover, so the reported return is one
  you could have earned rather than a frictionless upper bound;
* a benchmark curve on the same calendar, otherwise "12% annual" means nothing.

Signal timing: the signal computed on the close of ``t`` is executed at the close of
``t`` (no lag), matching the thesis. ``lag=1`` shifts execution one day forward.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import BacktestConfig


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series
    trades: pd.DataFrame
    benchmark_equity: pd.Series | None = None
    params: dict = field(default_factory=dict)


def price_matrix(bars_by_code: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for code, bars in bars_by_code.items():
        s = bars.set_index("date")["close"].astype(float)
        frames.append(s.rename(code))
    return pd.concat(frames, axis=1).sort_index()


def equal_weight_curve(bars_by_code: dict[str, pd.DataFrame], start: str, end: str) -> pd.Series:
    """Buy-and-hold, equal weight, rebalanced never.

    Included because the universe is the largest 300 names *as of today*, so any
    historical backtest over it is survivorship-biased. This curve measures the size of
    that bias: if equal-weight buy-and-hold of the same names already beats the index by
    a wide margin, most of the strategy's "alpha" is universe selection, not signal.
    """
    prices = price_matrix(bars_by_code).loc[start:end].ffill()
    prices = prices.dropna(how="all")
    if prices.empty:
        return pd.Series(dtype=float)
    base = prices.bfill().iloc[0]
    return (prices / base).mean(axis=1)


def _rsi_matrix(prices: pd.DataFrame, period: int) -> pd.DataFrame:
    delta = prices.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def run_backtest(
    bars_by_code: dict[str, pd.DataFrame],
    cfg: BacktestConfig | None = None,
    benchmark: pd.Series | None = None,
    lag: int = 0,
) -> BacktestResult:
    cfg = cfg or BacktestConfig()
    prices = price_matrix(bars_by_code).loc[cfg.start: cfg.end]
    prices = prices.dropna(how="all").ffill()
    if len(prices) < cfg.rebalance_days * 3:
        raise ValueError("not enough history in the requested window")
    prices = prices.loc[:, prices.notna().sum() > cfg.rebalance_days * 2]
    if prices.shape[1] == 0:
        raise ValueError("no tradable symbols in window")

    codes = list(prices.columns)
    rets = prices.pct_change().fillna(0.0)
    dates = prices.index
    n = len(dates)

    # --- precomputed signal layers (all causal: each row uses data up to that row) ---
    momentum = prices / prices.shift(cfg.rebalance_days) - 1.0
    trend_fast = prices.rolling(max(cfg.ma_short, 2)).mean()
    trend_slow = prices.rolling(max(cfg.ma_long, 3)).mean()
    rsi = _rsi_matrix(prices, cfg.rsi_period)
    realized_vol = rets.rolling(20).std()

    weights = pd.DataFrame(0.0, index=dates, columns=codes)
    turnover = pd.Series(0.0, index=dates)
    costs = pd.Series(0.0, index=dates)
    trades: list[dict] = []

    cost_rate = (cfg.cost_bps + cfg.slippage_bps) / 1e4
    current = pd.Series(0.0, index=codes)
    entry_price = pd.Series(np.nan, index=codes)
    entry_date: dict[str, pd.Timestamp] = {}

    for i in range(cfg.rebalance_days, n):
        date = dates[i]
        if (i - cfg.rebalance_days) % cfg.rebalance_days == 0:
            signal_idx = max(i - lag, cfg.rebalance_days)
            mom = momentum.iloc[signal_idx].dropna()

            if cfg.signal == "reversal":
                # buy the losers: the most negative trailing return first
                mom = mom[mom <= -max(abs(cfg.sentiment_threshold), 1e-4)]
                mom = mom.sort_values()
                if cfg.top_k > 0:
                    mom = mom.head(cfg.top_k)
                if len(mom):
                    mom = mom[realized_vol.iloc[signal_idx].reindex(mom.index) <= cfg.volatility_threshold]
                    mom = mom[rsi.iloc[signal_idx].reindex(mom.index) <= cfg.rsi_oversold]
            else:
                mom = mom[mom > 0]
                if cfg.top_k > 0:
                    mom = mom.sort_values(ascending=False).head(cfg.top_k)
                if len(mom):
                    mom = mom[mom >= cfg.sentiment_threshold]
                    mom = mom[realized_vol.iloc[signal_idx].reindex(mom.index) <= cfg.volatility_threshold]
                    mom = mom[rsi.iloc[signal_idx].reindex(mom.index) <= cfg.rsi_overbought]
                    above_fast = prices.iloc[signal_idx].reindex(mom.index) >= trend_fast.iloc[signal_idx].reindex(mom.index)
                    fast_above_slow = trend_fast.iloc[signal_idx].reindex(mom.index) >= trend_slow.iloc[signal_idx].reindex(mom.index)
                    mom = mom[above_fast.fillna(False) & fast_above_slow.fillna(False)]
            mom = mom.sort_values()

            target = pd.Series(0.0, index=codes)
            if len(mom):
                weight = min(1.0 / len(mom), cfg.position_size)
                target[mom.index] = weight
            traded = float((target - current).abs().sum())
            turnover.iloc[i] = traded
            costs.iloc[i] = traded * cost_rate

            for code in target[(target > 0) & (current == 0)].index:
                entry_price[code] = prices.at[date, code]
                entry_date[code] = date
            for code in current[(current > 0) & (target == 0)].index:
                ref = entry_price[code]
                if not np.isnan(ref):
                    trades.append({
                        "code": code,
                        "entry": entry_date.get(code),
                        "exit": date,
                        "pnl": float(prices.at[date, code] / ref - 1.0),
                        "reason": "rebalance",
                    })
                entry_price[code] = np.nan
                entry_date.pop(code, None)
            current = target

        if current.sum() > 0:
            for code in current[current > 0].index:
                ref = entry_price[code]
                if np.isnan(ref):
                    continue
                move = prices.at[date, code] / ref - 1.0
                if move <= -cfg.stop_loss or move >= cfg.take_profit:
                    trades.append({
                        "code": code,
                        "entry": entry_date.get(code),
                        "exit": date,
                        "pnl": float(move),
                        "reason": "stop_loss" if move < 0 else "take_profit",
                    })
                    current[code] = 0.0
                    entry_price[code] = np.nan
                    entry_date.pop(code, None)

        weights.iloc[i] = current.values

    # weights are set on the close of day i and earn the day i+1 return
    held = weights.shift(1).fillna(0.0)
    gross = (held * rets).sum(axis=1)
    net = gross - costs
    equity = (1.0 + net).cumprod()

    bench_equity = None
    if benchmark is not None:
        bench = benchmark.reindex(dates).ffill().dropna()
        if len(bench) > 1:
            bench_equity = bench / bench.iloc[0]

    return BacktestResult(
        equity=equity,
        returns=net,
        weights=weights,
        turnover=turnover,
        costs=costs,
        trades=pd.DataFrame(trades),
        benchmark_equity=bench_equity,
        params=cfg.__dict__.copy(),
    )
