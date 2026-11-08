"""Unit tests for the pieces that are easy to get quietly wrong.

The bar-order and causal-alignment tests exist because a transposed OHLC column or a
label that peeks one bar ahead both produce results that look plausible and are wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantagent.backtest.engine import run_backtest
from quantagent.backtest.metrics import max_drawdown, sharpe, summarize
from quantagent.config import BacktestConfig, FeatureConfig
from quantagent.data.dataset import make_windows
from quantagent.data.features import add_features, build_panel, feature_columns
from quantagent.data.tencent import symbol


def _fake_bars(code: str, n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0004, 0.018, n)
    close = 10 * np.cumprod(1 + rets)
    high = close * (1 + abs(rng.normal(0, 0.01, n)))
    low = close * (1 - abs(rng.normal(0, 0.01, n)))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    dates = pd.bdate_range("2021-01-04", periods=n)
    return pd.DataFrame({
        "date": dates, "open": open_, "close": close, "high": np.maximum(high, np.maximum(open_, close)),
        "low": np.minimum(low, np.minimum(open_, close)), "volume": rng.integers(1e5, 1e6, n).astype(float),
        "amount": 0.0, "pct_chg": 0.0, "turnover": 0.0, "name": f"T{code}", "code": code,
    })


def test_symbol_prefix():
    assert symbol("600519") == "sh600519"
    assert symbol("000858") == "sz000858"
    assert symbol("300750") == "sz300750"
    assert symbol("000300", index=True) == "sh000300"


def test_feature_columns_count():
    cfg = FeatureConfig()
    assert len(feature_columns(cfg)) == 9
    assert len(feature_columns(FeatureConfig(feature_set="raw9"))) == 9


def test_label_looks_forward_exactly_horizon_bars():
    cfg = FeatureConfig(horizon=7)
    bars = _fake_bars("000001", n=50, seed=1)
    feat = add_features(bars, cfg)
    row = feat.iloc[10]
    expected = bars["close"].iloc[17] / bars["close"].iloc[10] - 1
    assert row["fwd_ret"] == pytest.approx(np.clip(expected, -cfg.clip, cfg.clip))


def test_panel_is_causal_split():
    bars = {f"00000{i}": _fake_bars(f"00000{i}", 400, seed=i) for i in range(3)}
    panel = build_panel(bars, FeatureConfig())
    data = make_windows(panel, FeatureConfig(), train_ratio=0.8)
    assert data.train_dates.max() < np.datetime64(data.split_date)
    assert data.x_train.ndim == 4 and data.x_train.shape[1] == 1
    assert data.x_train.shape[2] == 7 and data.x_train.shape[3] == 9
    assert np.isfinite(data.x_train).all()


def test_backtest_flat_market_stays_flat():
    """A flat market has zero momentum, so the book should stay in cash and not bleed."""
    n = 300
    dates = pd.bdate_range("2023-01-02", periods=n)
    bars = {}
    for i in range(3):
        frame = _fake_bars(f"00000{i}", n)
        frame["date"] = dates
        frame[["open", "close", "high", "low"]] = 10.0
        bars[f"00000{i}"] = frame
    result = run_backtest(bars, BacktestConfig(start="2023-01-02", end="2024-12-31"))
    assert result.equity.iloc[-1] == pytest.approx(1.0, abs=1e-9)


def test_metrics_on_known_series():
    equity = pd.Series([1.0, 1.1, 0.99, 1.2])
    assert max_drawdown(equity) == pytest.approx(0.99 / 1.1 - 1)
    flat = pd.Series([0.0] * 50)
    assert sharpe(flat) == 0.0


def test_summarize_keys():
    bars = {f"00000{i}": _fake_bars(f"00000{i}", 400, seed=i) for i in range(4)}
    result = run_backtest(bars, BacktestConfig(start="2021-06-01", end="2022-07-01"))
    stats = summarize(result)
    for key in ("annual_return", "sharpe", "max_drawdown", "win_rate_daily", "cost_drag"):
        assert key in stats
