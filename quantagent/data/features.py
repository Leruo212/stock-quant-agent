"""Feature engineering.

Two feature sets are provided:

``stationary9`` (default)
    Nine scale-free / stationary descriptors. Cross-sectional pooling across ~300
    names only works if the inputs share a unit, so everything here is a ratio or a
    return rather than a price level.

``raw9``
    Open / high / low / close / volume plus MA5-MA30 and raw RSI6, i.e. the
    composition of the original feature table. Kept so the
    ablation in ``docs/RESULTS.md`` can be reproduced.

The label is always the ``horizon``-day forward relative return of the close:
``(close[t+h] / close[t]) - 1``, winsorised to keep a handful of limit-up spikes from
dominating the MSE.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import FeatureConfig

STATIONARY_COLUMNS = [
    "ret1", "gap", "hl_range", "atr14", "vol_ratio", "ma5_ratio", "ma10_ratio", "ma30_ratio", "rsi6",
]
RAW_COLUMNS = ["open", "high", "low", "close", "logvol", "ma5", "ma10", "ma30", "rsi6"]


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def _atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = frame["close"].shift(1)
    tr = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def add_features(bars: pd.DataFrame, cfg: FeatureConfig | None = None) -> pd.DataFrame:
    """Attach the feature columns and the forward-return label to one symbol's bars."""
    cfg = cfg or FeatureConfig()
    df = bars.sort_values("date").reset_index(drop=True).copy()
    close = df["close"].astype(float)

    df["ret1"] = close.pct_change()
    df["gap"] = (df["open"] - close.shift(1)) / close.shift(1)
    df["hl_range"] = (df["high"] - df["low"]) / close
    df["atr14"] = _atr(df) / close
    df["vol_ratio"] = df["volume"] / df["volume"].rolling(5).mean() - 1.0
    df["ma5"] = close.rolling(5).mean()
    df["ma10"] = close.rolling(10).mean()
    df["ma30"] = close.rolling(30).mean()
    df["ma5_ratio"] = close / df["ma5"] - 1.0
    df["ma10_ratio"] = close / df["ma10"] - 1.0
    df["ma30_ratio"] = close / df["ma30"] - 1.0
    df["rsi6"] = _rsi(close, 6) / 100.0 - 0.5
    df["logvol"] = np.log1p(df["volume"])

    df["fwd_ret"] = close.shift(-cfg.horizon) / close - 1.0
    df["fwd_ret"] = df["fwd_ret"].clip(-cfg.clip, cfg.clip)
    df["label_up"] = (df["fwd_ret"] > 0).astype("float")
    return df


def feature_columns(cfg: FeatureConfig | None = None) -> list[str]:
    cfg = cfg or FeatureConfig()
    return STATIONARY_COLUMNS if cfg.feature_set == "stationary9" else RAW_COLUMNS


def build_panel(bars_by_code: dict[str, pd.DataFrame], cfg: FeatureConfig | None = None) -> pd.DataFrame:
    """Concatenate per-symbol feature frames into one long panel."""
    cfg = cfg or FeatureConfig()
    frames = []
    for code, bars in bars_by_code.items():
        feat = add_features(bars, cfg)
        feat["code"] = code
        frames.append(feat)
    if not frames:
        raise ValueError("empty panel")
    panel = pd.concat(frames, ignore_index=True)
    cols = feature_columns(cfg)
    panel = panel.dropna(subset=cols + ["fwd_ret"])
    return panel.sort_values(["date", "code"]).reset_index(drop=True)
