"""Market snapshot construction for LLM prompting.

Everything here is computed from bars up to ``as_of`` only. That matters more than
usual: an LLM prompt that contains *any* post-``as_of`` number turns the whole
exercise into a leak. The snapshot is therefore built by slicing, never by indexing
backwards from the frame's last row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, period: int) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    value = (100 - 100 / (1 + rs)).iloc[-1]
    return float(value) if pd.notna(value) else 50.0


def market_snapshot(
    bars: pd.DataFrame,
    code: str = "",
    name: str = "",
    as_of: str | pd.Timestamp | None = None,
    benchmark: pd.Series | None = None,
) -> dict:
    """Compact, self-describing numeric snapshot ending at ``as_of``."""
    frame = bars.sort_values("date").copy()
    frame["date"] = pd.to_datetime(frame["date"])
    if as_of is not None:
        frame = frame[frame["date"] <= pd.Timestamp(as_of)]
    frame = frame.reset_index(drop=True)
    if len(frame) < 70:
        raise ValueError(f"need >=70 bars for a snapshot, got {len(frame)}")

    close = frame["close"].astype(float)
    last = float(close.iloc[-1])
    date = frame["date"].iloc[-1]

    def ret(days: int) -> float | None:
        if len(close) <= days:
            return None
        return round(float(close.iloc[-1] / close.iloc[-1 - days] - 1) * 100, 2)

    daily = close.pct_change().dropna()
    vol20 = float(daily.tail(20).std(ddof=1) * np.sqrt(252) * 100)

    ma = {}
    for window in (5, 10, 20, 60):
        if len(close) >= window:
            ma[f"ma{window}_dist_pct"] = round(float(last / close.tail(window).mean() - 1) * 100, 2)

    window_60 = close.tail(60)
    high60, low60 = float(window_60.max()), float(window_60.min())
    position = (last - low60) / (high60 - low60) if high60 > low60 else 0.5

    volume = frame["volume"].astype(float)
    vol_ratio = float(volume.tail(5).mean() / volume.tail(20).mean() - 1) if len(volume) >= 20 else 0.0

    prev_close = close.shift(1)
    tr = pd.concat(
        [(frame["high"] - frame["low"]), (frame["high"] - prev_close).abs(), (frame["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr_pct = float(tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean().iloc[-1] / last * 100)

    snapshot = {
        "code": str(code).zfill(6) if code else str(frame.get("code", pd.Series([""])).iloc[-1]),
        "name": name or str(frame.get("name", pd.Series([""])).iloc[-1] or ""),
        "as_of": date.strftime("%Y-%m-%d"),
        "close": round(last, 2),
        "ret_1d_pct": ret(1),
        "ret_5d_pct": ret(5),
        "ret_20d_pct": ret(20),
        "ret_60d_pct": ret(60),
        "vol_20d_annualised_pct": round(vol20, 1),
        "atr_14d_pct": round(atr_pct, 2),
        "rsi_6": round(_rsi(close, 6), 1),
        "rsi_14": round(_rsi(close, 14), 1),
        "volume_5d_vs_20d_pct": round(vol_ratio * 100, 1),
        "range_60d_position": round(float(position), 2),
        "distance_from_60d_high_pct": round((last / high60 - 1) * 100, 2),
        "distance_from_60d_low_pct": round((last / low60 - 1) * 100, 2),
    }
    snapshot.update(ma)

    if benchmark is not None:
        bench = benchmark.copy()
        bench.index = pd.to_datetime(bench.index)
        bench = bench[bench.index <= date]
        if len(bench) > 21:
            bench_ret20 = float(bench.iloc[-1] / bench.iloc[-21] - 1) * 100
            snapshot["benchmark_ret_20d_pct"] = round(bench_ret20, 2)
            if snapshot.get("ret_20d_pct") is not None:
                snapshot["relative_strength_20d_pct"] = round(snapshot["ret_20d_pct"] - bench_ret20, 2)

    return snapshot


def render_snapshot(snapshot: dict) -> str:
    """One key per line. Explicit labels beat prose: the model can cite them back."""
    lines = []
    for key, value in snapshot.items():
        if value is None:
            continue
        lines.append(f"{key}: {value}")
    return "\n".join(lines)
