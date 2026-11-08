"""Signal quality evaluation.

A single MSE number hides the only question that matters for a trading model: *when
you sort stocks by the prediction, does the ordering carry information?* So this module
reports, on the held-out period only:

* pooled Pearson correlation between predicted and realised forward return
* direction accuracy (sign agreement)
* cross-sectional Spearman IC per trading date, then its mean / std / information ratio
  and the share of dates with positive IC
* decile portfolio spread — mean realised forward return of the top vs bottom decile

An IC of 0.02 with IR 0.3 is a weak but real signal. An IC of 0.00 with IR 0.05 is a
model that has learned nothing, and no amount of "78% accuracy" in a slide deck changes
that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def signal_report(pred: np.ndarray, target: np.ndarray, dates: np.ndarray, n_deciles: int = 10) -> dict:
    pred = np.asarray(pred, dtype=float).ravel()
    target = np.asarray(target, dtype=float).ravel()
    dates = pd.to_datetime(pd.Series(dates))

    frame = pd.DataFrame({"pred": pred, "target": target, "date": dates}).dropna()
    if frame.empty:
        return {"error": "no overlapping predictions"}

    # a collapsed model that emits a constant has no ordering to evaluate
    if float(frame["pred"].std()) < 1e-8:
        return {
            "n_observations": int(len(frame)),
            "n_dates": int(frame["date"].nunique()),
            "degenerate": True,
            "note": "predictions are constant - the model collapsed to the mean; no ranking exists",
            "baseline_direction": round(float((frame["target"] > 0).mean()), 4),
        }

    pooled_r = float(np.corrcoef(frame["pred"], frame["target"])[0, 1]) if len(frame) > 2 else float("nan")
    direction = float((np.sign(frame["pred"]) == np.sign(frame["target"])).mean())

    daily_ic = (
        frame.groupby("date")
        .apply(lambda g: _spearman(g["pred"].to_numpy(), g["target"].to_numpy()), include_groups=False)
        .dropna()
    )
    ic_mean = float(daily_ic.mean()) if len(daily_ic) else float("nan")
    ic_std = float(daily_ic.std(ddof=1)) if len(daily_ic) > 1 else float("nan")
    ir = float(ic_mean / ic_std * np.sqrt(len(daily_ic))) if ic_std and not np.isnan(ic_std) else float("nan")

    frame["decile"] = pd.qcut(frame["pred"].rank(method="first"), n_deciles, labels=False) + 1
    by_decile = frame.groupby("decile")["target"].mean()
    spread = float(by_decile.iloc[-1] - by_decile.iloc[0]) if len(by_decile) >= 2 else float("nan")

    return {
        "n_observations": int(len(frame)),
        "n_dates": int(frame["date"].nunique()),
        "pred_std": round(float(frame["pred"].std()), 8),
        "target_std": round(float(frame["target"].std()), 6),
        "r2_vs_mean": round(1 - float(((frame["pred"] - frame["target"]) ** 2).mean()) / float(frame["target"].var()), 4),
        "collapsed": bool(frame["pred"].std() < 1e-4),
        "pooled_pearson": round(pooled_r, 4),
        "direction_accuracy": round(direction, 4),
        "mean_daily_ic": round(ic_mean, 4) if not np.isnan(ic_mean) else None,
        "ic_std": round(ic_std, 4) if not np.isnan(ic_std) else None,
        "ic_ir_annualised": round(ir, 3) if not np.isnan(ir) else None,
        "ic_positive_share": round(float((daily_ic > 0).mean()), 3) if len(daily_ic) else None,
        "decile_spread": round(spread, 5) if not np.isnan(spread) else None,
        "decile_mean_return": {int(k): round(float(v), 5) for k, v in by_decile.items()},
        "baseline_direction": round(float((frame["target"] > 0).mean()), 4),
    }


def regime_split(pred: np.ndarray, target: np.ndarray, dates: np.ndarray, vol_window: int = 60) -> dict:
    """Does the signal survive in volatile regimes? Reported separately because a model
    that only works in calm markets is a model you cannot size."""
    frame = pd.DataFrame({"pred": np.asarray(pred).ravel(), "target": np.asarray(target).ravel(),
                          "date": pd.to_datetime(pd.Series(dates))}).dropna()
    daily = pd.DataFrame({
        "ic": frame.groupby("date")[["pred", "target"]].apply(
            lambda g: _spearman(g["pred"].to_numpy(), g["target"].to_numpy())
        ),
        "vol": frame.groupby("date")["target"].std(),
    }).dropna(subset=["ic"])
    vol_series = daily["vol"].rolling(vol_window, min_periods=20).mean()
    median = vol_series.expanding(min_periods=20).median()
    high = daily[(vol_series > median)]["ic"].dropna()
    low = daily[(vol_series <= median)]["ic"].dropna()
    return {
        "ic_high_vol": round(float(high.mean()), 4) if len(high) else None,
        "ic_low_vol": round(float(low.mean()), 4) if len(low) else None,
        "n_high_vol_days": int(len(high)),
        "n_low_vol_days": int(len(low)),
    }
