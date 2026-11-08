"""Turn the panel into (N, 1, W, F) tensors plus the matching forward-return labels.

Splitting is strictly chronological by date — never by row — so no observation in the
validation or test block can see a bar that came after it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import FeatureConfig
from .features import feature_columns


@dataclass
class WindowedData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    train_dates: np.ndarray
    val_dates: np.ndarray
    feature_names: list[str]
    split_date: pd.Timestamp
    mu: np.ndarray | None = None
    sigma: np.ndarray | None = None

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return self.x_train.shape[1:]

    def transform(self, raw: np.ndarray) -> np.ndarray:
        """Apply the exact train-time scaling to a raw ``(N, W, F)`` window batch."""
        if self.mu is None or self.sigma is None:
            raise ValueError("scaling statistics missing")
        scale = self.mu.reshape(1, 1, -1) if self.mu.ndim == 1 else self.mu
        shift = self.sigma.reshape(1, 1, -1) if self.sigma.ndim == 1 else self.sigma
        scaled = (raw - scale) / (shift + 1e-6)
        return np.clip(scaled, -6.0, 6.0)[:, None, :, :].astype(np.float32)


def make_windows(
    panel: pd.DataFrame,
    cfg: FeatureConfig | None = None,
    train_ratio: float = 0.8,
    max_per_symbol: int = 0,
    seed: int = 42,
) -> WindowedData:
    cfg = cfg or FeatureConfig()
    cols = feature_columns(cfg)
    window, horizon = cfg.window, cfg.horizon

    dates = np.sort(panel["date"].unique())
    split_idx = int(len(dates) * train_ratio)
    split_date = pd.Timestamp(dates[split_idx])

    xs, ys, ds = [], [], []
    rng = np.random.default_rng(seed)

    for code, grp in panel.groupby("code", sort=True):
        grp = grp.sort_values("date").reset_index(drop=True)
        values = grp[cols].to_numpy(dtype=np.float32)
        labels = grp["fwd_ret"].to_numpy(dtype=np.float32)
        stamps = grp["date"].to_numpy()
        if len(grp) < window + 1:
            continue
        # window i covers rows [i, i+W) and predicts the label attached to row i+W-1
        windows = np.stack([values[i: i + window] for i in range(len(grp) - window + 1)])
        idx = np.arange(len(windows)) + window - 1
        lab = labels[idx]
        stamp = stamps[idx]
        keep = np.isfinite(lab)
        windows, lab, stamp = windows[keep], lab[keep], stamp[keep]
        if len(windows) == 0:
            continue
        if max_per_symbol and len(windows) > max_per_symbol:
            pick = rng.choice(len(windows), max_per_symbol, replace=False)
            pick.sort()
            windows, lab, stamp = windows[pick], lab[pick], stamp[pick]
        xs.append(windows)
        ys.append(lab)
        ds.append(stamp)

    if not xs:
        raise ValueError("no windows produced - check window/horizon against panel length")

    x = np.concatenate(xs)
    y = np.concatenate(ys)
    d = np.concatenate(ds)

    if cfg.label_mode == "rank":
        # Cross-sectional percentile per date, centred on zero. Predicting the *ordering*
        # instead of the raw return throws away the market-wide component (which is
        # unforecastable from single-name technicals) and keeps the part a stock picker
        # can actually use. Measured effect in docs/RESULTS.md.
        y = (
            pd.DataFrame({"y": y, "d": d})
            .groupby("d")["y"]
            .rank(pct=True)
            .to_numpy()
            - 0.5
        )

    # z-score with train statistics only: features are already scale-free but a few
    # ratios (vol_ratio) have heavy tails that would otherwise dominate.
    train_mask = d < np.datetime64(split_date)
    mu = x[train_mask].mean(axis=(0, 1), keepdims=True)
    sigma = x[train_mask].std(axis=(0, 1), keepdims=True) + 1e-6
    x = (x - mu) / sigma
    x = np.clip(x, -6.0, 6.0)

    x = x[:, None, :, :].astype(np.float32)

    return WindowedData(
        x_train=x[train_mask],
        y_train=y[train_mask].reshape(-1, 1),
        x_val=x[~train_mask],
        y_val=y[~train_mask].reshape(-1, 1),
        train_dates=d[train_mask],
        val_dates=d[~train_mask],
        feature_names=cols,
        split_date=split_date,
        mu=mu,
        sigma=sigma,
    )
