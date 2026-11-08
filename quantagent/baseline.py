"""Ridge baseline on the same windows.

Purpose: separate "the network failed" from "there is nothing to learn". If a linear
model on the identical features gets the same near-zero IC, the bottleneck is the
information content of the features at this horizon, not the capacity of the network.

Closed-form solution, no sklearn dependency:

    w = (XᵀX + λI)⁻¹ Xᵀy
"""

from __future__ import annotations

import numpy as np

from .evaluate import signal_report


def fit_predict_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    alpha: float = 10.0,
) -> np.ndarray:
    """Flatten (N, 1, W, F) windows and fit a ridge regression."""
    xt = x_train.reshape(len(x_train), -1).astype(np.float64)
    xv = x_val.reshape(len(x_val), -1).astype(np.float64)
    yt = y_train.ravel().astype(np.float64)

    # standardise columns; the design matrix is otherwise poorly conditioned
    mu = xt.mean(axis=0)
    sd = xt.std(axis=0) + 1e-8
    xt = (xt - mu) / sd
    xv = (xv - mu) / sd

    xt = np.hstack([xt, np.ones((len(xt), 1))])
    xv = np.hstack([xv, np.ones((len(xv), 1))])
    gram = xt.T @ xt + alpha * np.eye(xt.shape[1])
    weights = np.linalg.solve(gram, xt.T @ yt)
    return xv @ weights


def ridge_report(x_train, y_train, x_val, y_val, dates, alpha: float = 10.0) -> dict:
    pred = fit_predict_ridge(x_train, y_train, x_val, alpha=alpha)
    report = signal_report(pred, y_val.ravel(), dates)
    report["alpha"] = alpha
    return report
