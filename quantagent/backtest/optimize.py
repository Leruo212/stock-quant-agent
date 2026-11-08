"""Optuna search over the ten parameters that make up the signal surface.

Objective is the Sharpe ratio on the in-sample window, with a hard penalty on
drawdowns beyond 25% so the search cannot buy return with leverage-like tail risk.
Parameter importance is exported because "which knob mattered" is the only part of a
tuning report that transfers to the next dataset.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from ..config import BacktestConfig
from .engine import run_backtest
from .metrics import summarize

SEARCH_SPACE = {
    "position_size": ("float", 0.05, 0.30),
    "stop_loss": ("float", 0.02, 0.10),
    "take_profit": ("float", 0.05, 0.20),
    "ma_short": ("int", 5, 20),
    "ma_long": ("int", 20, 100),
    "rsi_period": ("int", 10, 30),
    "rsi_oversold": ("int", 20, 40),
    "rsi_overbought": ("int", 60, 80),
    "volatility_threshold": ("float", 0.01, 0.05),
    "sentiment_threshold": ("float", -0.5, 0.5),
}


def _suggest(trial, name: str):
    kind, low, high = SEARCH_SPACE[name]
    if kind == "int":
        return trial.suggest_int(name, low, high)
    return trial.suggest_float(name, low, high)


def optimize(
    bars_by_code: dict[str, pd.DataFrame],
    base: BacktestConfig | None = None,
    n_trials: int = 40,
    benchmark: pd.Series | None = None,
    out_dir: Path | None = None,
    seed: int = 42,
) -> dict:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    base = base or BacktestConfig()

    def objective(trial) -> float:
        params = {name: _suggest(trial, name) for name in SEARCH_SPACE}
        cfg = replace(base, **params)
        # rsi_oversold is a floor for a mean-reversion entry that the long-only momentum
        # core never reaches; keep it searchable so importance reflects the full surface.
        result = run_backtest(bars_by_code, cfg, benchmark=benchmark)
        stats = summarize(result, risk_free=base.risk_free)
        score = stats["sharpe"]
        if abs(stats["max_drawdown"]) > 0.25:
            score -= 1.0
        trial.set_user_attr("stats", stats)
        trial.set_user_attr("params", params)
        return float(score)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    importance = {}
    try:
        importance = {k: round(v, 4) for k, v in optuna.importance.get_param_importances(study).items()}
    except Exception:  # noqa: BLE001 - needs >=2 completed trials with variance
        importance = {}

    report = {
        "n_trials": len(study.trials),
        "best_value": round(float(study.best_value), 4),
        "best_params": study.best_params,
        "best_stats": study.best_trial.user_attrs.get("stats", {}),
        "param_importance": importance,
        "history": [
            {
                "number": t.number,
                "value": None if t.value is None else round(float(t.value), 4),
                "params": t.params,
            }
            for t in study.trials
        ],
    }

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "optuna_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    return report
