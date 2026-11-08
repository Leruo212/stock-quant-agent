"""QuantAgent — a reproducible CNN-LSTM-Attention stock quant stack with an LLM agent on top.

Sub-packages:
    data      market data client, caching, feature engineering, dataset building
    models    the CNN-LSTM-Attention network and its training loop
    backtest  event-driven momentum backtester, metrics, Optuna tuning
    agent     tool registry + LLM tool-calling loop
    api       FastAPI service used by the bundled web UI
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
