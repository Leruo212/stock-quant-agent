"""Central configuration. Everything overridable by environment variable or CLI flag."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    return Path(raw).expanduser().resolve() if raw else default


@dataclass
class Paths:
    root: Path = PROJECT_ROOT
    cache: Path = field(default_factory=lambda: _env_path("QA_CACHE_DIR", PROJECT_ROOT / "data" / "cache"))
    artifacts: Path = field(default_factory=lambda: _env_path("QA_ARTIFACT_DIR", PROJECT_ROOT / "artifacts"))
    reports: Path = field(default_factory=lambda: _env_path("QA_REPORT_DIR", PROJECT_ROOT / "reports"))
    ledger: Path = field(default_factory=lambda: _env_path("QA_LEDGER_DIR", PROJECT_ROOT / "data" / "ledger"))

    def ensure(self) -> "Paths":
        for p in (self.cache, self.artifacts, self.reports, self.ledger):
            p.mkdir(parents=True, exist_ok=True)
        return self


@dataclass
class DataConfig:
    # Universe: the N largest A shares by total market cap, frozen on first fetch.
    universe_size: int = 300
    # Eastmoney board id, only used if the Sina path is unavailable
    board: str = "BK0500"
    start: str = "20200101"
    end: str = ""
    adjust: str = "qfq"           # forward-adjusted prices
    min_history: int = 250        # drop names with less history than this
    request_interval: float = 0.12  # seconds between upstream calls, be a good citizen


@dataclass
class FeatureConfig:
    window: int = 7               # observation window W, as in the thesis
    horizon: int = 7              # predict close H trading days ahead
    feature_set: str = "stationary9"  # "stationary9" (default) | "raw9"
    label_mode: str = "raw"       # "raw" (relative return) | "rank" (cross-sectional percentile)
    clip: float = 0.20            # winsorise labels at +/-20%


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.2
    train_ratio: float = 0.8
    seed: int = 42
    threads: int = 0              # 0 -> leave torch default
    patience: int = 12            # early stop on val loss


@dataclass
class BacktestConfig:
    rebalance_days: int = 7
    signal: str = "momentum"      # "momentum" (thesis) | "reversal" (buy the losers)
    top_k: int = 0                # 0 -> hold every name that passes the filters
    position_size: float = 0.2    # cap on any single name's weight
    stop_loss: float = 0.08
    take_profit: float = 0.15
    # the remaining knobs are the ten-parameter surface from thesis table 3-3
    ma_short: int = 10
    ma_long: int = 60
    rsi_period: int = 14
    rsi_oversold: int = 30
    rsi_overbought: int = 75
    volatility_threshold: float = 0.035
    sentiment_threshold: float = 0.0
    # frictions
    cost_bps: float = 8.0         # commission + stamp duty, round trip
    slippage_bps: float = 2.0
    risk_free: float = 0.02
    start: str = "2023-01-01"
    end: str = "2024-12-31"


@dataclass
class LLMConfig:
    """Any OpenAI-compatible endpoint works: DeepSeek, Qwen, Moonshot, Ollama..."""

    base_url: str = field(default_factory=lambda: os.getenv("QA_LLM_BASE_URL", "https://api.deepseek.com/v1"))
    model: str = field(default_factory=lambda: os.getenv("QA_LLM_MODEL", "deepseek-chat"))
    api_key: str = field(default_factory=lambda: os.getenv("QA_LLM_API_KEY", ""))
    temperature: float = 0.2
    max_steps: int = 6
    timeout: float = 60.0
    # forecasting: >1 samples gives a free uncertainty estimate via disagreement
    n_samples: int = field(default_factory=lambda: int(os.getenv("QA_LLM_SAMPLES", "1")))
    provider: str = field(default_factory=lambda: os.getenv("QA_LLM_PROVIDER", "auto"))  # auto | openai | mock
    json_mode: bool = True
    max_output_tokens: int = 400      # a forecast JSON is ~200 tokens; cap the tail risk
    use_cache: bool = field(default_factory=lambda: os.getenv("QA_LLM_NO_CACHE", "") == "")

    @property
    def enabled(self) -> bool:
        return bool(self.api_key) or self.provider == "mock"


@dataclass
class Settings:
    paths: Paths = field(default_factory=Paths)
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)


def load_settings() -> Settings:
    settings = Settings()
    settings.paths.ensure()
    return settings
