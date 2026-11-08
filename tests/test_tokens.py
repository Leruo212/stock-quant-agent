"""Tests for token-saving machinery: result cache and batch forecasting.

The batch path is the one that can silently lose work — if the model returns 4 items for
5 snapshots, the fifth must be retried individually rather than dropped from the ledger.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from quantagent.config import LLMConfig
from quantagent.llm.cache import ForecastCache, cache_key
from quantagent.llm.client import ChatResponse, MockClient
from quantagent.llm.context import market_snapshot
from quantagent.llm.forecast import forecast, forecast_batch


def _snap(code: str, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n = 120
    close = 100 * np.cumprod(1 + rng.normal(0.0004, 0.015, n))
    bars = pd.DataFrame({
        "date": pd.bdate_range("2025-01-02", periods=n),
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": rng.integers(1e5, 1e6, n).astype(float),
    })
    return market_snapshot(bars, code=code, name=f"T{code}")


class CountingMock(MockClient):
    def __init__(self) -> None:
        super().__init__()
        self.count = 0

    def chat(self, messages, json_mode=False, **kwargs):
        self.count += 1
        return super().chat(messages, json_mode=json_mode)


class DroppingClient(MockClient):
    """Returns a batch array that omits the last symbol, forcing the fallback path."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def chat(self, messages, json_mode=False, **kwargs):
        self.calls += 1
        prompt = messages[-1]["content"]
        codes = [part.split("code: ")[1].split("\n")[0] for part in prompt.split("### 标的 ") if "code: " in part]
        if len(codes) > 1:
            # drop the last symbol on purpose
            items = [{"code": c, "direction": "up", "confidence": 0.5,
                      "expected_return_pct": 1.0, "drivers": ["d"], "risk": "r"} for c in codes[:-1]]
            return ChatResponse(content=json.dumps({"items": items}), model="drop", latency=0.0)
        item = {"code": codes[0] if codes else "000001", "direction": "down", "confidence": 0.4,
                "expected_return_pct": -1.0, "drivers": ["fallback"], "risk": "r"}
        return ChatResponse(content=json.dumps(item), model="drop", latency=0.0)


# --------------------------------------------------------------------- cache
def test_cache_roundtrip_and_ttl(tmp_path):
    cache = ForecastCache(tmp_path, ttl_hours=1)
    key = cache_key("000001", "2026-01-01", 7, "openai", "m", "v1", "abc")
    assert cache.get(key) is None
    cache.put(key, {"direction": "up"})
    assert cache.get(key)["direction"] == "up"

    stale = ForecastCache(tmp_path, ttl_hours=0)
    assert stale.get(key) is not None  # ttl<=0 disables expiry


def test_forecast_hits_cache_and_skips_api(tmp_path):
    cfg = LLMConfig(api_key="", provider="mock")
    client = CountingMock()
    cache = ForecastCache(tmp_path)
    snap = _snap("000001")

    first = forecast(snap, 7, cfg, client=client, cache=cache)
    assert first["cached"] is False and client.count == 1

    second = forecast(snap, 7, cfg, client=client, cache=cache)
    assert second["cached"] is True
    assert client.count == 1, "cache hit must not reach the client"


def test_forecast_cache_key_changes_with_snapshot(tmp_path):
    cfg = LLMConfig(api_key="", provider="mock")
    cache = ForecastCache(tmp_path)
    a = forecast(_snap("000001", seed=1), 7, cfg, client=MockClient(cfg), cache=cache)
    b = forecast(_snap("000001", seed=2), 7, cfg, client=MockClient(cfg), cache=cache)
    assert a["context_hash"] != b["context_hash"]
    assert b["cached"] is False


# --------------------------------------------------------------------- batch
def test_batch_returns_one_result_per_snapshot():
    cfg = LLMConfig(api_key="", provider="mock")
    snaps = [_snap("000001", 1), _snap("000002", 2), _snap("000003", 3)]
    out = forecast_batch(snaps, 7, cfg, client=MockClient(cfg), batch_size=3)
    assert [r["code"] for r in out] == ["000001", "000002", "000003"]
    assert all(r["direction"] in {"up", "down", "flat"} for r in out)


def test_batch_falls_back_when_model_drops_a_symbol():
    cfg = LLMConfig(api_key="", provider="mock")
    client = DroppingClient()
    snaps = [_snap("000001", 1), _snap("000002", 2)]
    out = forecast_batch(snaps, 7, cfg, client=client, batch_size=2)

    assert len(out) == 2
    by_code = {r["code"]: r for r in out}
    assert by_code["000001"]["direction"] == "up"      # from the batch array
    assert by_code["000002"]["direction"] == "down"    # individual retry
    assert client.calls == 2


def test_batch_uses_cache_per_symbol(tmp_path):
    cfg = LLMConfig(api_key="", provider="mock")
    cache = ForecastCache(tmp_path)
    client = CountingMock()
    snaps = [_snap("000001", 1), _snap("000002", 2)]

    forecast_batch(snaps, 7, cfg, client=client, batch_size=2, cache=cache)
    assert client.count == 1
    again = forecast_batch(snaps, 7, cfg, client=client, batch_size=2, cache=cache)
    assert client.count == 1, "second batch run must be fully served from cache"
    assert all(r["cached"] for r in again)
