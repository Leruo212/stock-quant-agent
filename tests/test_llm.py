"""Tests for the LLM forecasting layer.

These matter more than they look: the failure modes here are silent. A parser that
accepts a malformed forecast, a snapshot that reads one bar too far, or a ledger that
double-counts an entry all produce plausible-looking numbers that are wrong.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from quantagent.config import LLMConfig
from quantagent.llm.client import MockClient, extract_json
from quantagent.llm.context import market_snapshot, render_snapshot
from quantagent.llm.forecast import build_messages, forecast, heuristic_forecast, validate
from quantagent.llm.ledger import Ledger


def _bars(n: int = 120, seed: int = 0, start: str = "2025-01-02") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0004, 0.015, n))
    dates = pd.bdate_range(start, periods=n)
    return pd.DataFrame({
        "date": dates,
        "open": close * (1 + rng.normal(0, 0.003, n)),
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": rng.integers(1e5, 1e6, n).astype(float),
    })


# ------------------------------------------------------------------ parsing
def test_extract_json_clean():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_fenced_and_preamble():
    text = '好的，结果如下：\n```json\n{"direction": "up", "confidence": 0.6}\n```\n以上。'
    assert extract_json(text)["direction"] == "up"


def test_extract_json_nested_braces():
    text = 'noise {"a": {"b": 2}, "c": 3} trailing'
    assert extract_json(text) == {"a": {"b": 2}, "c": 3}


def test_extract_json_rejects_garbage():
    with pytest.raises(Exception):
        extract_json("抱歉，我无法判断。")


# --------------------------------------------------------------- validation
def test_validate_clamps_and_aligns_sign():
    out = validate({"direction": "down", "confidence": 3, "expected_return_pct": 12}, 7)
    assert out["confidence"] == 1.0
    assert out["expected_return_pct"] == -12.0   # direction wins over the sign
    assert out["horizon_days"] == 7


def test_validate_handles_synonyms_and_junk():
    assert validate({"direction": "看多"}, 7)["direction"] == "up"
    assert validate({"direction": "whatever"}, 7)["direction"] == "flat"
    assert validate({"confidence": "high"}, 7)["confidence"] == 0.5


def test_validate_caps_absurd_magnitude():
    assert validate({"direction": "up", "expected_return_pct": 999}, 7)["expected_return_pct"] == 30.0


# ------------------------------------------------------------------ context
def test_snapshot_is_causal():
    bars = _bars(120)
    as_of = bars["date"].iloc[79]
    cut = market_snapshot(bars, code="000001", as_of=as_of)
    # a violent jump after as_of must not appear anywhere in the snapshot
    tampered = bars.copy()
    tampered.loc[80:, "close"] *= 5
    tampered.loc[80:, "high"] *= 5
    after = market_snapshot(tampered, code="000001", as_of=as_of)
    assert cut["close"] == after["close"]
    assert cut["ret_5d_pct"] == after["ret_5d_pct"]


def test_snapshot_rejects_short_history():
    with pytest.raises(ValueError):
        market_snapshot(_bars(30), code="000001")


def test_render_includes_keys():
    text = render_snapshot({"ret_5d_pct": 2.5, "rsi_14": 61.0})
    assert "ret_5d_pct: 2.5" in text and "rsi_14: 61.0" in text


def test_prompt_carries_snapshot_and_horizon():
    snap = market_snapshot(_bars(120), code="000001", name="测试")
    messages = build_messages(snap, horizon=7, model_hint=0.012)
    assert "7 个交易日" in messages[0]["content"]
    assert "ret_5d_pct" in messages[1]["content"]
    assert "1.20%" in messages[1]["content"]  # model hint rendered


# ----------------------------------------------------------------- forecast
def test_mock_forecast_end_to_end():
    snap = market_snapshot(_bars(120), code="000001", name="测试")
    cfg = LLMConfig(api_key="", provider="mock")
    result = forecast(snap, 7, cfg, client=MockClient(cfg), n_samples=3)
    assert result["provider"] == "mock"
    assert result["direction"] in {"up", "down", "flat"}
    assert result["ensemble"]["n"] == 3
    assert result["context_hash"] == result["context_hash"]
    assert result["drivers"]


def test_heuristic_is_never_labelled_as_llm():
    snap = market_snapshot(_bars(120), code="000001")
    out = heuristic_forecast(snap, 7)
    assert out["provider"] == "heuristic"
    assert "不是大模型结论" in out["risk"]


# ------------------------------------------------------------------- ledger
def test_ledger_dedupes_same_key(tmp_path):
    ledger = Ledger(tmp_path / "p.jsonl")
    snap = market_snapshot(_bars(120), code="000001")
    fc = heuristic_forecast(snap, 7)
    first_id, created = ledger.record(fc)
    again_id, created_again = ledger.record(fc)
    assert created is True and created_again is False
    assert first_id == again_id
    assert len(ledger.entries()) == 1


def test_ledger_scores_matured_entries(tmp_path):
    bars = _bars(150, seed=3)
    as_of = bars["date"].iloc[100]
    ledger = Ledger(tmp_path / "p.jsonl")
    ledger.record({
        "code": "000001", "as_of": as_of.strftime("%Y-%m-%d"), "horizon_days": 7,
        "provider": "mock", "model": "mock", "prompt_version": "v1",
        "direction": "up", "confidence": 0.6, "expected_return_pct": 1.5,
        "drivers": [], "ensemble": {},
    })
    report = ledger.score(lambda code: bars, min_entries=1)
    assert report["n_scored"] == 1
    detail = report["detail"][0]
    expected = (bars["close"].iloc[107] / bars["close"].iloc[100] - 1) * 100
    assert detail["realized_return_pct"] == pytest.approx(round(expected, 2))
    assert report["direction_accuracy"] in {0.0, 1.0}


def test_ledger_marks_unexpired_as_pending(tmp_path):
    bars = _bars(150, seed=4)
    as_of = bars["date"].iloc[-2]
    ledger = Ledger(tmp_path / "p.jsonl")
    ledger.record({
        "code": "000001", "as_of": as_of.strftime("%Y-%m-%d"), "horizon_days": 7,
        "provider": "mock", "model": "mock", "prompt_version": "v1",
        "direction": "up", "expected_return_pct": 1.0,
    })
    report = ledger.score(lambda code: bars)
    assert report["n_scored"] == 0 and report["pending"] == 1
