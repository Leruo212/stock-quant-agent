"""LLM-based forward-return forecasting.

Design decisions worth knowing:

**The model is given numbers, not a story.** The prompt is a labelled key-value
snapshot plus an explicit instruction not to use anything outside it. Telling an LLM
"you are a quant, forecast Moutai" invites it to recite what it remembers about Moutai;
telling it "here are 18 numbers as of 2026-11-06, forecast the next 7 sessions" at least
makes the arithmetic checkable.

**Self-consistency over single-shot.** ``n_samples > 1`` runs the same prompt at a
non-zero temperature and aggregates. The spread across samples is a free uncertainty
estimate — a model that says "up" five times with wildly different magnitudes is not
confident, whatever its stated confidence.

**Optional reflection pass.** A second call shows the model its own answer and asks it
to drop any driver that is not traceable to a number in the snapshot, and to shrink the
forecast if it cannot justify the magnitude. Cheap, and it kills most hallucinated
justifications.

**What this is not:** a backtestable signal. See ``docs/LLM_PREDICTION.md``. The only
honest evaluation is a forward-looking ledger, which :mod:`quantagent.llm.ledger`
implements.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

from ..config import LLMConfig
from .client import LLMClient, LLMError, MockClient, extract_json
from .context import render_snapshot

PROMPT_VERSION = "v1"

DIRECTIONS = {"up", "down", "flat"}

SYSTEM_PROMPT = """你是一个 A 股量化研究员，任务只有一个：根据给定的数值快照，判断标的在未来 {horizon} 个交易日的方向。

硬性要求：
1. 只能使用快照里出现的数字。禁止引用你对这家公司的任何记忆、新闻、传闻或行业知识。
2. 不要解释你在做什么，不要输出 markdown 代码块，只输出一个 JSON 对象。
3. drivers 里的每一条都必须引用快照里的具体数字（写出字段名与数值）。
4. 如果不确定，就降低 confidence 并把 expected_return_pct 压到 ±2% 以内，不要为了显得有观点而编造幅度。

输出格式：
{{"direction": "up|down|flat",
  "confidence": 0.0-1.0,
  "expected_return_pct": 数字,
  "drivers": ["引用数字的一句", "..."],
  "risk": "一句话，指出这个判断最可能错在哪"}}"""

REFLECT_PROMPT = """下面是你刚才给出的预测，以及它依据的数值快照。

请逐条检查 drivers：凡是不能在快照里找到对应数字的，删掉。
然后判断：以快照能支撑的证据强度，expected_return_pct 的绝对值是否过大？如果过大，向 0 收缩。
如果删完后没有一条 driver 站得住，把 direction 改为 "flat"、confidence 压到 0.3 以下。

只输出修正后的同一个 JSON 对象，格式不变。"""


def _context_hash(snapshot: dict) -> str:
    payload = json.dumps(snapshot, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def build_messages(snapshot: dict, horizon: int, model_hint: float | None = None) -> list[dict]:
    body = [render_snapshot(snapshot)]
    if model_hint is not None:
        body.append(
            f"\n参考：一个本地训练的 CNN-LSTM-Attention 统计模型在同一窗口给出 {model_hint * 100:+.2f}%，"
            "它的样本外 IC 接近 0，只可作为参考，不应主导你的判断。"
        )
    body.append(f"\n请判断未来 {horizon} 个交易日的方向，按要求输出 JSON。")
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(horizon=horizon)},
        {"role": "user", "content": "\n".join(body)},
    ]


def validate(payload: dict, horizon: int) -> dict:
    """Coerce a model's JSON into the fields downstream code relies on."""
    direction = str(payload.get("direction", "")).strip().lower()
    if direction not in DIRECTIONS:
        if direction in {"bullish", "long", "rise", "看多"}:
            direction = "up"
        elif direction in {"bearish", "short", "fall", "看空"}:
            direction = "down"
        else:
            direction = "flat"

    try:
        confidence = float(payload.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    try:
        expected = float(payload.get("expected_return_pct", 0.0))
    except (TypeError, ValueError):
        expected = 0.0
    # a 7-day A-share forecast beyond +/-30% is a parsing accident, not a view
    expected = max(-30.0, min(30.0, expected))
    if direction == "up":
        expected = abs(expected) if expected != 0 else 0.5
    elif direction == "down":
        expected = -abs(expected) if expected != 0 else -0.5

    drivers = payload.get("drivers") or []
    if isinstance(drivers, str):
        drivers = [drivers]
    drivers = [str(d).strip()[:160] for d in drivers if str(d).strip()][:4]

    return {
        "direction": direction,
        "confidence": round(confidence, 3),
        "expected_return_pct": round(expected, 2),
        "drivers": drivers,
        "risk": str(payload.get("risk", "")).strip()[:200],
        "horizon_days": horizon,
    }


def forecast(
    snapshot: dict,
    horizon: int = 7,
    cfg: LLMConfig | None = None,
    client: LLMClient | MockClient | None = None,
    n_samples: int | None = None,
    reflect: bool = False,
    model_hint: float | None = None,
) -> dict:
    """Run one (or N) LLM forecasts and aggregate."""
    cfg = cfg or LLMConfig()
    client = client or (LLMClient(cfg) if cfg.api_key else MockClient(cfg))
    provider = "openai" if isinstance(client, LLMClient) else "mock"
    samples = n_samples or cfg.n_samples

    messages = build_messages(snapshot, horizon, model_hint=model_hint)
    results: list[dict] = []
    latencies: list[float] = []
    usage_total: dict = {}
    model_name = ""

    for i in range(max(1, samples)):
        try:
            response = client.chat(messages, json_mode=True, temperature=0.0 if samples == 1 else max(cfg.temperature, 0.7))
        except LLMError as exc:
            if not results:
                raise
            break
        model_name = response.model
        latencies.append(response.latency)
        for k, v in (response.usage or {}).items():
            if isinstance(v, (int, float)):
                usage_total[k] = usage_total.get(k, 0) + v
        results.append(validate(extract_json(response.content), horizon))

    if not results:
        raise LLMError("all samples failed")

    # self-consistency aggregation
    returns = sorted(r["expected_return_pct"] for r in results)
    median_return = returns[len(returns) // 2]
    mid = median_return
    votes = {}
    for r in results:
        votes[r["direction"]] = votes.get(r["direction"], 0) + 1
    direction = max(votes, key=votes.get)
    agreement = votes[direction] / len(results)
    spread = float(returns[-1] - returns[0]) if len(returns) > 1 else 0.0

    merged = {
        "direction": direction,
        "confidence": round(sum(r["confidence"] for r in results) / len(results), 3),
        "expected_return_pct": round(mid, 2),
        "drivers": results[0]["drivers"],
        "risk": results[0]["risk"],
        "horizon_days": horizon,
    }

    out = {
        "provider": provider,
        "model": model_name or cfg.model,
        "code": snapshot.get("code", ""),
        "name": snapshot.get("name", ""),
        "as_of": snapshot.get("as_of", ""),
        **merged,
        "ensemble": {
            "n": len(results),
            "direction_agreement": round(agreement, 3),
            "return_range_pct": round(spread, 2),
        },
        "latency_s": round(sum(latencies), 3),
        "usage": usage_total,
        "prompt_version": PROMPT_VERSION,
        "context_hash": _context_hash(snapshot),
        "snapshot": snapshot,
    }

    if reflect:
        out["reflection"] = _reflect(client, messages, merged, snapshot)

    return out


def _reflect(client, messages: list[dict], answer: dict, snapshot: dict) -> dict:
    critique = [
        {"role": "system", "content": SYSTEM_PROMPT.format(horizon=answer["horizon_days"])},
        {"role": "user", "content": "\n".join([
            render_snapshot(snapshot),
            "",
            REFLECT_PROMPT,
            "",
            json.dumps(answer, ensure_ascii=False),
        ])},
    ]
    try:
        response = client.chat(critique, json_mode=True, temperature=0.0)
        revised = validate(extract_json(response.content), answer["horizon_days"])
    except LLMError:
        return {"ok": False, "reason": "reflection call failed"}

    changed = [
        key for key in ("direction", "confidence", "expected_return_pct")
        if revised[key] != answer[key]
    ]
    return {
        "ok": True,
        "revised": revised,
        "changed_fields": changed,
        "drivers_dropped": max(len(answer["drivers"]) - len(revised["drivers"]), 0),
    }


def heuristic_forecast(snapshot: dict, horizon: int = 7) -> dict:
    """Transparent fallback when no API key is configured.

    Never pretend this is a model output. It is a documented baseline so the rest of the
    pipeline (ledger, UI, scoring) has something to run against offline.
    """
    ret5 = snapshot.get("ret_5d_pct") or 0.0
    ret20 = snapshot.get("ret_20d_pct") or 0.0
    rsi = snapshot.get("rsi_14") or 50.0

    score = ret5 * 0.6 + ret20 * 0.4
    if rsi > 75:
        score -= 1.2
    elif rsi < 25:
        score += 1.2

    direction = "up" if score > 0.5 else ("down" if score < -0.5 else "flat")
    return {
        "provider": "heuristic",
        "model": "none",
        "code": snapshot.get("code", ""),
        "name": snapshot.get("name", ""),
        "as_of": snapshot.get("as_of", ""),
        "direction": direction,
        "confidence": 0.3,
        "expected_return_pct": round(max(-6.0, min(6.0, score * 0.3)), 2),
        "drivers": [f"5 日动量 {ret5:+.2f}%", f"20 日动量 {ret20:+.2f}%", f"RSI14 {rsi:.0f}"],
        "risk": "启发式占位输出：未配置 QA_LLM_API_KEY，这不是大模型结论",
        "horizon_days": horizon,
        "ensemble": {"n": 1, "direction_agreement": 1.0, "return_range_pct": 0.0},
        "latency_s": 0.0,
        "usage": {},
        "prompt_version": PROMPT_VERSION,
        "context_hash": _context_hash(snapshot),
        "snapshot": snapshot,
    }


def forecast_or_fallback(snapshot: dict, horizon: int, cfg: LLMConfig, **kwargs) -> dict:
    explicit_client = kwargs.get("client") is not None
    if cfg.api_key or cfg.provider == "mock" or explicit_client:
        try:
            return forecast(snapshot, horizon, cfg, **kwargs)
        except LLMError as exc:
            fallback = heuristic_forecast(snapshot, horizon)
            fallback["error"] = str(exc)
            return fallback
    return heuristic_forecast(snapshot, horizon)


def config_dict(cfg: LLMConfig) -> dict:
    d = asdict(cfg)
    d["api_key"] = "***" if cfg.api_key else ""
    return d
