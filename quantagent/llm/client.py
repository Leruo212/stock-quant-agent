"""Chat client for any OpenAI-compatible endpoint, plus a deterministic stand-in.

One POST, one JSON decode, explicit retry policy. No SDK: the wire format is stable
enough that a dependency would only add version churn.

Retry policy is deliberately narrow — 429 and 5xx only. A 400 means the request is
wrong and retrying it just burns quota.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

import requests

from ..config import LLMConfig


class LLMError(RuntimeError):
    pass


@dataclass
class ChatResponse:
    content: str
    model: str
    latency: float
    usage: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)


class LLMClient:
    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self.session = requests.Session()

    @property
    def available(self) -> bool:
        return bool(self.cfg.api_key)

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        json_mode: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        retries: int = 2,
    ) -> ChatResponse:
        if not self.available:
            raise LLMError("no api key configured (set QA_LLM_API_KEY)")

        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature if temperature is None else temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {self.cfg.api_key}", "Content-Type": "application/json"}

        last: Exception | None = None
        for attempt in range(retries + 1):
            started = time.time()
            try:
                resp = self.session.post(url, json=payload, headers=headers, timeout=self.cfg.timeout)
            except requests.RequestException as exc:
                last = LLMError(f"request failed: {exc}")
                time.sleep(1.5 * (attempt + 1))
                continue

            latency = time.time() - started
            if resp.status_code == 429 or resp.status_code >= 500:
                last = LLMError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                time.sleep(2 ** attempt)
                continue
            if resp.status_code >= 400:
                raise LLMError(f"HTTP {resp.status_code}: {resp.text[:300]}")

            try:
                body = resp.json()
                message = body["choices"][0]["message"]
            except (KeyError, IndexError, ValueError) as exc:
                raise LLMError(f"unexpected response shape: {resp.text[:300]}") from exc

            return ChatResponse(
                content=message.get("content") or "",
                model=body.get("model", self.cfg.model),
                latency=round(latency, 3),
                usage=body.get("usage", {}) or {},
                raw=message,
            )

        raise last or LLMError("request failed")

    def chat_json(self, messages: list[dict], **kwargs) -> tuple[dict, ChatResponse]:
        response = self.chat(messages, json_mode=True, **kwargs)
        return extract_json(response.content), response


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model response.

    Handles the three shapes seen in practice: clean JSON, fenced JSON, and JSON with
    a sentence of preamble. Raises instead of guessing — a forecast with invented keys
    is worse than a forecast that failed.
    """
    if not text or not text.strip():
        raise LLMError("empty response")

    candidates = []
    fenced = _FENCE.search(text)
    if fenced:
        candidates.append(fenced.group(1))
    candidates.append(text)

    start = text.find("{")
    if start >= 0:
        depth = 0
        for i, ch in enumerate(text[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start: i + 1])
                    break

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            continue

    raise LLMError(f"no JSON object in response: {text[:200]!r}")


class MockClient:
    """Deterministic stand-in with the same interface as :class:`LLMClient`.

    Exists so the prompt → parse → ledger chain can be exercised in tests and by anyone
    cloning the repo without a key. It is *not* a model: it reads ``ret_5d_pct`` out of
    the prompt and emits a rule-based answer, and every forecast it produces is tagged
    ``provider="mock"`` so it can never be mistaken for an LLM output.
    """

    def __init__(self, cfg: LLMConfig | None = None) -> None:
        self.cfg = cfg or LLMConfig()
        self.calls: list[list[dict]] = []

    @property
    def available(self) -> bool:
        return True

    def chat(self, messages: list[dict], json_mode: bool = False, **kwargs) -> ChatResponse:
        self.calls.append(messages)
        prompt = messages[-1]["content"]

        # batch prompts carry one "### 标的 N" block per symbol; answer them together so
        # the offline path exercises the same code as a real batch call
        blocks = prompt.split("### 标的 ")
        if len(blocks) > 1:
            items = []
            for block in blocks[1:]:
                item = self._answer(block)
                item["code"] = _find_str(block, "code") or ""
                items.append(item)
            return ChatResponse(content=json.dumps({"items": items}, ensure_ascii=False), model="mock", latency=0.0)

        return ChatResponse(content=json.dumps(self._answer(prompt), ensure_ascii=False), model="mock", latency=0.0)

    @staticmethod
    def _answer(block: str) -> dict:
        ret5 = _find_number(block, "ret_5d_pct")
        ret20 = _find_number(block, "ret_20d_pct")
        rsi = _find_number(block, "rsi_14")

        score = (ret5 or 0) * 0.6 + (ret20 or 0) * 0.4
        if rsi is not None and rsi > 75:
            score -= 1.5
        if rsi is not None and rsi < 25:
            score += 1.5

        direction = "up" if score > 0.5 else ("down" if score < -0.5 else "flat")
        return {
            "direction": direction,
            "confidence": round(min(0.62, 0.35 + abs(score) / 40), 2),
            "expected_return_pct": round(max(-8.0, min(8.0, score * 0.35)), 2),
            "drivers": [f"5 日动量 {ret5}%", f"20 日动量 {ret20}%"],
            "risk": "mock 客户端输出，仅用于链路验证",
        }


def _find_number(text: str, key: str) -> float | None:
    match = re.search(rf"{key}:\s*(-?\d+(?:\.\d+)?)", text)
    return float(match.group(1)) if match else None


def _find_str(text: str, key: str) -> str | None:
    match = re.search(rf"{key}:\s*(\S+)", text)
    return match.group(1) if match else None
