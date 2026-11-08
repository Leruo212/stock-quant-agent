"""Minimal OpenAI-compatible chat client.

No SDK dependency: one POST, one JSON decode. Works with DeepSeek, Qwen (DashScope
compatible mode), Moonshot, vLLM, Ollama, or anything else exposing
``POST {base_url}/chat/completions`` with tool calling.
"""

from __future__ import annotations

from typing import Any

import requests

from ..config import LLMConfig


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self.session = requests.Session()

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict[str, Any]:
        if not self.cfg.enabled:
            raise LLMError("no api key configured")
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        try:
            resp = self.session.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {self.cfg.api_key}", "Content-Type": "application/json"},
                timeout=self.cfg.timeout,
            )
        except requests.RequestException as exc:
            raise LLMError(f"request failed: {exc}") from exc
        if resp.status_code >= 400:
            raise LLMError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        try:
            return resp.json()["choices"][0]["message"]
        except (KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"unexpected response shape: {resp.text[:300]}") from exc
