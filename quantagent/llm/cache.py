"""Response cache for LLM forecasts.

Two reasons this is not premature optimisation:

1. The same (code, as_of, prompt version) gets asked repeatedly during UI refreshes and
   re-runs. Every hit is a full prompt plus completion not paid for.
2. A cache makes the output reproducible. Two runs of `forecast` on the same day give
   identical numbers, which matters when the number ends up in a ledger that is later
   scored.

Provider-side context caching is separate and comes for free: the system prompt is a
fixed prefix, so a compatible endpoint can reuse its KV cache between calls. This module
caches the *result*, not the prefix.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path


def cache_key(
    code: str,
    as_of: str,
    horizon: int,
    provider: str,
    model: str,
    prompt_version: str,
    context_hash: str,
) -> str:
    raw = f"{code}|{as_of}|{horizon}|{provider}|{model}|{prompt_version}|{context_hash}"
    return hashlib.sha1(raw.encode()).hexdigest()[:24]


class ForecastCache:
    def __init__(self, root: Path, ttl_hours: int = 12) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_hours * 3600

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, key: str) -> dict | None:
        path = self._path(key)
        if not path.exists():
            return None
        if self.ttl_seconds and time.time() - path.stat().st_mtime > self.ttl_seconds:
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def put(self, key: str, payload: dict) -> None:
        try:
            self._path(key).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass  # a cache write failure must never break a forecast

    def stats(self) -> dict:
        files = list(self.root.glob("*.json"))
        return {
            "entries": len(files),
            "bytes": sum(f.stat().st_size for f in files),
            "root": str(self.root),
        }
