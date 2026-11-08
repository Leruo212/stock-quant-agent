"""Append-only prediction ledger.

Why a ledger instead of a backtest: an LLM has already read the price history you would
be backtesting on. Ask it in 2026 what it thinks about a 2023 window and you are not
measuring forecasting skill, you are measuring recall. There is no prompt that fixes
this, because the leakage happened during pretraining.

So the only defensible evaluation is forward: record what the model said, with the
snapshot it saw and the exact prompt version, then grade it once the horizon has
actually elapsed. This module does both halves.

Format: one JSON object per line (JSONL), append-only, so a partial write can never
corrupt earlier records and the file stays diffable.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


class Ledger:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ write
    def record(self, forecast: dict, note: str = "") -> tuple[str, bool]:
        """Append a forecast. Returns (id, created?) — duplicates are skipped."""
        key = (
            forecast.get("code"),
            forecast.get("as_of"),
            forecast.get("provider"),
            forecast.get("prompt_version"),
        )
        for existing in self.entries():
            if (existing.get("code"), existing.get("as_of"),
                    existing.get("provider"), existing.get("prompt_version")) == key:
                return existing.get("id", ""), False

        entry = {
            "id": uuid.uuid4().hex[:12],
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "code": forecast.get("code"),
            "name": forecast.get("name"),
            "as_of": forecast.get("as_of"),
            "horizon_days": forecast.get("horizon_days"),
            "provider": forecast.get("provider"),
            "model": forecast.get("model"),
            "prompt_version": forecast.get("prompt_version"),
            "context_hash": forecast.get("context_hash"),
            "direction": forecast.get("direction"),
            "confidence": forecast.get("confidence"),
            "expected_return_pct": forecast.get("expected_return_pct"),
            "drivers": forecast.get("drivers", []),
            "risk": forecast.get("risk", ""),
            "ensemble": forecast.get("ensemble", {}),
            "note": note,
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry["id"], True

    # ------------------------------------------------------------------- read
    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a truncated tail should not poison the whole ledger
        return out

    # ------------------------------------------------------------------ score
    def score(self, bars_provider, min_entries: int = 5) -> dict:
        """Grade every entry whose horizon has elapsed.

        ``bars_provider(code) -> DataFrame`` must return full history including the
        bars after ``as_of``.
        """
        entries = self.entries()
        if not entries:
            return {"n_entries": 0, "n_scored": 0, "pending": 0}

        cache: dict[str, pd.DataFrame] = {}
        scored: list[dict] = []
        pending = 0

        for entry in entries:
            code = entry.get("code")
            as_of = entry.get("as_of")
            horizon = int(entry.get("horizon_days") or 7)
            if not code or not as_of:
                continue

            if code not in cache:
                try:
                    frame = bars_provider(code)
                except Exception:  # noqa: BLE001 - missing symbol just stays unscored
                    cache[code] = pd.DataFrame()
                    continue
                frame = frame.copy()
                frame["date"] = pd.to_datetime(frame["date"])
                cache[code] = frame.sort_values("date").reset_index(drop=True)

            frame = cache[code]
            if frame.empty:
                continue
            dates = frame["date"]
            hit = frame.index[dates == pd.Timestamp(as_of)]
            if len(hit) == 0:
                continue
            idx = int(hit[0])
            if idx + horizon >= len(frame):
                pending += 1
                continue

            realized = float(frame["close"].iloc[idx + horizon] / frame["close"].iloc[idx] - 1) * 100
            expected = float(entry.get("expected_return_pct") or 0.0)
            direction = entry.get("direction")
            if direction == "up":
                correct = realized > 0
            elif direction == "down":
                correct = realized < 0
            else:
                correct = abs(realized) < 3.0  # "flat" graded as a +/-3% band

            scored.append({
                "code": code,
                "as_of": as_of,
                "provider": entry.get("provider"),
                "model": entry.get("model"),
                "direction": direction,
                "confidence": entry.get("confidence"),
                "expected_return_pct": expected,
                "realized_return_pct": round(realized, 2),
                "correct": bool(correct),
                "abs_error_pct": round(abs(expected - realized), 2),
            })

        if not scored:
            return {"n_entries": len(entries), "n_scored": 0, "pending": pending}

        frame = pd.DataFrame(scored)
        # pandas' method="spearman" pulls in scipy; ranking first avoids the dependency
        from ..evaluate import spearman

        per_date = (
            frame.groupby("as_of")[["expected_return_pct", "realized_return_pct"]]
            .apply(lambda g: spearman(g["expected_return_pct"].to_numpy(), g["realized_return_pct"].to_numpy()))
            .dropna()
        )
        n_dates = int(frame["as_of"].nunique())
        # a single day with 20 names is one observation of skill, not twenty
        enough = len(frame) >= min_entries and n_dates >= 3

        report = {
            "n_entries": len(entries),
            "n_scored": int(len(frame)),
            "pending": int(pending),
            "direction_accuracy": round(float(frame["correct"].mean()), 4),
            "baseline_up_rate": round(float((frame["realized_return_pct"] > 0).mean()), 4),
            "mae_pct": round(float(frame["abs_error_pct"].mean()), 2),
            "correlation": round(float(frame["expected_return_pct"].corr(frame["realized_return_pct"])), 4)
            if len(frame) > 2 else None,
            "mean_ic": round(float(per_date.mean()), 4) if len(per_date) else None,
            "n_dates": n_dates,
            "by_provider": {
                name: {
                    "n": int(len(g)),
                    "accuracy": round(float(g["correct"].mean()), 4),
                    "mae_pct": round(float(g["abs_error_pct"].mean()), 2),
                }
                for name, g in frame.groupby("provider")
            },
            "insufficient_sample": not enough,
            "verdict": (
                "样本尚不足以判断预测能力：至少需要 %d 条已到期记录且分布在 ≥3 个交易日" % min_entries
                if not enough else "样本量达到最低要求，可以看命中率与基准的差距"
            ),
            "detail": scored[:50],
        }
        return report
