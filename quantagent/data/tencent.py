"""Tencent quote client — the primary bar source.

Why this one: ``web.ifzq.gtimg.cn`` returns forward-adjusted daily bars as plain JSON,
accepts 640 bars per call, and is noticeably more tolerant of repeated requests than
Eastmoney's ``push2`` cluster, which started returning connection resets after ~3 calls
from one IP during development.

Payload shape::

    data["sh600519"]["qfqday"] = [
        ["2024-01-05", open, close, high, low, volume],   # volume in 手 (100 shares)
        ...
    ]

Note the column order: open, **close**, high, low. Getting that wrong silently produces
a chart that looks fine and a backtest that is nonsense.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd
import requests

BASE = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
PAGE_SIZE = 640

_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


class TencentError(RuntimeError):
    pass


def symbol(code: str, index: bool = False) -> str:
    code = str(code).zfill(6)
    if index or code[0] in {"6", "9"}:
        return f"sh{code}"
    return f"sz{code}"


@dataclass
class TencentClient:
    timeout: float = 20.0
    retries: int = 3
    interval: float = 0.1

    def __post_init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": _UA, "Referer": "https://gu.qq.com/"})

    def _fetch(self, sym: str, end: str, adjust: str) -> list[list[str]]:
        field = "qfqday" if adjust == "qfq" else "day"
        tail = "qfq" if adjust == "qfq" else ""
        param = f"{sym},day,,{end},{PAGE_SIZE},{tail}"
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                resp = self.session.get(BASE, params={"param": param}, timeout=self.timeout)
                resp.raise_for_status()
                payload = resp.json()
                if payload.get("code") != 0:
                    raise TencentError(f"upstream code {payload.get('code')}: {payload.get('msg')}")
                block = payload["data"].get(sym) or {}
                rows = block.get(field) or block.get("day") or []
                return rows
            except Exception as exc:  # noqa: BLE001 - retried below
                last = exc
                time.sleep(0.5 * (attempt + 1))
        raise TencentError(f"{sym}: {last}")

    def daily(
        self,
        code: str,
        start: str = "20200101",
        end: str | None = None,
        adjust: str = "qfq",
        index: bool = False,
    ) -> pd.DataFrame:
        sym = symbol(code, index=index)
        # indices have no adjustment series; asking for qfq on an index returns HTTP 501
        if index:
            adjust = "none"
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
        cursor = end_ts
        chunks: list[list[list[str]]] = []

        for _ in range(12):  # 12 * 640 bars ≈ 30 years, hard stop
            rows = self._fetch(sym, cursor.strftime("%Y-%m-%d"), adjust)
            if not rows:
                break
            chunks.append(rows)
            first = pd.Timestamp(rows[0][0])
            if first <= start_ts:
                break
            cursor = first - pd.Timedelta(days=1)
            time.sleep(self.interval)

        if not chunks:
            raise TencentError(f"no bars for {sym}")

        flat = [row for chunk in reversed(chunks) for row in chunk]
        frame = pd.DataFrame([r[:6] for r in flat], columns=["date", "open", "close", "high", "low", "volume"])
        frame["date"] = pd.to_datetime(frame["date"])
        for col in ("open", "close", "high", "low", "volume"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame = frame.dropna(subset=["open", "close", "high", "low"])
        frame = frame[frame["date"] >= start_ts].drop_duplicates("date").sort_values("date").reset_index(drop=True)
        frame["code"] = str(code).zfill(6)
        frame["amount"] = 0.0
        frame["pct_chg"] = frame["close"].pct_change().fillna(0.0) * 100
        frame["turnover"] = 0.0
        frame["name"] = ""
        return frame
