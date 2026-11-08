"""Thin, dependency-light client for Eastmoney's public quote endpoints.

Deliberately not using a heavyweight wrapper library: two endpoints are enough for
this project, and rolling them by hand keeps the install small and the failure modes
visible. All responses are cached on disk by :mod:`quantagent.data.cache`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Iterable

import pandas as pd
import requests

KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
LIST_URL = "https://push2.eastmoney.com/api/qt/clist/get"

# f51..f61 of the kline payload, in payload order
KLINE_FIELDS = "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"
KLINE_COLUMNS = [
    "date", "open", "close", "high", "low", "volume",
    "amount", "amplitude", "pct_chg", "change", "turnover",
]

_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


class UpstreamError(RuntimeError):
    """Raised when the upstream returns something we cannot parse."""


def secid(code: str, market: str | None = None) -> str:
    """Map a 6-digit A-share code to Eastmoney's ``market.code`` identifier.

    ``market`` overrides the heuristic, which is needed for Shanghai *indices* such as
    ``000300`` that share the 0-prefix with Shenzhen equities.
    """
    code = str(code).zfill(6)
    if market is None:
        # 6xxxxx / 9xxxxx are Shanghai, everything else (0/2/3/8/4) is Shenzhen or Beijing
        market = "1" if code[0] in {"6", "9"} else "0"
    return f"{market}.{code}"


@dataclass
class EastmoneyClient:
    timeout: float = 20.0
    retries: int = 3
    interval: float = 0.12
    session: requests.Session | None = None

    def __post_init__(self) -> None:
        if self.session is None:
            self.session = requests.Session()
            self.session.headers.update({"User-Agent": _UA, "Referer": "https://quote.eastmoney.com/"})

    # ------------------------------------------------------------------ helpers
    def _get(self, url: str, params: dict) -> dict:
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                resp.raise_for_status()
                payload = resp.json()
                if payload.get("data") is None:
                    raise UpstreamError(f"empty data for {params}")
                return payload
            except Exception as exc:  # noqa: BLE001 - retried below
                last = exc
                time.sleep(0.6 * (attempt + 1))
        raise UpstreamError(f"request failed after {self.retries} attempts: {last}")

    # -------------------------------------------------------------------- calls
    def constituents(self, board: str = "BK0500") -> pd.DataFrame:
        """CSI300 constituent list. ``BK0500`` is Eastmoney's board id for it."""
        payload = self._get(
            LIST_URL,
            {
                "pn": 1, "pz": 500, "po": 1, "np": 1, "fltt": 2, "invt": 2,
                "fid": "f3", "fs": f"b:{board}", "fields": "f12,f14",
            },
        )
        rows = payload["data"]["diff"]
        frame = pd.DataFrame({"code": [r["f12"] for r in rows], "name": [r["f14"] for r in rows]})
        return frame.drop_duplicates("code").reset_index(drop=True)

    def daily(
        self,
        code: str,
        start: str = "20191201",
        end: str = "20500101",
        adjust: str = "qfq",
        market: str | None = None,
    ) -> pd.DataFrame:
        """Forward-adjusted daily bars for one name (or index when ``market`` is given)."""
        fqt = {"qfq": "1", "hfq": "2", "none": "0"}[adjust]
        payload = self._get(
            KLINE_URL,
            {
                "secid": secid(code, market),
                "fields1": "f1,f2,f3,f4,f5,f6",
                "fields2": KLINE_FIELDS,
                "klt": 101,          # 101 = daily
                "fqt": int(fqt),
                "beg": start,
                "end": end,
                "lmt": 100000,
            },
        )
        klines: Iterable[str] = payload["data"]["klines"]
        rows = [line.split(",") for line in klines]
        if not rows:
            raise UpstreamError(f"no bars returned for {code}")
        frame = pd.DataFrame(rows, columns=KLINE_COLUMNS)
        frame["code"] = str(payload["data"].get("code", code)).zfill(6)
        frame["name"] = payload["data"].get("name", "")
        for col in KLINE_COLUMNS[1:]:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame["date"] = pd.to_datetime(frame["date"])
        # Eastmoney emits "-" for suspended days in a few fields
        frame = frame.dropna(subset=["open", "close", "high", "low"]).sort_values("date")
        return frame.reset_index(drop=True)

    def quote(self, code: str) -> dict:
        """Latest snapshot, used by the agent's ``get_quote`` tool."""
        bar = self.daily(code, start="20240101").tail(1)
        if bar.empty:
            raise UpstreamError(f"no quote for {code}")
        row = bar.iloc[0]
        return {
            "code": str(row["code"]),
            "name": str(row["name"]),
            "date": row["date"].strftime("%Y-%m-%d"),
            "close": float(row["close"]),
            "pct_chg": float(row["pct_chg"]),
            "volume": float(row["volume"]),
            "turnover": float(row["turnover"]),
        }
