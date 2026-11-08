"""Sina market-center client — used to build the universe.

Eastmoney's ``BK0500`` board endpoint hands back the official CSI300 constituent list,
but it throttles aggressively. Sina's market centre lets us sort *all* A shares by
total market cap, which gives a close, reproducible stand-in: the 300 largest listed
companies by cap overlap the CSI300 almost completely, and the rule ("top N by cap, as
of the fetch date") is easier to defend than a hard-coded list that silently rots.

The universe snapshot is frozen to ``data/cache/universe.parquet`` on first fetch, so
every later run uses exactly the same 300 names.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd
import requests

BASE = "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData"
KLINE = "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData"
PAGE_SIZE = 100  # the endpoint caps at 100 regardless of what you ask for
MAX_BARS = 1023


class SinaError(RuntimeError):
    pass


@dataclass
class SinaClient:
    timeout: float = 20.0
    retries: int = 3
    interval: float = 0.15

    def __post_init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
            "Referer": "https://finance.sina.com.cn/",
        })

    def _page(self, page: int) -> list[dict]:
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                resp = self.session.get(
                    BASE,
                    params={"page": page, "num": PAGE_SIZE, "sort": "mktcap", "asc": 0, "node": "hs_a"},
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                payload = resp.json()
                if not isinstance(payload, list):
                    raise SinaError(f"unexpected payload on page {page}")
                return payload
            except Exception as exc:  # noqa: BLE001
                last = exc
                time.sleep(0.5 * (attempt + 1))
        raise SinaError(f"page {page}: {last}")

    def top_by_market_cap(self, n: int = 300) -> pd.DataFrame:
        rows: list[dict] = []
        page = 1
        while len(rows) < n and page <= 6:
            batch = self._page(page)
            if not batch:
                break
            rows.extend(batch)
            page += 1
            time.sleep(self.interval)

        frame = pd.DataFrame(rows)
        if frame.empty:
            raise SinaError("empty universe")
        frame = frame[["code", "name", "mktcap", "nmc"]].copy()
        frame["code"] = frame["code"].astype(str).str.zfill(6)
        for col in ("mktcap", "nmc"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        # drop ST / delisting names, they are not investable in a long-only book
        frame = frame[~frame["name"].str.contains("ST|退", na=False)]
        frame = frame.drop_duplicates("code").sort_values("mktcap", ascending=False)
        return frame.head(n).reset_index(drop=True)

    def daily(self, code: str, index: bool = False, datalen: int = MAX_BARS) -> pd.DataFrame:
        """Daily bars, unadjusted. ``datalen`` is capped at 1023 by the endpoint.

        Used for index series (which have no adjustment) and as a last-resort fallback
        for single names. Unadjusted prices carry corporate-action jumps, so the caller
        is told which source it got in ``MarketData.notes``.
        """
        prefix = "sh" if (index or str(code).zfill(6)[0] in {"6", "9"}) else "sz"
        sym = f"{prefix}{str(code).zfill(6)}"
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                resp = self.session.get(
                    KLINE,
                    params={"symbol": sym, "scale": 240, "ma": "no", "datalen": min(datalen, MAX_BARS)},
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                payload = resp.json()
                if not isinstance(payload, list) or not payload:
                    raise SinaError(f"empty kline for {sym}")
                frame = pd.DataFrame(payload)
                frame = frame.rename(columns={"day": "date"})
                frame["date"] = pd.to_datetime(frame["date"])
                for col in ("open", "high", "low", "close", "volume"):
                    frame[col] = pd.to_numeric(frame[col], errors="coerce")
                frame = frame.dropna(subset=["open", "close", "high", "low"])
                frame["code"] = str(code).zfill(6)
                frame["amount"] = 0.0
                frame["pct_chg"] = frame["close"].pct_change().fillna(0.0) * 100
                frame["turnover"] = 0.0
                frame["name"] = ""
                return frame.sort_values("date").reset_index(drop=True)
            except Exception as exc:  # noqa: BLE001
                last = exc
                time.sleep(0.5 * (attempt + 1))
        raise SinaError(f"{sym}: {last}")
