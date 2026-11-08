"""Universe + bar loading with source fallback and a parquet cache.

Primary path: Sina (universe) + Tencent (bars). Fallback: Eastmoney. Both are wrapped
so a caller never sees which one answered — except in the log line, which we keep
because "which source produced this number" is part of the provenance.

The cache is the reproducibility boundary: once ``data/cache/`` is populated a full
pipeline run touches the network zero times.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from ..config import DataConfig, Paths
from .cache import ParquetCache
from .eastmoney import EastmoneyClient
from .sina import SinaClient, SinaError
from .tencent import TencentClient, TencentError


class MarketData:
    def __init__(
        self,
        cfg: DataConfig | None = None,
        paths: Paths | None = None,
        offline: bool = False,
        source: str = "auto",
    ) -> None:
        self.cfg = cfg or DataConfig()
        self.paths = paths or Paths()
        self.paths.ensure()
        self.cache = ParquetCache(self.paths.cache)
        self.offline = offline
        self.source = source
        self.tencent = TencentClient(interval=self.cfg.request_interval)
        self.sina = SinaClient(interval=self.cfg.request_interval)
        self.eastmoney = EastmoneyClient(interval=self.cfg.request_interval)
        self.notes: list[str] = []

    # ---------------------------------------------------------------- universe
    def universe(self, refresh: bool = False, limit: int = 0) -> pd.DataFrame:
        key = f"universe_top{self.cfg.universe_size}"
        cached = self.cache.load(key)
        if cached is not None and not refresh:
            return cached.head(limit or None)

        if self.offline:
            if cached is None:
                raise RuntimeError("offline mode and no cached universe — run once online first")
            return cached.head(limit or None)

        frame: pd.DataFrame | None = None
        if self.source in {"auto", "sina"}:
            try:
                frame = self.sina.top_by_market_cap(self.cfg.universe_size)
                self.notes.append(f"universe: sina top{self.cfg.universe_size} by market cap")
            except SinaError as exc:
                self.notes.append(f"universe: sina failed ({exc})")
        if frame is None:
            try:
                frame = self.eastmoney.constituents(self.cfg.board)
                self.notes.append("universe: eastmoney CSI300 constituents")
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"all universe sources failed: {exc}") from exc

        self.cache.store(key, frame)
        return frame.head(limit or None)

    # -------------------------------------------------------------------- bars
    def bars(self, code: str, refresh: bool = False, market: str | None = None) -> pd.DataFrame:
        key = f"bars_{market or 'auto'}_{code}_{self.cfg.adjust}_{self.cfg.start}"
        if not refresh:
            cached = self.cache.load(key)
            if cached is not None:
                cached["date"] = pd.to_datetime(cached["date"])
                return cached
        if self.offline:
            raise RuntimeError(f"offline mode and {code} is not cached")

        frame: pd.DataFrame | None = None
        errors: list[str] = []

        if self.source in {"auto", "tencent"}:
            try:
                frame = self.tencent.daily(
                    code, self.cfg.start, self.cfg.end or None, self.cfg.adjust,
                    index=(market == "index"),
                )
            except TencentError as exc:
                errors.append(f"tencent: {exc}")

        if frame is None and self.source in {"auto", "sina"}:
            try:
                frame = self.sina.daily(code, index=(market == "index"))
                errors.append("fell back to sina (unadjusted, last 1023 bars)")
            except SinaError as exc:
                errors.append(f"sina: {exc}")

        if frame is None and self.source in {"auto", "eastmoney"}:
            try:
                frame = self.eastmoney.daily(
                    code, self.cfg.start, self.cfg.end or "20500101", self.cfg.adjust,
                    market=("1" if market == "index" else market),
                )
                errors.append("fell back to eastmoney")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"eastmoney: {exc}")

        if frame is None:
            raise RuntimeError(f"no bars for {code}: {'; '.join(errors)}")

        self.cache.store(key, frame)
        time.sleep(self.cfg.request_interval)
        return frame

    def index_bars(self, code: str = "000300", refresh: bool = False) -> pd.DataFrame:
        return self.bars(code, refresh=refresh, market="index")

    # ------------------------------------------------------------------- bulk
    def download(self, codes: list[str], workers: int = 6, refresh: bool = False) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        failures: list[str] = []
        names = {}
        try:
            uni = self.universe()
            names = dict(zip(uni["code"], uni["name"]))
        except Exception:  # noqa: BLE001
            pass

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(self.bars, code, refresh): code for code in codes}
            for fut in as_completed(futures):
                code = futures[fut]
                try:
                    frame = fut.result()
                    if len(frame) >= self.cfg.min_history:
                        frame = frame.copy()
                        frame["name"] = names.get(code, "")
                        out[code] = frame
                except Exception as exc:  # noqa: BLE001
                    failures.append(f"{code}: {exc}")

        if failures:
            print(f"[data] {len(failures)} symbols skipped (first 3: {failures[:3]})")
        return dict(sorted(out.items()))

    # ------------------------------------------------------------------ panel
    def panel_path(self) -> Path:
        return self.paths.cache / "panel.parquet"

    def save_panel(self, panel: pd.DataFrame) -> None:
        panel.to_parquet(self.panel_path(), index=False)

    def load_panel(self) -> pd.DataFrame | None:
        p = self.panel_path()
        if not p.exists():
            return None
        frame = pd.read_parquet(p)
        frame["date"] = pd.to_datetime(frame["date"])
        return frame
