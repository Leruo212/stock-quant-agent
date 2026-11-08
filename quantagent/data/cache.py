"""On-disk cache for market data.

Parquet per symbol. The cache is the reproducibility boundary: once
``data/cache/*.parquet`` exists a full pipeline run touches the network zero times,
which is what makes ``make reproduce`` meaningful. Cached data is never committed
(see .gitignore) because redistribution of vendor quotes is not ours to grant.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


def _safe(name: str) -> str:
    digest = hashlib.sha1(name.encode()).hexdigest()[:8]
    return f"{name.replace('/', '_')}_{digest}"


class ParquetCache:
    def __init__(self, root: Path, ttl_days: int = 1) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.ttl_days = ttl_days

    def path(self, key: str) -> Path:
        return self.root / f"{_safe(key)}.parquet"

    def exists(self, key: str) -> bool:
        p = self.path(key)
        if not p.exists():
            return False
        if self.ttl_days <= 0:
            return True
        age = pd.Timestamp.now() - pd.Timestamp(p.stat().st_mtime, unit="s")
        return age.days < self.ttl_days

    def load(self, key: str) -> pd.DataFrame | None:
        p = self.path(key)
        return pd.read_parquet(p) if p.exists() else None

    def store(self, key: str, frame: pd.DataFrame) -> None:
        frame.to_parquet(self.path(key), index=False)

    def keys(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.parquet"))
