"""Load a trained checkpoint and score a single symbol's most recent window."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..config import FeatureConfig
from ..data.features import add_features, feature_columns
from .net import CNNLSTMAttention


class Predictor:
    def __init__(self, artifact_dir: Path) -> None:
        self.artifact_dir = Path(artifact_dir)
        ckpt_path = self.artifact_dir / "cnn_lstm_attn.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"no checkpoint at {ckpt_path} - train first")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.window = int(ckpt["window"])
        self.n_features = int(ckpt["n_features"])
        self.feature_names = ckpt.get("feature_names") or feature_columns()
        self.mu = None if ckpt.get("mu") is None else np.asarray(ckpt["mu"], dtype=np.float32)
        self.sigma = None if ckpt.get("sigma") is None else np.asarray(ckpt["sigma"], dtype=np.float32)
        self.model = CNNLSTMAttention(n_features=self.n_features, window=self.window)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    def _scale(self, raw: np.ndarray) -> torch.Tensor:
        if self.mu is None or self.sigma is None:
            scaled = raw
        else:
            scaled = (raw - self.mu) / (self.sigma + 1e-6)
        scaled = np.clip(scaled, -6.0, 6.0)
        return torch.from_numpy(scaled[None, None, :, :].astype(np.float32))

    def predict_bars(self, bars: pd.DataFrame, cfg: FeatureConfig | None = None) -> dict:
        cfg = cfg or FeatureConfig()
        feat = add_features(bars, cfg)
        cols = [c for c in self.feature_names if c in feat.columns]
        if len(cols) != self.n_features:
            raise ValueError(f"checkpoint expects {self.n_features} features, frame has {len(cols)}")
        window = feat.dropna(subset=cols).tail(self.window)
        if len(window) < self.window:
            raise ValueError(f"need {self.window} usable bars, got {len(window)}")
        raw = window[cols].to_numpy(dtype=np.float32)
        tensor = self._scale(raw)
        with torch.no_grad():
            out, attn = self.model(tensor, return_attention=True)
        pred = float(out.squeeze())
        weights = attn.squeeze(0).numpy().round(4).tolist()
        return {
            "predicted_return": round(pred, 5),
            "predicted_direction": "up" if pred > 0 else "down",
            "attention_weights": weights,
            "as_of": window["date"].iloc[-1].strftime("%Y-%m-%d"),
            "last_close": round(float(window["close"].iloc[-1]), 3),
            "horizon_days": cfg.horizon,
        }

    def predict_panel(self, panel: pd.DataFrame, cfg: FeatureConfig | None = None) -> pd.DataFrame:
        """Batch-score the latest window of every symbol in a long panel."""
        cfg = cfg or FeatureConfig()
        cols = [c for c in self.feature_names if c in panel.columns]
        rows = []
        batches = []
        meta = []
        for code, grp in panel.groupby("code", sort=True):
            grp = grp.dropna(subset=cols).tail(self.window)
            if len(grp) < self.window:
                continue
            batches.append(grp[cols].to_numpy(dtype=np.float32))
            meta.append({"code": code, "as_of": grp["date"].iloc[-1], "last_close": float(grp["close"].iloc[-1])})
        if not batches:
            return pd.DataFrame(columns=["code", "predicted_return"])
        stacked = np.stack(batches)
        if self.mu is not None:
            stacked = np.clip((stacked - self.mu) / (self.sigma + 1e-6), -6.0, 6.0)
        tensor = torch.from_numpy(stacked[:, None, :, :].astype(np.float32))
        with torch.no_grad():
            preds = self.model(tensor).squeeze(-1).numpy()
        rows = [
            {**m, "predicted_return": round(float(p), 5), "predicted_direction": "up" if p > 0 else "down"}
            for m, p in zip(meta, preds)
        ]
        return pd.DataFrame(rows).sort_values("predicted_return", ascending=False).reset_index(drop=True)
