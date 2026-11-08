"""Training loop: MSE on forward relative return, Adam + cosine annealing.

Deviations from the thesis, stated up front:

* initial LR 1e-3 instead of 0.1. With MSE on returns of order 1e-2, an LR of 0.1
  diverges within a handful of steps on every seed we tried.
* early stopping on validation loss (patience in ``TrainConfig``), because the thesis
  fixes 100 epochs and reports only the training loss.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..config import TrainConfig
from ..data.dataset import WindowedData
from .net import CNNLSTMAttention, count_parameters


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def evaluate(model: nn.Module, loader: DataLoader, loss_fn) -> tuple[float, np.ndarray]:
    model.eval()
    total, n, preds = 0.0, 0, []
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb)
            total += loss_fn(out, yb).item() * len(xb)
            n += len(xb)
            preds.append(out.numpy())
    return total / max(n, 1), np.concatenate(preds).ravel()


def train_model(
    data: WindowedData,
    cfg: TrainConfig | None = None,
    artifact_dir: Path | None = None,
    verbose: bool = True,
) -> tuple[CNNLSTMAttention, dict]:
    cfg = cfg or TrainConfig()
    set_seed(cfg.seed)
    if cfg.threads:
        torch.set_num_threads(cfg.threads)

    _, _, window, n_features = data.x_train.shape
    model = CNNLSTMAttention(n_features=n_features, window=window, dropout=cfg.dropout)
    total_params = count_parameters(model)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.01)

    train_loader = _loader(data.x_train, data.y_train, cfg.batch_size, shuffle=True)
    val_loader = _loader(data.x_val, data.y_val, cfg.batch_size, shuffle=False)

    history: list[dict] = []
    best_loss, best_state, stale = float("inf"), None, 0
    started = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item() * len(xb)
            seen += len(xb)
        train_loss = running / max(seen, 1)
        val_loss, _ = evaluate(model, val_loader, loss_fn)
        scheduler.step()

        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
             "lr": optimizer.param_groups[0]["lr"]}
        )
        if verbose and (epoch == 1 or epoch % 5 == 0):
            print(f"[train] epoch {epoch:3d}  train {train_loss:.6f}  val {val_loss:.6f}  "
                  f"lr {optimizer.param_groups[0]['lr']:.2e}")

        if val_loss < best_loss - 1e-6:
            best_loss, stale = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= cfg.patience:
                if verbose:
                    print(f"[train] early stop at epoch {epoch} (best val {best_loss:.6f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_loss, val_pred = evaluate(model, val_loader, loss_fn)
    pred_dir = float(np.mean(np.sign(val_pred) == np.sign(data.y_val.ravel())))
    summary = {
        "params": total_params,
        "input_shape": list(data.x_train.shape[1:]),
        "n_train": int(len(data.x_train)),
        "n_val": int(len(data.x_val)),
        "best_val_mse": round(best_loss, 8),
        "val_mae": round(float(np.mean(np.abs(val_pred - data.y_val.ravel()))), 6),
        "val_direction_acc": round(pred_dir, 4),
        "val_corr": round(float(np.corrcoef(val_pred, data.y_val.ravel())[0, 1]), 4),
        "epochs_run": len(history),
        "seconds": round(time.time() - started, 1),
        "hparams": asdict(cfg),
        "history": history,
    }

    if artifact_dir is not None:
        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "n_features": n_features,
                "window": window,
                "feature_names": data.feature_names,
                "mu": None if data.mu is None else data.mu.ravel(),
                "sigma": None if data.sigma is None else data.sigma.ravel(),
                "split_date": str(data.split_date),
            },
            artifact_dir / "cnn_lstm_attn.pt",
        )
        (artifact_dir / "train_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        np.savez_compressed(
            artifact_dir / "val_predictions.npz",
            pred=val_pred, target=data.y_val.ravel(), date=data.val_dates,
        )

    return model, summary
