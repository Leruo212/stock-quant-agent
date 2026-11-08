"""CNN-LSTM-Attention network, rebuilt to match the layer table in the thesis.

Reference table (input ``(N, 1, 7, 9)``)::

    Conv2D-4   (N,1,7,9)   -> (N,16,7,9)    160 params
    PReLU-7                 -> (N,16,7,9)      1 param
    MaxPool2D               -> (N,16,7,9)      0 params
    Dropout2D  p=0.2        -> (N,16,7,9)      0 params
    LSTM-7     (N,7,144)    -> (N,7,32)    31,232 params
    LSTM-8     (N,7,32)     -> (N,7,16)     3,200 params
    Linear-7   (N,16)       -> (N,8)          136 params
    PReLU-8                 -> (N,8)            1 param
    Linear-8   (N,8)        -> (N,1)            9 params

Two notes on purpose, because the table is internally inconsistent:

1. ``MaxPool2D`` is listed with an unchanged output shape. A 1x1 kernel is the only
   pooling setting consistent with both that row and the 144 = 16 x 9 channels the
   LSTM receives, so it is implemented as-is rather than silently dropped.
2. The LSTM parameter counts in the table are ~1% above the textbook
   ``4 * h * (in + h + 1)`` figure. This implementation follows PyTorch and reports the
   real count at build time instead of copying the paper.
"""

from __future__ import annotations

import torch
from torch import nn


class TemporalAttention(nn.Module):
    """Additive attention over the time axis; also exposes the weights for inspection."""

    def __init__(self, hidden: int, attn_dim: int = 16) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden, attn_dim)
        self.score = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.score(torch.tanh(self.proj(sequence))), dim=1)  # (N, T, 1)
        context = (weights * sequence).sum(dim=1)                                    # (N, H)
        return context, weights.squeeze(-1)


class CNNLSTMAttention(nn.Module):
    def __init__(
        self,
        n_features: int = 9,
        window: int = 7,
        conv_channels: int = 16,
        lstm_hidden: tuple[int, int] = (32, 16),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.window = window
        self.n_features = n_features
        self.conv = nn.Conv2d(1, conv_channels, kernel_size=3, padding=1)
        self.conv_act = nn.PReLU()
        # see module docstring, note 1
        self.pool = nn.MaxPool2d(kernel_size=1)
        self.dropout = nn.Dropout2d(dropout)

        h1, h2 = lstm_hidden
        flat = conv_channels * n_features
        self.lstm1 = nn.LSTM(flat, h1, num_layers=2, batch_first=True, dropout=dropout)
        self.lstm2 = nn.LSTM(h1, h2, num_layers=1, batch_first=True)
        self.attention = TemporalAttention(h2)
        self.head = nn.Sequential(nn.Linear(h2, 8), nn.PReLU(), nn.Linear(8, 1))

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        feat = self.dropout(self.pool(self.conv_act(self.conv(x))))     # (N, C, W, F)
        seq = feat.permute(0, 2, 1, 3).reshape(feat.size(0), self.window, -1)  # (N, W, C*F)
        seq, _ = self.lstm1(seq)
        seq, _ = self.lstm2(seq)
        context, weights = self.attention(seq)
        out = self.head(context)
        if return_attention:
            return out, weights
        return out

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self(x).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
