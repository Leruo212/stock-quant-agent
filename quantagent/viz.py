"""Tiny SVG chart helper. No matplotlib: the figures ship in the repo as text, stay
diffable, and render on GitHub without a build step."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PALETTE = {
    "ink": "#1f2933",
    "muted": "#7b8794",
    "grid": "#e4e7eb",
    "strategy": "#b91c1c",   # 红 = 涨，A 股习惯
    "benchmark": "#94a3b8",
    "loss": "#1d4ed8",
}


def _scale(values: pd.Series | np.ndarray, lo: float, hi: float, out_lo: float, out_hi: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    span = (hi - lo) or 1.0
    return out_lo + (arr - lo) / span * (out_hi - out_lo)


def line_chart(
    series: dict[str, pd.Series],
    path: Path,
    title: str = "",
    y_label: str = "",
    width: int = 720,
    height: int = 320,
    colors: dict[str, str] | None = None,
) -> Path:
    """Multi-series line chart, x = index (dates or ints)."""
    colors = colors or {}
    pad_l, pad_r, pad_t, pad_b = 56, 16, 36, 34
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b

    all_values = pd.concat([pd.Series(v) for v in series.values()]).dropna()
    if all_values.empty:
        raise ValueError("no data to plot")
    lo, hi = float(all_values.min()), float(all_values.max())
    pad = (hi - lo) * 0.08 or 1.0
    lo, hi = lo - pad, hi + pad
    n_max = max(len(v) for v in series.values())

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
    ]
    if title:
        parts.append(f'<text x="{pad_l}" y="22" font-size="13" font-weight="600" fill="{PALETTE["ink"]}">{title}</text>')

    # y grid
    for i in range(5):
        frac = i / 4
        y = pad_t + plot_h * (1 - frac)
        value = lo + (hi - lo) * frac
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + plot_w}" y2="{y:.1f}" stroke="{PALETTE["grid"]}"/>')
        parts.append(
            f'<text x="{pad_l - 8}" y="{y + 3.5:.1f}" font-size="10" fill="{PALETTE["muted"]}" '
            f'text-anchor="end">{value:.3g}</text>'
        )
    if y_label:
        parts.append(
            f'<text x="14" y="{pad_t + plot_h / 2:.0f}" font-size="10" fill="{PALETTE["muted"]}" '
            f'transform="rotate(-90 14 {pad_t + plot_h / 2:.0f})" text-anchor="middle">{y_label}</text>'
        )

    for name, values in series.items():
        vals = pd.Series(values).dropna()
        if vals.empty:
            continue
        x = _scale(np.arange(len(vals)), 0, max(n_max - 1, 1), pad_l, pad_l + plot_w)
        y = _scale(vals.to_numpy(), lo, hi, pad_t + plot_h, pad_t)
        points = " ".join(f"{xi:.1f},{yi:.1f}" for xi, yi in zip(x, y))
        color = colors.get(name, PALETTE["loss"])
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.8" stroke-linejoin="round"/>')

    # legend
    lx = pad_l
    for i, (name, values) in enumerate(series.items()):
        color = colors.get(name, PALETTE["loss"])
        ly = height - 12
        parts.append(f'<rect x="{lx + i * 150}" y="{ly - 8}" width="10" height="3" fill="{color}"/>')
        parts.append(
            f'<text x="{lx + i * 150 + 15}" y="{ly - 3}" font-size="10" fill="{PALETTE["ink"]}">{name}</text>'
        )
    parts.append("</svg>")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def candle_chart(bars: pd.DataFrame, path: Path, title: str = "", width: int = 720, height: int = 320) -> Path:
    bars = bars.tail(120).reset_index(drop=True)
    pad_l, pad_r, pad_t, pad_b = 56, 16, 36, 34
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
    lo, hi = float(bars["low"].min()), float(bars["high"].max())
    span = (hi - lo) or 1.0
    step = plot_w / max(len(bars), 1)
    body_w = max(step * 0.6, 1.0)

    def y_of(v: float) -> float:
        return pad_t + plot_h * (1 - (v - lo) / span)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
    ]
    if title:
        parts.append(f'<text x="{pad_l}" y="22" font-size="13" font-weight="600" fill="{PALETTE["ink"]}">{title}</text>')
    for i in range(5):
        frac = i / 4
        y = pad_t + plot_h * (1 - frac)
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + plot_w}" y2="{y:.1f}" stroke="{PALETTE["grid"]}"/>')
        parts.append(
            f'<text x="{pad_l - 8}" y="{y + 3.5:.1f}" font-size="10" fill="{PALETTE["muted"]}" '
            f'text-anchor="end">{lo + span * frac:.2f}</text>'
        )
    for i, row in bars.iterrows():
        x = pad_l + step * i + step / 2
        up = row["close"] >= row["open"]
        color = PALETTE["strategy"] if up else "#0f766e"
        parts.append(f'<line x1="{x:.1f}" y1="{y_of(row["high"]):.1f}" x2="{x:.1f}" y2="{y_of(row["low"]):.1f}" stroke="{color}" stroke-width="1"/>')
        top, bottom = y_of(max(row["open"], row["close"])), y_of(min(row["open"], row["close"]))
        parts.append(
            f'<rect x="{x - body_w / 2:.1f}" y="{top:.1f}" width="{body_w:.1f}" height="{max(bottom - top, 1):.1f}" '
            f'fill="{color}" opacity="0.85"/>'
        )
    parts.append("</svg>")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")
    return path
