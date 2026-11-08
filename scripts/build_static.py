#!/usr/bin/env python3
"""Build the static GitHub Pages bundle into ``docs/``.

GitHub Pages serves files, not processes, so everything the page needs is precomputed
here: klines, model predictions, the equity curve, the signal report. The one thing that
cannot be precomputed is the LLM call — that happens in the visitor's browser against
their own API key.

    python scripts/build_static.py                # data only
    python scripts/build_static.py --with-llm 4   # also bake 4 real forecasts as samples

Output layout::

    docs/index.html
    docs/assets/{theme.css,charts.js,agent.js}
    docs/data/{meta,predictions,backtest,signal,llm_examples}.json
    docs/data/kline/<code>.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quantagent.backtest.engine import equal_weight_curve, run_backtest  # noqa: E402
from quantagent.backtest.metrics import annual_return, max_drawdown, summarize  # noqa: E402
from quantagent.config import load_settings  # noqa: E402
from quantagent.data.market import MarketData  # noqa: E402
from quantagent.models.inference import Predictor  # noqa: E402

OUT = ROOT / "docs"
BARS = 120


def _dump(path: Path, payload) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    path.write_text(text, encoding="utf-8")
    return len(text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", type=int, default=60, help="打包多少只股票的 K 线")
    ap.add_argument("--with-llm", type=int, default=0, help="额外烘焙几条真实大模型预测作为示例")
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default="2024-12-31")
    args = ap.parse_args()

    settings = load_settings()
    market = MarketData(settings.data, settings.paths)
    panel = market.load_panel()
    if panel is None:
        print("[static] 没有特征面板，先跑 fetch", file=sys.stderr)
        return 1

    universe = market.universe()
    names = dict(zip(universe["code"], universe["name"]))
    ranked_codes = list(universe["code"])[: args.symbols]
    if "600519" not in ranked_codes:
        ranked_codes.insert(0, "600519")

    total = 0
    # ---------------------------------------------------------------- klines
    bars_cache = {}
    for code in ranked_codes:
        try:
            bars = market.bars(code)
        except Exception as exc:  # noqa: BLE001
            print(f"[static] skip {code}: {exc}", file=sys.stderr)
            continue
        bars = bars.tail(BARS)
        bars_cache[code] = bars
        payload = {
            "code": code,
            "name": names.get(code, ""),
            "bars": [
                {
                    "date": r.date.strftime("%Y-%m-%d"),
                    "open": round(float(r.open), 2),
                    "high": round(float(r.high), 2),
                    "low": round(float(r.low), 2),
                    "close": round(float(r.close), 2),
                    "volume": float(r.volume),
                }
                for r in bars.itertuples()
            ],
        }
        total += _dump(OUT / "data" / "kline" / f"{code}.json", payload)
    print(f"[static] klines: {len(bars_cache)} symbols, {total / 1024:.0f}KB")

    # ------------------------------------------------------------ predictions
    predictor = Predictor(settings.paths.artifacts)
    scored = predictor.predict_panel(panel, settings.features)
    predictions = {
        "as_of": str(scored["as_of"].max().date()),
        "horizon_days": settings.features.horizon,
        "items": [
            {
                "code": r.code,
                "name": names.get(r.code, ""),
                "pred": round(float(r.predicted_return), 5),
                "close": round(float(r.last_close), 2),
            }
            for r in scored.itertuples()
        ],
    }
    _dump(OUT / "data" / "predictions.json", predictions)

    # --------------------------------------------------------------- backtest
    bars_dict = dict(bars_cache)
    full_dict = {}
    for code in universe["code"]:
        try:
            frame = market.bars(code)
        except Exception:  # noqa: BLE001
            continue
        if len(frame) >= 200:
            full_dict[code] = frame
    bench = None
    try:
        bench = market.index_bars("000300").set_index("date")["close"]
    except Exception:  # noqa: BLE001
        pass

    cfg = settings.backtest
    result = run_backtest(full_dict, cfg, benchmark=bench)
    stats = summarize(result, cfg.risk_free)
    ew = equal_weight_curve(full_dict, cfg.start, cfg.end)
    if len(ew) > 1:
        stats["equal_weight_annual_return"] = round(annual_return(ew), 4)
        stats["equal_weight_max_drawdown"] = round(max_drawdown(ew), 4)
        stats["excess_vs_equal_weight"] = round(stats["annual_return"] - stats["equal_weight_annual_return"], 4)

    curve = []
    for date, value in result.equity.items():
        row = {"d": date.strftime("%Y-%m-%d"), "s": round(float(value), 4)}
        if result.benchmark_equity is not None and date in result.benchmark_equity.index:
            row["b"] = round(float(result.benchmark_equity.loc[date]), 4)
        if len(ew) > 1 and date in ew.index:
            row["e"] = round(float(ew.loc[date]), 4)
        curve.append(row)
    _dump(OUT / "data" / "backtest.json", {"stats": stats, "params": asdict(cfg), "curve": curve})
    print(f"[static] backtest: {len(curve)} points, annual {stats['annual_return']:.2%}")

    # ----------------------------------------------------------------- signal
    signal_path = settings.paths.reports / "signal.json"
    if signal_path.exists():
        shutil.copyfile(signal_path, OUT / "data" / "signal.json")

    # ------------------------------------------------------------ llm samples
    examples = []
    if args.with_llm:
        from quantagent.agent.tools import AgentContext
        from quantagent.llm.forecast import forecast_batch

        if not settings.llm.api_key:
            print("[static] --with-llm 需要 QA_LLM_API_KEY，跳过", file=sys.stderr)
        else:
            ctx = AgentContext(settings)
            picks = ranked_codes[: args.with_llm]
            snaps = []
            for code in picks:
                try:
                    snaps.append(ctx.snapshot(code))
                except Exception:  # noqa: BLE001
                    continue
            if snaps:
                results = forecast_batch(snaps, settings.features.horizon, settings.llm, batch_size=len(snaps))
                for r in results:
                    examples.append({
                        "code": r["code"], "name": r.get("name", ""), "as_of": r["as_of"],
                        "direction": r["direction"], "confidence": r["confidence"],
                        "expected_return_pct": r["expected_return_pct"],
                        "drivers": r.get("drivers", []), "risk": r.get("risk", ""),
                        "model": r.get("model", ""), "usage": r.get("usage", {}),
                    })
                print(f"[static] llm examples: {len(examples)}，tokens {examples[0]['usage']}")
    examples_path = OUT / "data" / "llm_examples.json"
    if examples or not examples_path.exists():
        _dump(examples_path, {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "items": examples,
        })
    else:
        print("[static] 保留已有的 llm_examples.json（本次未传 --with-llm）")

    # ------------------------------------------------------------------- meta
    meta = {
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "n_symbols_scored": int(len(scored)),
        "n_symbols_kline": len(bars_cache),
        "data_start": str(panel["date"].min().date()),
        "data_end": str(panel["date"].max().date()),
        "as_of": predictions["as_of"],
        "model": json.loads((settings.paths.artifacts / "train_summary.json").read_text()),
        "backtest": stats,
        "features": settings.features.__dict__,
    }
    meta["model"].pop("history", None)
    _dump(OUT / "data" / "meta.json", meta)

    # --------------------------------------------------------- page + assets
    (OUT / "assets").mkdir(parents=True, exist_ok=True)
    for name in ("theme.css", "charts.js", "agent.js"):
        src = ROOT / "web" / "assets" / name
        if src.exists():
            shutil.copyfile(src, OUT / "assets" / name)
    shutil.copyfile(ROOT / "web" / "static.html", OUT / "index.html")
    (OUT / ".nojekyll").write_text("", encoding="utf-8")

    print(f"[static] 完成 → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
