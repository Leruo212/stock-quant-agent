"""FastAPI service backing the bundled UI. Deliberately small: four REST endpoints and
one server-sent-events stream for the agent, no auth, no database, no queue — the
original design's Flask + Vue + PostgreSQL + RabbitMQ + Kubernetes stack was three
orders of magnitude more machinery than the workload needed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..backtest.engine import run_backtest
from ..backtest.metrics import summarize
from ..config import Settings, load_settings
from ..data.features import add_features, feature_columns
from ..data.market import MarketData
from ..models.inference import Predictor

WEB_DIR = Path(__file__).resolve().parent.parent.parent / "web"


def _clean(obj):
    """JSON-safe without NaN (NaN is not valid JSON and breaks JS parsers)."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, float):
        return None if (obj != obj or obj in (float("inf"), float("-inf"))) else round(obj, 6)
    return obj


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    market = MarketData(settings.data, settings.paths)
    app = FastAPI(title="QuantAgent", version="0.1.0", docs_url="/api/docs")

    state: dict = {"predictor": None, "agent": None}

    def get_predictor() -> Predictor:
        if state["predictor"] is None:
            try:
                state["predictor"] = Predictor(settings.paths.artifacts)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        return state["predictor"]

    def bars_for(code: str) -> pd.DataFrame:
        try:
            return market.bars(code)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=404, detail=f"no data for {code}: {exc}") from exc

    @app.get("/api/health")
    def health() -> dict:
        ckpt = (settings.paths.artifacts / "cnn_lstm_attn.pt").exists()
        return {
            "ok": True,
            "model_ready": ckpt,
            "llm_mode": "llm" if settings.llm.enabled else "local",
            "cache_dir": str(settings.paths.cache),
        }

    @app.get("/api/universe")
    def universe() -> JSONResponse:
        try:
            frame = market.universe()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return JSONResponse(_clean(frame.to_dict(orient="records")))

    @app.get("/api/kline")
    def kline(code: str = "600519", limit: int = Query(180, ge=30, le=1000)) -> JSONResponse:
        bars = bars_for(code).tail(limit)
        name = str(bars.iloc[-1].get("name", "") or "")
        if not name:
            try:
                uni = market.universe()
                hit = uni.loc[uni["code"] == str(code).zfill(6), "name"]
                name = str(hit.iloc[0]) if len(hit) else ""
            except Exception:  # noqa: BLE001 - name is cosmetic
                name = ""
        payload = {
            "code": code,
            "name": name,
            "bars": [
                {
                    "date": r["date"].strftime("%Y-%m-%d"),
                    "open": float(r["open"]), "high": float(r["high"]),
                    "low": float(r["low"]), "close": float(r["close"]),
                    "volume": float(r["volume"]),
                }
                for _, r in bars.iterrows()
            ],
        }
        return JSONResponse(_clean(payload))

    @app.get("/api/predict")
    def predict(code: str = "600519") -> JSONResponse:
        predictor = get_predictor()
        out = predictor.predict_bars(bars_for(code), settings.features)
        return JSONResponse(_clean({"code": code, **out}))

    @app.get("/api/features")
    def features(code: str = "600519") -> JSONResponse:
        bars = bars_for(code)
        feat = add_features(bars.tail(80), settings.features).dropna()
        if feat.empty:
            raise HTTPException(status_code=422, detail="not enough history for features")
        last = feat.iloc[-1]
        cols = feature_columns(settings.features)
        return JSONResponse(_clean({
            "code": code,
            "date": last["date"].strftime("%Y-%m-%d"),
            "features": {c: float(last[c]) for c in cols},
        }))

    @app.get("/api/backtest")
    def backtest(
        start: str = "2023-01-01",
        end: str = "2024-12-31",
        top_k: int = 0,
        rebalance_days: int = 7,
        signal: str = "momentum",
        limit: int = 0,
    ) -> JSONResponse:
        from ..cli import _bars_dict  # local import: cli pulls in argparse and torch

        bars = _bars_dict(market, limit)
        if not bars:
            raise HTTPException(status_code=503, detail="no cached bars, run `fetch` first")
        try:
            bench = market.index_bars("000300").set_index("date")["close"]
        except Exception:  # noqa: BLE001
            bench = None
        cfg = replace(settings.backtest, start=start, end=end, top_k=top_k,
                      rebalance_days=rebalance_days, signal=signal)
        result = run_backtest(bars, cfg, benchmark=bench)
        stats = summarize(result, cfg.risk_free)

        from ..backtest.engine import equal_weight_curve
        from ..backtest.metrics import annual_return, max_drawdown

        ew = equal_weight_curve(bars, start, end)
        if len(ew) > 1:
            stats["equal_weight_annual_return"] = round(annual_return(ew), 4)
            stats["equal_weight_max_drawdown"] = round(max_drawdown(ew), 4)
            stats["excess_vs_equal_weight"] = round(stats["annual_return"] - stats["equal_weight_annual_return"], 4)

        curve = []
        for d, v in result.equity.items():
            row = {"date": d.strftime("%Y-%m-%d"), "strategy": float(v)}
            if result.benchmark_equity is not None and d in result.benchmark_equity.index:
                row["benchmark"] = float(result.benchmark_equity.loc[d])
            if len(ew) > 1 and d in ew.index:
                row["equal_weight"] = float(ew.loc[d])
            curve.append(row)
        return JSONResponse(_clean({"stats": stats, "params": asdict(cfg), "curve": curve}))

    @app.get("/api/signal")
    def signal() -> JSONResponse:
        path = settings.paths.reports / "signal.json"
        if not path.exists():
            raise HTTPException(status_code=503, detail="no signal report, run `evaluate` first")
        return JSONResponse(_clean(json.loads(path.read_text(encoding="utf-8"))))

    @app.get("/api/agent")
    def agent_stream(q: str = Query(..., min_length=1)) -> StreamingResponse:
        from ..agent.loop import Agent

        if state["agent"] is None:
            state["agent"] = Agent(settings)

        def event_stream():
            def emit(event: dict) -> None:
                yield_queue.append(event)

            yield_queue: list[dict] = []
            # the agent runs synchronously; emit() buffers, the generator drains
            result_holder: dict = {}

            def work() -> None:
                result_holder["result"] = state["agent"].ask(q, on_event=emit)

            import threading

            worker = threading.Thread(target=work, daemon=True)
            worker.start()
            cursor = 0
            while worker.is_alive() or cursor < len(yield_queue):
                while cursor < len(yield_queue):
                    yield f"data: {json.dumps(_clean(yield_queue[cursor]), ensure_ascii=False)}\n\n"
                    cursor += 1
                worker.join(timeout=0.15)
            while cursor < len(yield_queue):  # drain the tail written at the very end
                yield f"data: {json.dumps(_clean(yield_queue[cursor]), ensure_ascii=False)}\n\n"
                cursor += 1
            final = result_holder.get("result", {})
            yield f"data: {json.dumps({'type': 'done', 'answer': final.get('answer', ''), 'mode': final.get('mode', '')}, ensure_ascii=False)}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    if WEB_DIR.exists():
        app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

    return app
