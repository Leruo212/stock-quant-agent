"""Tool registry handed to the LLM.

Every tool is a plain Python function with a JSON schema. Keeping them independent of
the agent loop means they can be unit-tested and reused by the HTTP API without an LLM
in the loop at all — which is also how ``--local`` mode works when no API key is set.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..config import BacktestConfig, Settings
from ..backtest.engine import price_matrix, run_backtest
from ..backtest.metrics import summarize
from ..data.features import add_features, feature_columns
from ..data.market import MarketData
from ..models.inference import Predictor


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    func: Callable[..., Any]

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }


class AgentContext:
    """Lazy handles for data and model, shared across one agent session."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.market = MarketData(settings.data, settings.paths)
        self._predictor: Predictor | None = None
        self._bars: dict[str, pd.DataFrame] = {}
        self._names: dict[str, str] | None = None

    # ------------------------------------------------------------------ data
    def names(self) -> dict[str, str]:
        if self._names is None:
            try:
                uni = self.market.universe()
                self._names = dict(zip(uni["code"], uni["name"]))
            except Exception:  # noqa: BLE001 - name lookup is cosmetic
                self._names = {}
        return self._names

    def bars(self, code: str, refresh: bool = False) -> pd.DataFrame:
        code = normalize_code(code)
        if refresh or code not in self._bars:
            self._bars[code] = self.market.bars(code)
        return self._bars[code]

    def all_bars(self, limit: int = 0) -> dict[str, pd.DataFrame]:
        if limit and len(self._bars) >= limit:
            return dict(list(self._bars.items())[:limit])
        uni = self.market.universe()
        codes = list(uni["code"])[: limit or None]
        fresh = {c: f for c, f in self.market.download(codes).items() if c not in self._bars}
        self._bars.update(fresh)
        return self._bars

    @property
    def predictor(self) -> Predictor:
        if self._predictor is None:
            self._predictor = Predictor(self.settings.paths.artifacts)
        return self._predictor

    # ------------------------------------------------------------------- llm
    def snapshot(self, code: str, as_of: str | None = None) -> dict:
        from ..llm.context import market_snapshot

        bars = self.bars(code)
        name = str(bars.iloc[-1].get("name", "") or self.names().get(normalize_code(code), ""))
        bench = None
        try:
            bench = self.market.index_bars("000300").set_index("date")["close"]
        except Exception:  # noqa: BLE001 - benchmark is optional context
            bench = None
        return market_snapshot(bars, code=normalize_code(code), name=name, as_of=as_of, benchmark=bench)

    def ledger(self):
        from ..llm.ledger import Ledger

        return Ledger(self.settings.paths.ledger / "predictions.jsonl")


def normalize_code(code: str) -> str:
    digits = "".join(ch for ch in str(code) if ch.isdigit())
    if len(digits) != 6:
        raise ValueError(f"invalid A-share code: {code!r}")
    return digits.zfill(6)


# --------------------------------------------------------------------------- tools
def _build_tools(ctx: AgentContext) -> list[Tool]:
    def list_symbols(query: str = "", limit: int = 15) -> dict:
        names = ctx.names()
        items = [{"code": c, "name": n} for c, n in names.items()]
        if query:
            q = str(query).strip()
            items = [i for i in items if q in i["code"] or q in i["name"]]
        return {"ok": True, "count": len(items), "items": items[:limit]}

    def get_quote(code: str) -> dict:
        try:
            bars = ctx.bars(code)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        last = bars.iloc[-1]
        prev = bars.iloc[-2] if len(bars) > 1 else last
        code = normalize_code(code)
        return {
            "ok": True,
            "code": code,
            "name": str(last.get("name", "") or ctx.names().get(code, "")),
            "date": last["date"].strftime("%Y-%m-%d"),
            "close": round(float(last["close"]), 3),
            "pct_chg": round(float(last["close"] / prev["close"] - 1) * 100, 2),
            "turnover": round(float(last["turnover"]), 3),
        }

    def predict_return(code: str) -> dict:
        try:
            bars = ctx.bars(code)
            out = ctx.predictor.predict_bars(bars, ctx.settings.features)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        code = normalize_code(code)
        out.update({"ok": True, "code": code, "name": str(bars.iloc[-1].get("name", "") or ctx.names().get(code, ""))})
        return out

    def screen_universe(top_k: int = 10, limit_symbols: int = 0) -> dict:
        try:
            from ..data.features import build_panel

            bars = ctx.all_bars(limit=limit_symbols)
            panel = build_panel(bars, ctx.settings.features)
            ranked = ctx.predictor.predict_panel(panel, ctx.settings.features)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        ranked = ranked.head(max(int(top_k), 1))
        names = ctx.names()
        return {
            "ok": True,
            "as_of": str(ranked["as_of"].max().date()) if len(ranked) else None,
            "n_scored": int(len(ranked)),
            "items": [
                {
                    "code": r["code"],
                    "name": names.get(r["code"], ""),
                    "predicted_return_pct": round(float(r["predicted_return"]) * 100, 2),
                }
                for _, r in ranked.iterrows()
            ],
        }

    def momentum_scan(days: int = 7, top_k: int = 10, limit_symbols: int = 0) -> dict:
        try:
            bars = ctx.all_bars(limit=limit_symbols)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        prices = price_matrix(bars)
        mom = (prices.iloc[-1] / prices.shift(int(days)).iloc[-1] - 1.0).dropna()
        mom = mom.sort_values(ascending=False).head(max(int(top_k), 1))
        names = ctx.names()
        return {
            "ok": True,
            "window_days": int(days),
            "as_of": str(prices.index[-1].date()),
            "items": [
                {"code": c, "name": names.get(c, ""), "return_pct": round(float(v) * 100, 2)}
                for c, v in mom.items()
            ],
        }

    def risk_profile(code: str, lookback: int = 250) -> dict:
        try:
            bars = ctx.bars(code).tail(int(lookback))
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        ret = bars["close"].pct_change().dropna()
        equity = (1 + ret).cumprod()
        peak = equity.cummax()
        dd = (equity / peak - 1).min()
        ann_vol = float(ret.std(ddof=1) * np.sqrt(252))
        return {
            "ok": True,
            "code": normalize_code(code),
            "name": str(bars.iloc[-1].get("name", "")),
            "lookback_days": int(len(ret)),
            "annual_vol": round(ann_vol, 4),
            "annual_return": round(float((equity.iloc[-1] ** (252 / len(ret))) - 1), 4) if len(ret) > 20 else None,
            "max_drawdown": round(float(dd), 4),
            "worst_day": round(float(ret.min()), 4),
            "best_day": round(float(ret.max()), 4),
            "downside_vol": round(float(ret[ret < 0].std(ddof=1) * np.sqrt(252)), 4),
        }

    def backtest_strategy(
        start: str = "2023-01-01",
        end: str = "2024-12-31",
        rebalance_days: int = 7,
        top_k: int = 0,
        stop_loss: float = 0.08,
        take_profit: float = 0.15,
        limit_symbols: int = 0,
    ) -> dict:
        try:
            bars = ctx.all_bars(limit=limit_symbols)
            base = ctx.settings.backtest
            cfg = BacktestConfig(
                start=start, end=end, rebalance_days=int(rebalance_days), top_k=int(top_k),
                stop_loss=float(stop_loss), take_profit=float(take_profit),
                cost_bps=base.cost_bps, slippage_bps=base.slippage_bps, risk_free=base.risk_free,
            )
            bench = None
            try:
                bench = ctx.market.index_bars("000300").set_index("date")["close"]
            except Exception:  # noqa: BLE001
                bench = None
            result = run_backtest(bars, cfg, benchmark=bench)
            stats = summarize(result, cfg.risk_free)
            # equal-weight buy-and-hold of the same universe: the honest baseline
            from ..backtest.engine import equal_weight_curve
            from ..backtest.metrics import annual_return as _ann

            ew = equal_weight_curve(bars, cfg.start, cfg.end)
            if len(ew) > 1:
                stats["equal_weight_annual_return"] = round(_ann(ew), 4)
                stats["excess_vs_equal_weight"] = round(stats["annual_return"] - stats["equal_weight_annual_return"], 4)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "stats": stats, "params": cfg.__dict__, "n_symbols": len(bars)}

    def llm_forecast(code: str, horizon: int = 7, record: bool = False) -> dict:
        """LLM 方向预测。与统计模型（predict_return）是两条独立路径。"""
        from ..llm.client import MockClient
        from ..llm.forecast import forecast_or_fallback

        try:
            snapshot = ctx.snapshot(code)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

        hint = None
        try:
            hint = ctx.predictor.predict_bars(ctx.bars(code), ctx.settings.features)["predicted_return"]
        except Exception:  # noqa: BLE001 - model hint is optional
            hint = None

        cfg = ctx.settings.llm
        client = MockClient(cfg) if cfg.provider == "mock" else None
        cache = None
        if cfg.use_cache and (cfg.api_key or cfg.provider == "mock"):
            from ..llm.cache import ForecastCache

            cache = ForecastCache(ctx.settings.paths.cache / "llm")
        try:
            result = forecast_or_fallback(
                snapshot, int(horizon), cfg, client=client, model_hint=hint, cache=cache
            )
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        payload = {k: v for k, v in result.items() if k != "snapshot"}
        payload["ok"] = True
        if record:
            entry_id, created = ctx.ledger().record(result, note="via agent tool")
            payload["ledger_id"] = entry_id
            payload["ledger_created"] = created
        return payload

    def llm_signal_quality() -> dict:
        """台账打分：只有已到期的预测才会被计入。"""
        try:
            ledger = ctx.ledger()
            report = ledger.score(lambda c: ctx.bars(c))
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        report["ok"] = True
        return report

    def feature_snapshot(code: str) -> dict:
        try:
            bars = ctx.bars(code)
            feat = add_features(bars.tail(60), ctx.settings.features)
            last = feat.dropna().iloc[-1]
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        cols = feature_columns(ctx.settings.features)
        return {
            "ok": True,
            "code": normalize_code(code),
            "date": last["date"].strftime("%Y-%m-%d"),
            "features": {c: round(float(last[c]), 5) for c in cols},
        }

    return [
        Tool(
            "list_symbols",
            "按代码或名称模糊搜索沪深300成分股，返回代码与名称。",
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "代码片段或名称关键字，留空返回前若干只"},
                    "limit": {"type": "integer", "description": "返回条数上限", "default": 15},
                },
            },
            list_symbols,
        ),
        Tool(
            "get_quote",
            "查询单只股票最新收盘价与涨跌幅。",
            {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "6 位股票代码，如 600519"}},
                "required": ["code"],
            },
            get_quote,
        ),
        Tool(
            "predict_return",
            "用训练好的 CNN-LSTM-Attention 模型预测该股未来 7 个交易日的相对收益，并返回注意力权重。",
            {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "6 位股票代码"}},
                "required": ["code"],
            },
            predict_return,
        ),
        Tool(
            "screen_universe",
            "对全市场（沪深300）打分，按模型预测收益排序，返回最看好的前 K 只。计算较重。",
            {
                "type": "object",
                "properties": {
                    "top_k": {"type": "integer", "description": "返回数量", "default": 10},
                    "limit_symbols": {"type": "integer", "description": "仅扫描前 N 只，0 表示全部", "default": 0},
                },
            },
            screen_universe,
        ),
        Tool(
            "momentum_scan",
            "按过去 N 个交易日累计涨幅排序，返回动量最强的股票（不使用模型）。",
            {
                "type": "object",
                "properties": {
                    "days": {"type": "integer", "default": 7},
                    "top_k": {"type": "integer", "default": 10},
                    "limit_symbols": {"type": "integer", "default": 0},
                },
            },
            momentum_scan,
        ),
        Tool(
            "risk_profile",
            "统计单只股票近一年的年化波动率、最大回撤、最差单日等风险指标。",
            {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "lookback": {"type": "integer", "default": 250},
                },
                "required": ["code"],
            },
            risk_profile,
        ),
        Tool(
            "backtest_strategy",
            "在指定区间运行动量策略回测，返回年化收益、夏普、最大回撤、胜率、成本拖累等指标。",
            {
                "type": "object",
                "properties": {
                    "start": {"type": "string", "default": "2023-01-01"},
                    "end": {"type": "string", "default": "2024-12-31"},
                    "rebalance_days": {"type": "integer", "default": 7},
                    "top_k": {"type": "integer", "default": 0},
                    "stop_loss": {"type": "number", "default": 0.08},
                    "take_profit": {"type": "number", "default": 0.15},
                    "limit_symbols": {"type": "integer", "default": 0},
                },
            },
            backtest_strategy,
        ),
        Tool(
            "feature_snapshot",
            "查看某只股票当前 9 个输入特征的取值，用于解释模型判断依据。",
            {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
            feature_snapshot,
        ),
        Tool(
            "llm_forecast",
            "把该股截止当日的数据快照交给大模型，让它判断未来 N 个交易日的方向与幅度，"
            "返回方向、置信度、预期涨跌幅、依据与风险。与 predict_return（本地统计模型）是两条独立路径。",
            {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "6 位股票代码"},
                    "horizon": {"type": "integer", "description": "预测天数，默认 7", "default": 7},
                    "record": {"type": "boolean", "description": "是否写入预测台账（供日后打分）", "default": False},
                },
                "required": ["code"],
            },
            llm_forecast,
        ),
        Tool(
            "llm_signal_quality",
            "统计预测台账里已到期预测的命中率与误差。样本不足 5 条时结论不可用。",
            {"type": "object", "properties": {}},
            llm_signal_quality,
        ),
    ]


class ToolRegistry:
    def __init__(self, ctx: AgentContext) -> None:
        self.ctx = ctx
        self.tools = {t.name: t for t in _build_tools(ctx)}

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self.tools.values()]

    def call(self, name: str, arguments: str | dict) -> dict:
        tool = self.tools.get(name)
        if tool is None:
            return {"ok": False, "error": f"unknown tool {name!r}"}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments or "{}")
            except json.JSONDecodeError as exc:
                return {"ok": False, "error": f"bad arguments: {exc}"}
        try:
            return tool.func(**arguments)
        except TypeError as exc:
            return {"ok": False, "error": f"bad arguments for {name}: {exc}"}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
