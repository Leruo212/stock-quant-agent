"""The agent loop.

Two execution modes behind one interface:

``llm``
    Real tool-calling loop against any OpenAI-compatible endpoint. The model chooses
    tools, we execute them, feed results back, repeat until it stops calling tools.

``local``
    No API key required. A deterministic intent router picks the tool(s) that a
    question obviously needs and renders the numbers into a Chinese summary. It is not
    as flexible as the LLM mode, but it means ``git clone && make demo`` produces a
    working agent with zero credentials — which matters for anyone reviewing this repo.
"""

from __future__ import annotations

import json
import re
from typing import Callable, Any

from ..config import Settings, load_settings
from .llm import LLMClient, LLMError
from .prompts import build_messages
from .tools import AgentContext, ToolRegistry, normalize_code

EventSink = Callable[[dict], None]


def _emit(sink: EventSink | None, event: dict) -> None:
    if sink is not None:
        sink(event)


# --------------------------------------------------------------------- LLM mode
def run_llm_agent(
    question: str,
    registry: ToolRegistry,
    settings: Settings,
    on_event: EventSink | None = None,
) -> dict:
    client = LLMClient(settings.llm)
    messages = build_messages(question)
    trace: list[dict] = []

    for step in range(settings.llm.max_steps):
        _emit(on_event, {"type": "thinking", "step": step + 1})
        try:
            message = client.chat(messages, tools=registry.schemas())
        except LLMError as exc:
            _emit(on_event, {"type": "error", "message": str(exc)})
            return {"ok": False, "error": str(exc), "trace": trace}

        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            answer = (message.get("content") or "").strip()
            _emit(on_event, {"type": "final", "content": answer})
            return {"ok": True, "answer": answer, "trace": trace, "mode": "llm"}

        messages.append({"role": "assistant", "content": message.get("content") or "", "tool_calls": tool_calls})
        for call in tool_calls:
            name = call["function"]["name"]
            raw_args = call["function"].get("arguments") or "{}"
            _emit(on_event, {"type": "tool_call", "name": name, "arguments": raw_args})
            result = registry.call(name, raw_args)
            trace.append({"tool": name, "arguments": raw_args, "result": result})
            _emit(on_event, {"type": "tool_result", "name": name, "result": result})
            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id", ""),
                "content": json.dumps(result, ensure_ascii=False)[:6000],
            })

    fallback = "已达到工具调用步数上限，先给到这里的结果。"
    _emit(on_event, {"type": "final", "content": fallback})
    return {"ok": True, "answer": fallback, "trace": trace, "mode": "llm", "truncated": True}


# ------------------------------------------------------------------- local mode
_CODE_RE = re.compile(r"\b(\d{6})\b")


def _pick_codes(question: str, registry: ToolRegistry) -> list[str]:
    codes = [normalize_code(c) for c in _CODE_RE.findall(question)]
    if codes:
        return codes
    # fall back to name matching against the universe
    try:
        hits = registry.call("list_symbols", {"query": question.strip(), "limit": 3})
        if hits.get("ok"):
            return [item["code"] for item in hits.get("items", [])]
    except Exception:  # noqa: BLE001
        pass
    return []


def _fmt_pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:+.2f}%"


def run_local_agent(question: str, registry: ToolRegistry, on_event: EventSink | None = None) -> dict:
    q = question.strip()
    trace: list[dict] = []

    def call(name: str, args: dict) -> dict:
        _emit(on_event, {"type": "tool_call", "name": name, "arguments": json.dumps(args, ensure_ascii=False)})
        result = registry.call(name, args)
        trace.append({"tool": name, "arguments": args, "result": result})
        _emit(on_event, {"type": "tool_result", "name": name, "result": result})
        return result

    codes = _pick_codes(q, registry)
    lines: list[str] = []

    wants_backtest = any(k in q for k in ("回测", "策略", "夏普", "回撤", "收益"))
    wants_screen = any(k in q for k in ("选股", "看好", "推荐", "哪些", "排行", "打分"))
    wants_risk = any(k in q for k in ("风险", "波动", "回撤"))
    wants_momentum = any(k in q for k in ("动量", "涨幅", "涨得"))

    if wants_backtest and not codes:
        res = call("backtest_strategy", {})
        if res.get("ok"):
            s = res["stats"]
            lines.append(
                f"动量策略在 {s['n_days']} 个交易日内年化 {_fmt_pct(s['annual_return'])}、"
                f"夏普 {s['sharpe']}、最大回撤 {_fmt_pct(s['max_drawdown'])}、"
                f"日胜率 {s['win_rate_daily'] * 100:.1f}%、"
                f"年均换手 {s['turnover_per_year']}。"
            )
            if "benchmark_annual_return" in s:
                lines.append(
                    f"同期沪深300 年化 {_fmt_pct(s['benchmark_annual_return'])}，"
                    f"超额 {_fmt_pct(s['excess_annual_return'])}；成本拖累 {s['cost_drag'] * 100:.2f}%。"
                )
            if "equal_weight_annual_return" in s:
                lines.append(
                    f"但同一批股票等权买入不动是 {_fmt_pct(s['equal_weight_annual_return'])}，"
                    f"策略相对它 {_fmt_pct(s['excess_vs_equal_weight'])}——"
                    f"股票池是今天的市值前 300，跟指数比会系统性高估策略。"
                )
        else:
            lines.append(f"回测失败：{res.get('error')}")

    if wants_screen or (not codes and not wants_backtest):
        res = call("screen_universe", {"top_k": 8})
        if res.get("ok") and res.get("items"):
            top = "、".join(f"{i['name']}({i['code']}) {i['predicted_return_pct']:+.2f}%" for i in res["items"][:5])
            lines.append(f"模型在全市场打分里排前 5 的是：{top}（截至 {res.get('as_of')}）。")
        elif not res.get("ok"):
            lines.append(f"全市场打分失败：{res.get('error')}")

    if wants_momentum:
        res = call("momentum_scan", {"days": 7, "top_k": 5})
        if res.get("ok") and res.get("items"):
            top = "、".join(f"{i['name']}({i['code']}) {i['return_pct']:+.2f}%" for i in res["items"][:5])
            lines.append(f"过去 7 个交易日动量最强的 5 只是：{top}。")

    for code in codes[:2]:
        quote = call("get_quote", {"code": code})
        if quote.get("ok"):
            lines.append(f"{quote['name']}({code}) 最新收盘 {quote['close']}，日涨跌 {quote['pct_chg']:+.2f}%。")
        else:
            lines.append(f"{code} 行情获取失败：{quote.get('error')}")
            continue

        if wants_risk:
            risk = call("risk_profile", {"code": code})
            if risk.get("ok"):
                lines.append(
                    f"风险画像：年化波动 {_fmt_pct(risk['annual_vol'])}、最大回撤 {_fmt_pct(risk['max_drawdown'])}、"
                    f"下行波动 {_fmt_pct(risk['downside_vol'])}（近 {risk['lookback_days']} 个交易日）。"
                )
        else:
            pred = call("predict_return", {"code": code})
            if pred.get("ok"):
                weights = pred["attention_weights"]
                spread = max(weights) - min(weights) if weights else 0.0
                if spread < 0.01:
                    attention_note = "注意力权重接近均匀，模型对哪一天都没有偏好"
                else:
                    peak = int(max(range(len(weights)), key=lambda i: weights[i])) + 1
                    attention_note = f"注意力最集中在第 {peak} 天"
                lines.append(
                    f"模型对截至 {pred['as_of']} 的窗口给出未来 {pred['horizon_days']} 日相对收益 "
                    f"{_fmt_pct(pred['predicted_return'])}，{attention_note}。"
                )
            else:
                lines.append(f"模型预测失败：{pred.get('error')}")

    if not lines:
        lines.append(
            "我可以做这几件事：查行情（「看下 600519」）、模型预测（「预测 000001 未来一周」）、"
            "全市场选股（「今天看好哪些」）、风险画像（「000858 风险怎么样」）、策略回测（「跑一次回测」）。"
        )

    lines.append("以上为模型输出与历史统计，不构成投资建议。")
    answer = "\n".join(lines)
    _emit(on_event, {"type": "final", "content": answer})
    return {"ok": True, "answer": answer, "trace": trace, "mode": "local"}


# ------------------------------------------------------------------- entrypoint
class Agent:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        self.context = AgentContext(self.settings)
        self.registry = ToolRegistry(self.context)

    def ask(self, question: str, on_event: EventSink | None = None) -> dict[str, Any]:
        if self.settings.llm.enabled:
            try:
                return run_llm_agent(question, self.registry, self.settings, on_event)
            except Exception as exc:  # noqa: BLE001 - degrade instead of 500
                _emit(on_event, {"type": "error", "message": f"LLM 模式失败，回退本地规划器：{exc}"})
        _emit(on_event, {"type": "mode", "mode": "local"})
        return run_local_agent(question, self.registry, on_event)
