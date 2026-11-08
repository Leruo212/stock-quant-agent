"""Single entry point: ``python -m quantagent.cli <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest.engine import run_backtest
from .backtest.metrics import summarize
from .backtest.optimize import optimize
from .config import FeatureConfig, TrainConfig, load_settings
from .data.features import build_panel, feature_columns
from .data.market import MarketData
from .models.inference import Predictor
from .models.train import train_model
from .viz import candle_chart, line_chart


def _settings():
    return load_settings()


# --------------------------------------------------------------------- commands
def cmd_fetch(args: argparse.Namespace) -> int:
    s = _settings()
    market = MarketData(s.data, s.paths)
    uni = market.universe(refresh=args.refresh)
    codes = list(uni["code"])
    if args.limit:
        codes = codes[: args.limit]
    if args.codes:
        codes = [c.strip().zfill(6) for c in args.codes.split(",")]
    uni.to_parquet(s.paths.cache / "universe.parquet", index=False)
    print(f"[fetch] {len(codes)} symbols, {s.data.start} -> today")
    bars = market.download(codes, workers=args.workers, refresh=args.refresh)
    print(f"[fetch] usable: {len(bars)}")
    if not bars:
        print("[fetch] nothing downloaded", file=sys.stderr)
        return 1

    panel = build_panel(bars, s.features)
    market.save_panel(panel)
    span = f"{panel['date'].min().date()} .. {panel['date'].max().date()}"
    print(f"[fetch] panel: {len(panel):,} rows, {panel['code'].nunique()} symbols, {span}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from .data.dataset import make_windows

    s = _settings()
    if args.feature_set:
        s.features.feature_set = args.feature_set
    if getattr(args, "label_mode", None):
        s.features.label_mode = args.label_mode
    market = MarketData(s.data, s.paths)
    panel = market.load_panel()
    if panel is None:
        print("[train] no panel - run `fetch` first", file=sys.stderr)
        return 1

    data = make_windows(
        panel, s.features,
        train_ratio=s.train.train_ratio,
        max_per_symbol=args.max_per_symbol,
        seed=s.train.seed,
    )
    print(f"[train] windows: {len(data.x_train):,} train / {len(data.x_val):,} val, "
          f"split {data.split_date.date()}, features {feature_columns(s.features)}")

    cfg = replace(s.train, epochs=args.epochs, batch_size=args.batch_size) if (args.epochs or args.batch_size) else s.train
    _, summary = train_model(data, cfg, artifact_dir=s.paths.artifacts)

    print(f"[train] params {summary['params']:,} | val MSE {summary['best_val_mse']:.6f} | "
          f"direction acc {summary['val_direction_acc']:.3f} | corr {summary['val_corr']:.3f}")
    print(f"[train] artifacts -> {s.paths.artifacts}")
    return 0


def _bars_dict(market: MarketData, limit: int = 0) -> dict[str, pd.DataFrame]:
    uni_path = market.paths.cache / "universe.parquet"
    if uni_path.exists():
        codes = list(pd.read_parquet(uni_path)["code"])
    else:
        codes = list(market.universe()["code"])
    if limit:
        codes = codes[:limit]
    out: dict[str, pd.DataFrame] = {}
    for code in codes:
        try:
            df = market.bars(code)
        except Exception:  # noqa: BLE001 - not cached in offline mode
            continue
        if len(df) >= 200:
            out[code] = df
    return out


def cmd_backtest(args: argparse.Namespace) -> int:
    s = _settings()
    market = MarketData(s.data, s.paths)
    bars = _bars_dict(market, args.limit)
    if not bars:
        print("[backtest] no cached bars - run `fetch` first", file=sys.stderr)
        return 1
    try:
        bench = market.index_bars("000300").set_index("date")["close"]
    except Exception:  # noqa: BLE001
        bench = None

    cfg = replace(
        s.backtest,
        start=args.start, end=args.end, top_k=args.top_k,
        rebalance_days=args.rebalance_days, signal=args.signal,
    )
    if args.no_cost:
        cfg = replace(cfg, cost_bps=0.0, slippage_bps=0.0)

    result = run_backtest(bars, cfg, benchmark=bench, lag=args.lag)
    stats = summarize(result, cfg.risk_free)
    stats["n_symbols"] = len(bars)

    from .backtest.engine import equal_weight_curve
    from .backtest.metrics import annual_return, max_drawdown

    ew = equal_weight_curve(bars, cfg.start, cfg.end)
    if len(ew) > 1:
        stats["equal_weight_annual_return"] = round(annual_return(ew), 4)
        stats["equal_weight_max_drawdown"] = round(max_drawdown(ew), 4)
        stats["excess_vs_equal_weight"] = round(stats["annual_return"] - stats["equal_weight_annual_return"], 4)

    print(json.dumps(stats, indent=2, ensure_ascii=False))
    out = s.paths.reports
    out.mkdir(parents=True, exist_ok=True)
    (out / "backtest.json").write_text(json.dumps({"stats": stats, "params": asdict(cfg)}, indent=2, ensure_ascii=False))
    result.equity.to_frame("strategy").to_csv(out / "equity.csv")
    if not result.trades.empty:
        result.trades.to_csv(out / "trades.csv", index=False)

    series = {"策略净值": result.equity}
    colors = {"策略净值": "#b91c1c"}
    if result.benchmark_equity is not None:
        series["沪深300"] = result.benchmark_equity
        colors["沪深300"] = "#94a3b8"
    if len(ew) > 1:
        series["等权持有"] = ew
        colors["等权持有"] = "#1d4ed8"
    line_chart(series, out / "equity_curve.svg", title=f"动量策略净值 {cfg.start} – {cfg.end}", colors=colors)
    drawdown = result.equity / result.equity.cummax() - 1
    line_chart({"回撤": drawdown}, out / "drawdown.svg", title="回撤曲线", colors={"回撤": "#0f766e"})
    print(f"[backtest] reports -> {out}")
    return 0


def cmd_tune(args: argparse.Namespace) -> int:
    s = _settings()
    market = MarketData(s.data, s.paths)
    bars = _bars_dict(market, args.limit)
    if not bars:
        print("[tune] no cached bars - run `fetch` first", file=sys.stderr)
        return 1
    try:
        bench = market.index_bars("000300").set_index("date")["close"]
    except Exception:  # noqa: BLE001
        bench = None
    report = optimize(bars, s.backtest, n_trials=args.trials, benchmark=bench, out_dir=s.paths.reports)
    print(json.dumps({k: report[k] for k in ("n_trials", "best_value", "best_params", "param_importance")},
                     indent=2, ensure_ascii=False))
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    s = _settings()
    market = MarketData(s.data, s.paths)
    predictor = Predictor(s.paths.artifacts)
    if args.all:
        panel = market.load_panel()
        if panel is None:
            print("[predict] no panel", file=sys.stderr)
            return 1
        ranked = predictor.predict_panel(panel, s.features).head(args.top_k)
        print(ranked.to_string(index=False))
    else:
        out = predictor.predict_bars(market.bars(args.code), s.features)
        print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


def cmd_agent(args: argparse.Namespace) -> int:
    from .agent.loop import Agent

    agent = Agent(_settings())
    if args.question:
        result = agent.ask(args.question, on_event=_print_event)
        print("\n" + result["answer"])
        return 0

    print("QuantAgent 交互模式，输入 exit 退出。")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"exit", "quit", ":q"}:
            break
        if not question:
            continue
        result = agent.ask(question, on_event=_print_event)
        print("\n" + result["answer"])
    return 0


def _print_event(event: dict) -> None:
    kind = event.get("type")
    if kind == "tool_call":
        print(f"  → {event['name']}({event['arguments']})")
    elif kind == "tool_result":
        ok = event["result"].get("ok", False)
        print(f"  ← {event['name']}: {'ok' if ok else event['result'].get('error')}")
    elif kind == "error":
        print(f"  ! {event['message']}")
    elif kind == "mode":
        print(f"  · mode={event['mode']}")


def cmd_evaluate(args: argparse.Namespace) -> int:
    from .baseline import ridge_report
    from .data.dataset import make_windows
    from .data.market import MarketData
    from .evaluate import regime_split, signal_report

    s = _settings()
    npz = s.paths.artifacts / "val_predictions.npz"
    if not npz.exists():
        print("[evaluate] no validation predictions - run `train` first", file=sys.stderr)
        return 1
    blob = np.load(npz, allow_pickle=True)
    report = signal_report(blob["pred"], blob["target"], blob["date"])
    if not report.get("degenerate"):
        report.update(regime_split(blob["pred"], blob["target"], blob["date"]))

    summary_path = s.paths.artifacts / "train_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        report["model"] = {k: summary.get(k) for k in ("params", "n_train", "n_val", "best_val_mse", "epochs_run")}

    # same features, linear model: is the network the problem, or the information?
    try:
        market = MarketData(s.data, s.paths)
        panel = market.load_panel()
        if panel is not None:
            data = make_windows(panel, s.features, train_ratio=s.train.train_ratio,
                                max_per_symbol=args.max_per_symbol, seed=s.train.seed)
            report["ridge_baseline"] = ridge_report(
                data.x_train, data.y_train, data.x_val, data.y_val, data.val_dates
            )
    except Exception as exc:  # noqa: BLE001 - baseline is informative, not required
        report["ridge_baseline"] = {"error": str(exc)}

    print(json.dumps(report, indent=2, ensure_ascii=False))
    out = s.paths.reports / "signal.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    hist = json.loads(summary_path.read_text())["history"] if summary_path.exists() else []
    if hist:
        line_chart(
            {"训练误差": pd.Series([h["train_loss"] for h in hist]),
             "验证误差": pd.Series([h["val_loss"] for h in hist])},
            s.paths.reports / "training_curve.svg",
            title="训练 / 验证 MSE", y_label="MSE",
            colors={"训练误差": "#1d4ed8", "验证误差": "#b91c1c"},
        )
    print(f"[evaluate] -> {out}")
    return 0


def cmd_forecast(args: argparse.Namespace) -> int:
    from .agent.tools import AgentContext
    from .llm.client import MockClient
    from .llm.forecast import forecast_or_fallback

    s = _settings()
    if args.provider:
        s.llm.provider = args.provider
    if args.samples:
        s.llm.n_samples = args.samples

    codes = [c.strip().zfill(6) for c in (args.codes or args.code).split(",") if c.strip()]
    if not codes:
        print("[forecast] 需要 --code 或 --codes", file=sys.stderr)
        return 1

    ctx = AgentContext(s)
    client = MockClient(s.llm) if s.llm.provider == "mock" else None
    if s.llm.n_samples > 1 and not s.llm.api_key and s.llm.provider != "mock":
        print("[forecast] 未配置 QA_LLM_API_KEY，--samples 对启发式占位无意义", file=sys.stderr)

    cache = None
    if s.llm.use_cache and not args.no_cache and (s.llm.api_key or s.llm.provider == "mock"):
        from .llm.cache import ForecastCache

        cache = ForecastCache(s.paths.cache / "llm")

    snapshots: list[dict] = []
    hints: dict[str, float] = {}
    for code in codes:
        try:
            snapshots.append(ctx.snapshot(code, as_of=args.as_of))
        except Exception as exc:  # noqa: BLE001
            print(f"[forecast] {code}: {exc}", file=sys.stderr)
            continue
        if not args.no_model_hint:
            try:
                hints[code] = ctx.predictor.predict_bars(ctx.bars(code), s.features)["predicted_return"]
            except Exception:  # noqa: BLE001
                pass

    if not snapshots:
        return 1

    if args.batch > 1 and len(snapshots) > 1:
        from .llm.forecast import forecast_batch

        results = forecast_batch(
            snapshots, args.horizon, s.llm, client=client,
            batch_size=args.batch, cache=cache, model_hint_map=hints,
        )
        usage = (results[0].get("usage") if results else {}) or {}
        if usage.get("prompt_tokens"):
            total = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
            per_symbol = total / max(len(snapshots), 1)
            single = 800  # measured one-symbol calls: 773-848 total tokens
            print(
                f"[forecast] 批量 {len(snapshots)} 只 / 每批 {args.batch} 只："
                f"合计 {total} tokens，平均 {per_symbol:.0f}/只；"
                f"单只调用实测 770–850/只，节省约 {max(0, (1 - per_symbol / single)) * 100:.0f}%",
                file=sys.stderr,
            )
        else:
            print(f"[forecast] 批量模式：{len(snapshots)} 只 / 每批 {args.batch} 只", file=sys.stderr)
    else:
        results = []
        for snapshot in snapshots:
            try:
                results.append(forecast_or_fallback(
                    snapshot, args.horizon, s.llm,
                    client=client, n_samples=s.llm.n_samples,
                    reflect=args.reflect, model_hint=hints.get(str(snapshot.get("code", ""))),
                    cache=cache,
                ))
            except Exception as exc:  # noqa: BLE001
                print(f"[forecast] {snapshot.get('code')}: {exc}", file=sys.stderr)

    for result in results:
        code = result.get("code", "")
        if args.record:
            entry_id, created = ctx.ledger().record(result, note=args.note)
            result["ledger_id"] = entry_id
            result["ledger_created"] = created

        if args.json:
            print(json.dumps({k: v for k, v in result.items() if k != "snapshot"}, indent=2, ensure_ascii=False))
            continue

        ens = result.get("ensemble", {})
        print(f"\n{result.get('name', '')}({code})  截至 {result.get('as_of', '')}  "
              f"[{result.get('provider')}/{result.get('model')}]"
              + ("  · 缓存命中" if result.get("cached") else ""))
        print(f"  方向 {result.get('direction')}  置信 {result.get('confidence', 0):.2f}  "
              f"预期 {result.get('expected_return_pct', 0):+.2f}%  ({result.get('horizon_days')} 日)")
        if ens.get("n", 1) > 1:
            print(f"  {ens['n']} 次采样：方向一致率 {ens['direction_agreement']:.0%}，幅度极差 {ens['return_range_pct']}pp")
        for driver in result.get("drivers", []):
            print(f"  · {driver}")
        if result.get("risk"):
            print(f"  风险：{result['risk']}")
        if result.get("reflection", {}).get("changed_fields"):
            print(f"  反思修正了：{', '.join(result['reflection']['changed_fields'])}")
        if result.get("ledger_id"):
            state = "已写入" if result.get("ledger_created") else "已存在"
            print(f"  台账 {state}：{result['ledger_id']}")

    return 0


def cmd_ledger(args: argparse.Namespace) -> int:
    from .agent.tools import AgentContext
    from .llm.ledger import Ledger

    s = _settings()
    ledger = Ledger(s.paths.ledger / "predictions.jsonl")
    entries = ledger.entries()

    if args.score:
        ctx = AgentContext(s)
        report = ledger.score(lambda c: ctx.bars(c), min_entries=args.min_entries)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    if not entries:
        print("[ledger] 空台账。用 `forecast --record` 记录第一条预测。")
        return 0
    print(f"{'as_of':<12}{'code':<8}{'方向':<6}{'预期':>9}{'置信':>7}  {'模型':<18}{'provider':<10}")
    for e in entries[-args.limit:]:
        print(f"{e.get('as_of',''):<12}{e.get('code',''):<8}{e.get('direction',''):<6}"
              f"{e.get('expected_return_pct',0):+7.2f}%{e.get('confidence',0):7.2f}  "
              f"{str(e.get('model',''))[:16]:<18}{e.get('provider',''):<10}")
    print(f"\n共 {len(entries)} 条。评分：python -m quantagent.cli ledger --score")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .api.server import create_app

    app = create_app(_settings())
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def cmd_pipeline(args: argparse.Namespace) -> int:
    """fetch -> train -> backtest, the one command a reviewer runs."""
    for step, fn in (
        ("fetch", cmd_fetch),
        ("train", cmd_train),
        ("backtest", cmd_backtest),
    ):
        print(f"\n=== {step} ===")
        code = fn(argparse.Namespace(
            refresh=False, limit=args.limit, codes=None, workers=6,
            epochs=args.epochs, batch_size=None, max_per_symbol=args.max_per_symbol, feature_set=None,
            start=args.start, end=args.end, top_k=0, rebalance_days=7, no_cost=False, lag=0,
        ))
        if code != 0:
            return code
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="quantagent", description="CNN-LSTM-Attention A-share quant stack")
    sub = p.add_subparsers(dest="command", required=True)

    f = sub.add_parser("fetch", help="下载行情并生成特征面板")
    f.add_argument("--limit", type=int, default=0, help="只取前 N 只（0=全部）")
    f.add_argument("--codes", type=str, default="", help="逗号分隔的指定代码")
    f.add_argument("--workers", type=int, default=6)
    f.add_argument("--refresh", action="store_true")
    f.set_defaults(func=cmd_fetch)

    t = sub.add_parser("train", help="训练 CNN-LSTM-Attention")
    t.add_argument("--epochs", type=int, default=0)
    t.add_argument("--batch-size", type=int, default=0)
    t.add_argument("--max-per-symbol", type=int, default=200, help="每只股票最多取多少窗口")
    t.add_argument("--feature-set", choices=["stationary9", "raw9"], default=None)
    t.add_argument("--label-mode", choices=["raw", "rank"], default=None,
                   help="raw=预测相对收益, rank=预测横截面排序")
    t.set_defaults(func=cmd_train)

    b = sub.add_parser("backtest", help="运行动量策略回测")
    b.add_argument("--start", default="2023-01-01")
    b.add_argument("--end", default="2024-12-31")
    b.add_argument("--top-k", type=int, default=0)
    b.add_argument("--rebalance-days", type=int, default=7)
    b.add_argument("--signal", choices=["momentum", "reversal"], default="momentum",
                   help="momentum=买涨得多的, reversal=买跌得多的")
    b.add_argument("--limit", type=int, default=0)
    b.add_argument("--no-cost", action="store_true", help="关闭手续费与滑点，用于看毛收益")
    b.add_argument("--lag", type=int, default=0)
    b.set_defaults(func=cmd_backtest)

    o = sub.add_parser("tune", help="Optuna 参数搜索")
    o.add_argument("--trials", type=int, default=40)
    o.add_argument("--limit", type=int, default=0)
    o.set_defaults(func=cmd_tune)

    ev = sub.add_parser("evaluate", help="评估模型信号质量（IC、分层收益、线性基线）")
    ev.add_argument("--max-per-symbol", type=int, default=200)
    ev.set_defaults(func=cmd_evaluate)

    fc = sub.add_parser("forecast", help="调用大模型做方向预测")
    fc.add_argument("--code", default="600519")
    fc.add_argument("--codes", default="", help="逗号分隔，批量预测")
    fc.add_argument("--horizon", type=int, default=7)
    fc.add_argument("--as-of", default=None, help="只用该日期及之前的数据（默认最新）")
    fc.add_argument("--samples", type=int, default=0, help="自一致性采样次数，>1 可估计不确定性")
    fc.add_argument("--reflect", action="store_true", help="增加一轮自我审查（多一次调用）")
    fc.add_argument("--record", action="store_true", help="写入预测台账")
    fc.add_argument("--provider", choices=["auto", "openai", "mock"], default=None)
    fc.add_argument("--no-model-hint", action="store_true", help="不把统计模型结果作为参考给大模型")
    fc.add_argument("--no-cache", action="store_true", help="跳过结果缓存，强制真实调用")
    fc.add_argument("--batch", type=int, default=1, help="每批几只股票（>1 时共用一个提示词，省 token）")
    fc.add_argument("--json", action="store_true")
    fc.add_argument("--note", default="")
    fc.set_defaults(func=cmd_forecast)

    lg = sub.add_parser("ledger", help="查看 / 打分预测台账")
    lg.add_argument("--score", action="store_true")
    lg.add_argument("--limit", type=int, default=20)
    lg.add_argument("--min-entries", type=int, default=5)
    lg.set_defaults(func=cmd_ledger)

    pr = sub.add_parser("predict", help="模型推理")
    pr.add_argument("--code", default="600519")
    pr.add_argument("--all", action="store_true")
    pr.add_argument("--top-k", type=int, default=20)
    pr.set_defaults(func=cmd_predict)

    a = sub.add_parser("agent", help="与 QuantAgent 对话")
    a.add_argument("question", nargs="?", default="")
    a.set_defaults(func=cmd_agent)

    sv = sub.add_parser("serve", help="启动 Web 服务")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.set_defaults(func=cmd_serve)

    pl = sub.add_parser("pipeline", help="fetch -> train -> backtest 一条龙")
    pl.add_argument("--limit", type=int, default=120)
    pl.add_argument("--epochs", type=int, default=60)
    pl.add_argument("--max-per-symbol", type=int, default=200)
    pl.add_argument("--start", default="2023-01-01")
    pl.add_argument("--end", default="2024-12-31")
    pl.set_defaults(func=cmd_pipeline)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
