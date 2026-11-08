/* Browser-side agent with bring-your-own-key.
 *
 * GitHub Pages cannot run a server, so the LLM call happens here, in the visitor's
 * browser, against the visitor's own key. The key is kept in localStorage and sent only
 * to the endpoint they configured — there is no intermediary and no analytics.
 *
 * Honest trade-off, stated in the UI too: anything running on this page could read
 * localStorage. That is why the key is stored under a clearly-named key, why there is a
 * one-click wipe, and why the server-served version (python -m quantagent.cli serve)
 * keeps the key out of the browser entirely.
 *
 * Tools mirror the Python registry, but read from the precomputed static bundle instead
 * of hitting a data API: quote, kline summary, model prediction, risk profile, universe
 * screen, backtest summary.
 */
(function (global) {
  "use strict";

  const KEY_STORE = "qa_llm_key";
  const CFG_STORE = "qa_llm_cfg";
  const DEFAULT_CFG = { baseUrl: "https://api.deepseek.com/v1", model: "deepseek-chat" };

  const cache = new Map();

  async function loadJSON(path) {
    if (cache.has(path)) return cache.get(path);
    const res = await fetch(path, { cache: "no-cache" });
    if (!res.ok) throw new Error(`${path} → HTTP ${res.status}`);
    const data = await res.json();
    cache.set(path, data);
    return data;
  }

  const cfg = () => {
    try { return { ...DEFAULT_CFG, ...JSON.parse(localStorage.getItem(CFG_STORE) || "{}") }; }
    catch { return { ...DEFAULT_CFG }; }
  };
  const saveCfg = (c) => localStorage.setItem(CFG_STORE, JSON.stringify(c));
  const getKey = () => localStorage.getItem(KEY_STORE) || "";
  const setKey = (k) => k ? localStorage.setItem(KEY_STORE, k) : localStorage.removeItem(KEY_STORE);
  const clearKey = () => localStorage.removeItem(KEY_STORE);

  /* ------------------------------------------------------------------ data */
  let ctx = { predictions: null, klines: new Map(), backtest: null, meta: null };

  async function prime() {
    if (!ctx.predictions) ctx.predictions = await loadJSON("data/predictions.json");
    if (!ctx.meta) ctx.meta = await loadJSON("data/meta.json");
    if (!ctx.backtest) ctx.backtest = await loadJSON("data/backtest.json");
    return ctx;
  }

  async function kline(code) {
    if (ctx.klines.has(code)) return ctx.klines.get(code);
    const data = await loadJSON(`data/kline/${code}.json`);
    ctx.klines.set(code, data);
    return data;
  }

  function findPrediction(code) {
    return (ctx.predictions?.items || []).find(x => x.code === code);
  }

  function returnsOf(bars, days) {
    if (bars.length <= days) return null;
    const a = bars[bars.length - 1 - days].close, b = bars[bars.length - 1].close;
    return ((b / a - 1) * 100);
  }

  function rsiOf(bars, period = 14) {
    let gain = 0, loss = 0;
    for (let i = bars.length - period; i < bars.length; i++) {
      const d = bars[i].close - bars[i - 1].close;
      if (d >= 0) gain += d; else loss -= d;
    }
    if (loss === 0) return 100;
    const rs = (gain / period) / (loss / period);
    return 100 - 100 / (1 + rs);
  }

  /* ----------------------------------------------------------------- tools */
  const TOOLS = [
    {
      type: "function",
      function: {
        name: "get_quote",
        description: "查询单只股票最新收盘价与近 5/20/60 日涨跌幅。数据来自静态快照。",
        parameters: { type: "object", properties: { code: { type: "string", description: "6 位代码" } }, required: ["code"] },
      },
    },
    {
      type: "function",
      function: {
        name: "predict_return",
        description: "读取本地 CNN-LSTM-Attention 模型对该股未来 7 个交易日的预测收益（样本外 IC≈0，仅作参考）。",
        parameters: { type: "object", properties: { code: { type: "string" } }, required: ["code"] },
      },
    },
    {
      type: "function",
      function: {
        name: "risk_profile",
        description: "统计该股近 120 个交易日的年化波动、最大回撤、RSI 与区间位置。",
        parameters: { type: "object", properties: { code: { type: "string" } }, required: ["code"] },
      },
    },
    {
      type: "function",
      function: {
        name: "screen_universe",
        description: "按模型预测收益排序，返回最看好的前 K 只（快照内共若干只）。",
        parameters: { type: "object", properties: { top_k: { type: "integer", default: 10 } } },
      },
    },
    {
      type: "function",
      function: {
        name: "backtest_summary",
        description: "返回动量策略回测的绩效摘要，含策略、沪深300、等权持有三条基准。",
        parameters: { type: "object", properties: {} },
      },
    },
  ];

  async function callTool(name, args) {
    await prime();
    const code = args && args.code ? String(args.code).padStart(6, "0") : "";
    try {
      if (name === "get_quote") {
        const k = await kline(code);
        const bars = k.bars;
        const last = bars[bars.length - 1], prev = bars[bars.length - 2];
        return {
          ok: true, code, name: k.name, date: last.date, close: last.close,
          pct_chg: +(((last.close / prev.close) - 1) * 100).toFixed(2),
          ret_5d_pct: returnsOf(bars, 5), ret_20d_pct: returnsOf(bars, 20), ret_60d_pct: returnsOf(bars, 60),
        };
      }
      if (name === "predict_return") {
        const p = findPrediction(code);
        if (!p) return { ok: false, error: `快照里没有 ${code}，这只股票未被打包进静态数据` };
        const k = await kline(code).catch(() => null);
        return {
          ok: true, code, name: p.name, as_of: ctx.predictions.as_of,
          horizon_days: ctx.predictions.horizon_days, predicted_return: p.pred,
          last_close: p.close, close_check: k ? k.bars[k.bars.length - 1].close : null,
          note: "该模型样本外方向准确率 49.74%、日频 IC -0.016，接近无信号",
        };
      }
      if (name === "risk_profile") {
        const k = await kline(code);
        const bars = k.bars;
        const rets = [];
        for (let i = 1; i < bars.length; i++) rets.push(bars[i].close / bars[i - 1].close - 1);
        const mean = rets.reduce((a, b) => a + b, 0) / rets.length;
        const sd = Math.sqrt(rets.reduce((a, b) => a + (b - mean) ** 2, 0) / rets.length);
        let peak = -Infinity, mdd = 0;
        bars.forEach(b => { peak = Math.max(peak, b.close); mdd = Math.min(mdd, b.close / peak - 1); });
        const hi = Math.max(...bars.map(b => b.high)), lo = Math.min(...bars.map(b => b.low));
        const last = bars[bars.length - 1].close;
        return {
          ok: true, code, name: k.name, bars: bars.length,
          annual_vol_pct: +(sd * Math.sqrt(252) * 100).toFixed(2),
          max_drawdown_pct: +(mdd * 100).toFixed(2),
          rsi_14: +rsiOf(bars, 14).toFixed(1),
          range_position: +((last - lo) / (hi - lo || 1)).toFixed(2),
        };
      }
      if (name === "screen_universe") {
        const k = Math.max(1, parseInt(args && args.top_k, 10) || 10);
        const items = ctx.predictions.items.slice(0, k)
          .map(x => ({ code: x.code, name: x.name, predicted_return_pct: +(x.pred * 100).toFixed(2) }));
        return { ok: true, as_of: ctx.predictions.as_of, scored: ctx.predictions.items.length, items };
      }
      if (name === "backtest_summary") {
        return { ok: true, stats: ctx.backtest.stats, params: ctx.backtest.params };
      }
      return { ok: false, error: `未知工具 ${name}` };
    } catch (err) {
      return { ok: false, error: String(err.message || err) };
    }
  }

  /* ------------------------------------------------------------------- LLM */
  const SYSTEM = [
    "你是 QuantAgent，一个 A 股量化研究助理。",
    "规则：所有数字必须来自工具返回，禁止凭记忆编造行情或指标；",
    "模型样本外几乎没有预测能力（方向准确率 49.74%，IC≈0），不要把它说成可靠信号；",
    "回测同时有沪深300 与「等权持有」两个基准，只报指数会高估策略；",
    "不给出「买入」「卖出」这类指令，用描述性语言；中文回答，200 字以内。",
  ].join("\n");

  async function chat(messages, tools, signal) {
    const c = cfg();
    const key = getKey();
    if (!key) throw new Error("NO_KEY");
    const res = await fetch(c.baseUrl.replace(/\/$/, "") + "/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${key}` },
      body: JSON.stringify({ model: c.model, messages, tools, tool_choice: "auto", temperature: 0.2, max_tokens: 700 }),
      signal,
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`HTTP ${res.status}: ${text.slice(0, 200)}`);
    }
    const body = await res.json();
    return { message: body.choices[0].message, usage: body.usage || {}, model: body.model };
  }

  async function ask(question, onEvent) {
    await prime();
    const messages = [{ role: "system", content: SYSTEM }, { role: "user", content: question }];
    const trace = [];
    const emit = (e) => onEvent && onEvent(e);

    for (let step = 0; step < 4; step++) {
      emit({ type: "thinking", step: step + 1 });
      let reply;
      try {
        reply = await chat(messages, TOOLS);
      } catch (err) {
        const msg = String(err.message || err);
        if (msg === "NO_KEY") return { ok: false, error: "未填写 API key" };
        if (msg.includes("Failed to fetch")) {
          return { ok: false, error: "网络请求失败：检查 base_url 是否正确，或该服务是否允许浏览器跨域调用" };
        }
        return { ok: false, error: msg };
      }

      const calls = reply.message.tool_calls || [];
      if (!calls.length) {
        return { ok: true, answer: (reply.message.content || "").trim(), trace, usage: reply.usage, model: reply.model };
      }

      messages.push({ role: "assistant", content: reply.message.content || "", tool_calls: calls });
      for (const call of calls) {
        const name = call.function.name;
        let args = {};
        try { args = JSON.parse(call.function.arguments || "{}"); } catch { args = {}; }
        emit({ type: "tool_call", name, arguments: args });
        const result = await callTool(name, args);
        trace.push({ name, args, result });
        emit({ type: "tool_result", name, result });
        messages.push({ role: "tool", tool_call_id: call.id || name, content: JSON.stringify(result).slice(0, 4000) });
      }
    }
    return { ok: false, error: "工具调用超过 4 轮，已停止" };
  }

  global.QA = {
    prime, ask, kline, loadJSON, getKey, setKey, clearKey, cfg, saveCfg, DEFAULT_CFG,
    findPrediction, returnsOf, rsiOf,
    asOf: () => (ctx.predictions ? ctx.predictions.as_of : "—"),
    predictions: () => ctx.predictions,
  };
})(window);
