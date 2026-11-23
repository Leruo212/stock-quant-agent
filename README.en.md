<h1 align="center">QuantAgent</h1>

<p align="center">
  <a href="README.md"><img src="https://img.shields.io/badge/%E4%B8%AD%E6%96%87-SWITCH-4E9A06?style=flat-square" alt="Switch to Chinese"></a>
  &nbsp;
  <img src="https://img.shields.io/badge/English-CURRENT-1F4E79?style=flat-square" alt="English">
</p>

<p align="center">
  A daily-frequency quant research workbench for the CSI 300, built to be audited rather than advertised.
</p>

> A quant pipeline that reports a 49.74% directional accuracy against a 50.00% baseline is not a broken project. It is a project telling you the truth.

This repository is a full research loop over Chinese A-share index constituents: data acquisition with fallbacks, a deep baseline, a backtest against three benchmarks, an LLM forecasting agent wired to ten tools, and an append-only ledger that scores every live prediction once it matures. The baseline model is published with its negative result intact, and the strategy that replaced it is published with its backtest.

## Contents

| Module | What it does | Entry point |
|---|---|---|
| **Data** | Three-tier fallback acquisition (Tencent, Sina, Eastmoney), parquet cache, 9 features, causal date splits | `cli fetch` |
| **Model** | CNN-LSTM-Attention, 35,027 parameters, exportable attention weights | `cli train` |
| **Backtest** | Event-driven, fees and slippage modelled, three benchmark lines, Optuna search over ten parameters | `cli backtest` |
| **Evaluation** | Daily IC, decile returns, volatility buckets, Ridge baseline, degradation check | `cli evaluate` |
| **Agent** | Ten tools plus a tool-calling loop; LLM forecasts with self-consistency sampling and a reflection pass | `cli agent` |
| **Ledger** | Append-only prediction record, auto-scored on maturity | `cli ledger` |
| **Frontend** | FastAPI with hand-written SVG, no build step, no CDN; same-origin static build deployable to Pages | `cli serve` |

## Results

### Model signal, out of sample

| Metric | CNN-LSTM-Attn | Ridge baseline | Reading |
|---|---|---|---|
| Directional accuracy | 49.74% | 50.00% | Indistinguishable from always predicting up |
| Mean daily IC | -0.016 | -0.0026 | Both effectively zero |
| IC information ratio (annualised) | -0.99 | -0.19 | \|IR\| < 1, not significant |
| Prediction standard deviation | 4.2e-06 | 1.1e-02 | The convolutional net collapsed to a constant |

The last row explains the first three. A model whose output variance is six orders of magnitude below its baseline is not ranking anything; it has learned the mean. That failure is documented in the repository rather than tuned away.

### Strategy

| Period | Signal | Annualised | Sharpe | Max drawdown | CSI 300 | Equal weight |
|---|---|---|---|---|---|---|
| 2023-01 to 2024-12 | Momentum | 6.50% | 0.35 | -16.65% | 0.63% | 28.46% |
| 2023-01 to 2024-12 | Reversal | 2.72% | 0.12 | -25.24% | 0.63% | 28.46% |
| 2025-01 to 2026-10 | Momentum | 30.04% | 1.72 | -8.66% | 3.90% | 36.86% |

Momentum beats the index and loses to equal weight. Both facts are in the table.

## Quick start

```bash
git clone https://github.com/Leruo212/stock-quant-agent.git
cd stock-quant-agent
pip install -r requirements.txt

python -m quantagent.cli fetch      # acquire and cache
python -m quantagent.cli train      # train the baseline
python -m quantagent.cli backtest   # run and compare against benchmarks
python -m quantagent.cli evaluate   # IC, deciles, degradation check
python -m quantagent.cli serve      # open the local dashboard
```

The LLM agent needs an API key. Everything else runs offline once the cache is warm.

## Design notes

**Why three data sources.** Chinese market data endpoints fail without warning, and a research run that dies halfway through is worse than one that never started. The fetcher tries Tencent, then Sina, then Eastmoney, writes parquet, and records which source served which range.

**Why an append-only ledger.** Backtests are cheap to overfit. A ledger that timestamps each prediction before the outcome is known is the only way to tell whether a signal survives outside the sample that produced it.

**Why publish the collapse.** The interesting question is not whether a CNN can be made to fit index returns. It is why this one degenerated into a constant, and what that says about the feature set and the loss. Deleting the run would have hidden the answer.

## Limitations

- Single index, daily frequency, roughly four years of usable history.
- The deep baseline does not work. Treat `cli train` as a control, not a product.
- Backtests assume the modelled slippage; live execution on small caps would be worse.
- The LLM agent is a research instrument for comparison, not a trading system.

## Licence

MIT. See [LICENSE](LICENSE).
