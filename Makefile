PY ?= .venv/bin/python
PIP ?= uv pip
INDEX ?= https://pypi.tuna.tsinghua.edu.cn/simple

.PHONY: help install fetch train backtest tune serve test lint clean reproduce forecast ledger

help:
	@echo "make install    创建虚拟环境并安装依赖"
	@echo "make fetch      下载行情并生成特征面板（全量约 3-5 分钟）"
	@echo "make train      训练 CNN-LSTM-Attention"
	@echo "make backtest   运行动量策略回测，产出 reports/"
	@echo "make tune       Optuna 参数搜索"
	@echo "make forecast  调用大模型预测并写入台账（需 QA_LLM_API_KEY）"
	@echo "make ledger    查看台账并打分"
	@echo "make serve      启动 Web 工作台 http://127.0.0.1:8000"
	@echo "make test       运行测试"
	@echo "make reproduce  fetch + train + backtest 一条龙"

install:
	uv venv --python 3.13 .venv
	UV_DEFAULT_INDEX=$(INDEX) $(PIP) install --python $(PY) -e ".[dev]"

fetch:
	$(PY) -m quantagent.cli fetch --workers 8

train:
	$(PY) -m quantagent.cli train --max-per-symbol 200

backtest:
	$(PY) -m quantagent.cli backtest --start 2023-01-01 --end 2024-12-31

tune:
	$(PY) -m quantagent.cli tune --trials 40

forecast:
	$(PY) -m quantagent.cli forecast --codes 600519,000858,601318,600036,000001 --batch 5 --record

ledger:
	$(PY) -m quantagent.cli ledger --score

serve:
	$(PY) -m quantagent.cli serve --port 8000

test:
	$(PY) -m pytest -q --basetemp=$$(mktemp -d)

lint:
	$(PY) -m ruff check quantagent tests

reproduce:
	$(PY) -m quantagent.cli pipeline --limit 300 --epochs 60

clean:
	rm -rf data/cache reports artifacts .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
