# US ETF Pairs-Trading Bot

A Python-based statistical-arbitrage system that identifies economically related US ETF pairs, calculates spread z-scores, backtests mean-reversion strategies, and executes trades through Futu’s paper-trading API.

## Overview

The system scans a configurable universe of US ETFs, selects economically related pairs, estimates hedge ratios, calculates rolling spread z-scores, and generates entry and exit signals.

It supports:

- Daily backtesting
- Paper-trading execution through Futu OpenD
- Continuous intraday monitoring
- Live quote updates
- Persistent pair-state tracking
- Duplicate-order prevention
- Fill reconciliation
- Exit-order generation
- Trade-history logging

## Strategy

The bot uses a pairs-trading approach:

1. Select economically related ETF pairs.
2. Estimate a rolling hedge ratio.
3. Calculate the log-price spread.
4. Calculate a rolling z-score.
5. Enter when the z-score crosses an entry threshold.
6. Exit when the z-score mean-reverts, hits a stop threshold, or reaches the maximum holding period.
7. Apply a cooldown period after closing a pair.

## Example pair

The system may trade:

```text
Long IYW
Short VGT
```

when the spread z-score crosses the configured entry threshold.

## Project structure

```text
.
├── config/
│   └── strategy.yaml
├── src/
│   ├── daily.py
│   ├── execution_paper.py
│   ├── live_monitor.py
│   ├── live_quotes.py
│   ├── reconciliation.py
│   ├── state.py
│   └── trade_history.py
├── data/
│   ├── processed/
│   ├── results/
│   ├── state/
│   └── logs/
├── pipeline.py
├── requirements.txt
└── README.md
```

## Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Futu OpenD setup

1. Install and launch Futu OpenD.
2. Log in to your Futu account.
3. Ensure OpenD is listening on:

```text
127.0.0.1:11111
```

4. Open a paper-trading account in the Futu app.
5. Keep OpenD running while using the bot.

## Configuration

The main configuration file is:

```text
config/strategy.yaml
```

Key settings include:

```yaml
data:
  start_date: "2019-01-01"
  end_date: "2026-11-11"
  refresh: false

strategy:
  hedge_ratio_window: 252
  zscore_window: 60
  entry_zscore: 2.0
  exit_zscore: 0.5
  stop_zscore: 3.5
  max_holding_days: 30
  cooldown_days: 10

execution:
  mode: "paper"
  host: "127.0.0.1"
  port: 11111
  order_type: "limit"
  price_buffer_bps: 5
  max_order_notional: 500
  max_orders_per_day: 10
  submit_orders: true
```

## Run daily pipeline

```powershell
python -B pipeline.py --config config/strategy.yaml --mode daily
```

This will:

1. Load cached daily prices.
2. Select eligible pairs.
3. Calculate current signals.
4. Read paper positions and open orders.
5. Prevent duplicate orders.
6. Build and submit required paper orders.
7. Save signals, order plans, and state.

## Run continuous monitor

```powershell
python -B -m src.live_monitor --config config/strategy.yaml --interval 60
```

This will:

1. Check whether the US market is open.
2. Fetch live quotes.
3. Overlay the latest prices.
4. Recalculate signals.
5. Read paper positions and pending orders.
6. Prevent duplicate orders.
7. Submit entry or exit orders when required.
8. Save state and logs.
9. Repeat every 60 seconds.

Press `Ctrl + C` to stop the monitor.

## Paper-trading safety

This project is configured for **paper trading only**.

The execution engine uses:

```python
TrdEnv.SIMULATE
```

and requires:

```yaml
execution:
  mode: "paper"
```

Do not enable live trading until the paper system has completed multiple full entry-to-exit cycles.

## State files

The bot stores its execution state in:

```text
data/state/paper_state.json
```

Example:

```json
{
  "VGT_IYW": {
    "target_signal": -1,
    "status": "pending_entry",
    "entry_date": "2026-11-10",
    "entry_zscore": 0.401204,
    "cooldown_remaining": 0,
    "open_order_ids": [8387472][8387473]
  }
}
```

## Results

The bot saves:

```text
data/results/daily_signals_*.csv
data/results/order_plan_*.csv
data/results/submitted_orders_*.csv
data/results/submitted_exit_orders_*.csv
data/results/trade_history.jsonl
```

## Logs

Continuous-monitor logs are saved to:

```text
data/logs/live_monitor.log
```

## Current status

The system is designed for paper trading and includes:

- Daily signal generation
- Paper-order execution
- Continuous monitoring
- Live quote integration
- Duplicate-order prevention
- Persistent state
- Fill reconciliation
- Exit reconciliation
- Trade-history logging

## Disclaimer

This software is for educational and research purposes only. It is not financial advice. Trading involves risk, and past performance does not guarantee future results.