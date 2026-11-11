"""Daily paper-trading pipeline.

This module:
- Loads configuration
- Loads cached daily prices
- Selects eligible pairs
- Calculates current signals
- Reads paper positions and open orders
- Prevents duplicate orders
- Builds and submits required orders
- Saves logs and persistent state
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.state import StateStore
from src.execution_paper import PaperExecutionEngine


logger = logging.getLogger(__name__)

RESULTS_DIR = Path("data/results")
STATE_DIR = Path("data/state")
STATE_FILE = STATE_DIR / "paper_state.json"


def load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid configuration file: {config_path}")

    return config


def load_prices(config: dict[str, Any]) -> pd.DataFrame:
    """Load and normalize the cached price dataset."""

    data_file = config["project"]["data_file"]
    data_path = Path(data_file)

    if not data_path.is_absolute():
        project_root = Path(__file__).resolve().parents[1]
        data_path = project_root / data_path

    if not data_path.exists():
        raise FileNotFoundError(
            f"Price data file not found: {data_path.resolve()}"
        )

    prices = pd.read_parquet(data_path)

    logger.info(
        "Loaded parquet with columns: %s",
        list(prices.columns),
    )

    # Case 1: already long format with date/ticker/close columns
    date_column = None

    for candidate in ["date", "Date", "DATE", "datetime", "timestamp"]:
        if candidate in prices.columns:
            date_column = candidate
            break

    if date_column is not None:
        prices = prices.rename(
            columns={
                date_column: "date",
                "ticker": "ticker",
                "Ticker": "ticker",
                "symbol": "ticker",
                "Symbol": "ticker",
                "close": "close",
                "Close": "close",
                "adj_close": "close",
                "Adj Close": "close",
            }
        )

        required = {"date", "ticker", "close"}

        if not required.issubset(prices.columns):
            raise ValueError(
                f"Parquet has date column but is missing one of {required}. "
                f"Actual columns: {list(prices.columns)}"
            )

        prices["date"] = pd.to_datetime(prices["date"])
        prices["close"] = pd.to_numeric(prices["close"], errors="coerce")

        return prices.dropna(subset=["date", "ticker", "close"])

    # Case 2: wide format, with dates as the index and tickers as columns
    if isinstance(prices.index, pd.DatetimeIndex):
        prices = prices.reset_index()

    else:
        index_name = prices.index.name

        if index_name is None:
            prices = prices.reset_index()
            first_column = prices.columns[0]
        else:
            prices = prices.reset_index()
            first_column = index_name

        prices = prices.rename(columns={first_column: "date"})

    prices["date"] = pd.to_datetime(prices["date"])

    # Melt ticker columns into long format
    value_columns = [
        column
        for column in prices.columns
        if column != "date"
    ]

    if not value_columns:
        raise ValueError(
            "No ticker price columns found in parquet file. "
            f"Actual columns: {list(prices.columns)}"
        )

    prices = prices.melt(
        id_vars="date",
        value_vars=value_columns,
        var_name="ticker",
        value_name="close",
    )

    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")

    return prices.dropna(subset=["date", "ticker", "close"])


def select_pairs(
    prices: pd.DataFrame,
    config: dict[str, Any],
) -> list[tuple[str, str]]:
    pair_cfg = config["pair_selection"]
    economic_pairs = pair_cfg.get("economic_pairs", [])

    available = set(prices["ticker"].unique())
    selected = []

    for asset_a, asset_b in economic_pairs:
        if asset_a in available and asset_b in available:
            selected.append((asset_a, asset_b))

    return selected


def calculate_pair_signal(
    prices: pd.DataFrame,
    asset_a: str,
    asset_b: str,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    strategy_cfg = config["strategy"]

    hedge_window = int(strategy_cfg["hedge_ratio_window"])
    zscore_window = int(strategy_cfg["zscore_window"])

    pair_prices = prices[
        prices["ticker"].isin([asset_a, asset_b])
    ].pivot_table(
        index="date",
        columns="ticker",
        values="close",
    ).dropna()

    required_columns = {asset_a, asset_b}

    if not required_columns.issubset(pair_prices.columns):
        return None

    if len(pair_prices) < max(hedge_window, zscore_window):
        return None

    pair_prices = pair_prices.tail(hedge_window)

    log_a = np.log(pair_prices[asset_a])
    log_b = np.log(pair_prices[asset_b])

    hedge_ratio = calculate_hedge_ratio(log_a, log_b)
    spread = log_a - hedge_ratio * log_b

    mean = spread.rolling(zscore_window).mean()
    std = spread.rolling(zscore_window).std()

    zscore = (spread - mean) / std
    latest_zscore = float(zscore.iloc[-1])

    if not pd.notna(latest_zscore):
        return None

    entry = float(strategy_cfg["entry_zscore"])
    exit_threshold = float(strategy_cfg["exit_zscore"])
    stop = float(strategy_cfg["stop_zscore"])

    if latest_zscore <= -entry:
        target_signal = 1
    elif latest_zscore >= entry:
        target_signal = -1
    else:
        target_signal = 0

    market_ok = check_market_filter(prices, config)

    return {
        "asset_a": asset_a,
        "asset_b": asset_b,
        "signal_date": pair_prices.index[-1].date().isoformat(),
        "latest_spread": float(spread.iloc[-1]),
        "latest_zscore": latest_zscore,
        "hedge_ratio": hedge_ratio,
        "target_signal": target_signal,
        "market_ok": market_ok,
        "entry_threshold": entry,
        "exit_threshold": exit_threshold,
        "stop_threshold": stop,
    }


def calculate_hedge_ratio(
    log_a: pd.Series,
    log_b: pd.Series,
) -> float:
    x = log_b.to_numpy()
    y = log_a.to_numpy()

    x_mean = x.mean()
    y_mean = y.mean()

    denominator = ((x - x_mean) ** 2).sum()

    if denominator == 0:
        return 1.0

    beta = ((x - x_mean) * (y - y_mean)).sum() / denominator

    return float(beta)


def check_market_filter(
    prices: pd.DataFrame,
    config: dict[str, Any],
) -> bool:
    strategy_cfg = config["strategy"]

    ticker = strategy_cfg.get("market_filter_ticker", "SPY")
    lookback = int(strategy_cfg.get("market_filter_lookback_days", 20))
    max_drawdown = float(
        strategy_cfg.get("market_filter_max_drawdown", -0.10)
    )

    ticker_prices = prices[prices["ticker"] == ticker].sort_values("date")

    if len(ticker_prices) < lookback:
        return True

    recent = ticker_prices.tail(lookback)
    start_price = float(recent["close"].iloc[0])
    lowest_price = float(recent["close"].min())

    if start_price <= 0:
        return True

    drawdown = lowest_price / start_price - 1

    return drawdown >= max_drawdown

def reconcile_pending_pair(
    pair_key: str,
    pair_state: dict[str, Any],
    positions: pd.DataFrame,
    execution_engine: PaperExecutionEngine,
) -> bool:
    """Promote a pending pair to active once both legs have filled."""

    asset_a, asset_b = pair_key.split("_")

    open_orders = execution_engine.get_open_orders()

    remaining_orders = [
        order
        for order in open_orders
        if order["ticker"] in {asset_a, asset_b}
        and int(order["order_id"]) in pair_state.get("open_order_ids", [])
    ]

    if remaining_orders:
        return False

    position_map = {}

    if not positions.empty:
        position_map = {
            str(row["ticker"]): row
            for _, row in positions.iterrows()
        }

    asset_a_position = position_map.get(asset_a)
    asset_b_position = position_map.get(asset_b)

    if asset_a_position is None or asset_b_position is None:
        return False

    asset_a_quantity = float(asset_a_position["signed_quantity"])
    asset_b_quantity = float(asset_b_position["signed_quantity"])

    target_signal = int(pair_state.get("target_signal", 0))

    if target_signal == 0:
        return False

    expected_asset_a_direction = np.sign(target_signal)
    expected_asset_b_direction = np.sign(-target_signal)

    asset_a_direction = np.sign(asset_a_quantity)
    asset_b_direction = np.sign(asset_b_quantity)

    if asset_a_direction != expected_asset_a_direction:
        return False

    if asset_b_direction != expected_asset_b_direction:
        return False


    return True

def build_reconciled_order_plan(
    signals: pd.DataFrame,
    positions: pd.DataFrame,
    execution_engine: PaperExecutionEngine,
    state_store: StateStore,
    config: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    strategy_cfg = config["strategy"]
    execution_cfg = config["execution"]

    cooldown_days = int(strategy_cfg.get("cooldown_days", 10))
    max_orders = int(execution_cfg.get("max_orders_per_day", 10))
    max_notional = float(execution_cfg.get("max_order_notional", 500))

    state_store.decrement_cooldowns()
    active_pairs = [
        pair_key
        for pair_key, pair_state in state_store.state["pairs"].items()
        if pair_state.get("status") in {"pending_entry", "active"}
    ]

    max_active_pairs = int(
        execution_cfg.get("max_active_pairs", 3)
    )

    if len(active_pairs) >= max_active_pairs:
        logger.warning(
            "Maximum active pairs reached: %s",
            max_active_pairs,
        )
        return pd.DataFrame(
            columns=[
                "ticker",
                "side",
                "quantity",
                "limit_price",
                "pair_key",
                "target_signal",
                "entry_zscore",
                "signal_date",
            ]
        )

    position_map: dict[str, pd.Series] = {}

    if not positions.empty:
        position_map = {
            str(row["ticker"]): row
            for _, row in positions.iterrows()
        }

    for _, signal in signals.iterrows():
        asset_a = str(signal["asset_a"])
        asset_b = str(signal["asset_b"])
        pair_key = f"{asset_a}_{asset_b}"

        target_signal = int(signal["target_signal"])
        latest_zscore = float(signal["latest_zscore"])
        signal_date = str(signal["signal_date"])

        pair_state = state_store.get_pair(pair_key)
        if pair_state.get("status") == "pending_entry":
            is_fully_filled = reconcile_pending_pair(
                pair_key=pair_key,
                pair_state=pair_state,
                positions=positions,
                execution_engine=execution_engine,
            )

            if is_fully_filled:
                state_store.mark_active(
                    pair_key=pair_key,
                    target_signal=int(pair_state["target_signal"]),
                    entry_date=str(pair_state["entry_date"]),
                    entry_zscore=float(pair_state["entry_zscore"]),
                )

                pair_state = state_store.get_pair(pair_key)
        current_status = str(pair_state.get("status", "flat"))
        current_target = int(pair_state.get("target_signal", 0))

        leg_a_orders = execution_engine.get_orders_for_ticker(asset_a)
        leg_b_orders = execution_engine.get_orders_for_ticker(asset_b)

        saved_order_ids = pair_state.get("open_order_ids", [])

        saved_orders = [
            order
            for order in leg_a_orders + leg_b_orders
            if int(order["order_id"]) in saved_order_ids
        ]

        both_legs_pending = bool(leg_a_orders) and bool(leg_b_orders)

        if saved_orders or both_legs_pending:
            active_order_ids = [
                int(order["order_id"])
                for order in leg_a_orders + leg_b_orders
                if str(order["order_id"]).isdigit()
            ]

            pair_state["status"] = "pending_entry"
            pair_state["open_order_ids"] = active_order_ids
            continue

        if current_status == "pending_entry":
            state_store.mark_active(
                pair_key=pair_key,
                target_signal=current_target,
                entry_date=str(
                    pair_state.get("entry_date") or signal_date
                ),
                entry_zscore=float(
                    pair_state.get("entry_zscore") or latest_zscore
                ),
            )
            current_status = "active"

        if target_signal == current_target:
            continue

        if target_signal == 0:
            continue

        if not bool(signal.get("market_ok", True)):
            continue

        if pair_state.get("cooldown_remaining", 0) > 0:
            continue

        if len(rows) >= max_orders:
            break

        leg_a = execution_engine.build_order(
            ticker=asset_a,
            target_signal=target_signal,
            capital=max_notional,
            pair_key=pair_key,
            entry_zscore=latest_zscore,
            signal_date=signal_date,
        )

        leg_b = execution_engine.build_order(
            ticker=asset_b,
            target_signal=-target_signal,
            capital=max_notional,
            pair_key=pair_key,
            entry_zscore=latest_zscore,
            signal_date=signal_date,
        )

        if leg_a is None or leg_b is None:
            continue

        rows.append(
            {
                "ticker": leg_a.ticker,
                "side": leg_a.side,
                "quantity": leg_a.quantity,
                "limit_price": leg_a.limit_price,
                "pair_key": leg_a.pair_key,
                "target_signal": leg_a.target_signal,
                "entry_zscore": leg_a.entry_zscore,
                "signal_date": leg_a.signal_date,
            }
        )

        rows.append(
            {
                "ticker": leg_b.ticker,
                "side": leg_b.side,
                "quantity": leg_b.quantity,
                "limit_price": leg_b.limit_price,
                "pair_key": leg_b.pair_key,
                "target_signal": leg_b.target_signal,
                "entry_zscore": leg_b.entry_zscore,
                "signal_date": leg_b.signal_date,
            }
        )

        state_store.mark_pending_entry(
            pair_key=pair_key,
            target_signal=target_signal,
            entry_zscore=latest_zscore,
            order_ids=[],
        )

    return pd.DataFrame(
        rows,
        columns=[
            "ticker",
            "side",
            "quantity",
            "limit_price",
            "pair_key",
            "target_signal",
            "entry_zscore",
            "signal_date",
        ],
    )


def save_csv(
    dataframe: pd.DataFrame,
    filename: str,
) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"{filename}_{timestamp}.csv"

    dataframe.to_csv(path, index=False)

    return path


def run_daily_from_config(config: dict[str, Any]) -> int:
    """Run the daily pipeline using an already-loaded configuration."""

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )

    logger.info("Starting daily paper-trading pipeline")

    prices = load_prices(config)
    logger.info(
        "Loaded %s tickers and %s trading days",
        prices["ticker"].nunique(),
        prices["date"].nunique(),
    )

    pairs = select_pairs(prices, config)
    logger.info("Selected %s eligible pairs", len(pairs))

    signals: list[dict[str, Any]] = []

    for asset_a, asset_b in pairs:
        signal = calculate_pair_signal(
            prices=prices,
            asset_a=asset_a,
            asset_b=asset_b,
            config=config,
        )

        if signal is not None:
            signals.append(signal)

    signals_df = pd.DataFrame(signals)

    if signals_df.empty:
        logger.warning("No valid signals calculated")
        return 0

    state_store = StateStore(STATE_FILE)

    execution_engine = PaperExecutionEngine(
        config=config,
        state_store=state_store,
        prices=prices,
    )

    execution_engine.connect()

    try:
        positions = execution_engine.get_positions()
        open_orders = execution_engine.get_open_orders()

        logger.info("Current paper positions:")
        logger.info(
            positions.to_string(index=False)
            if not positions.empty
            else "Empty"
        )

        logger.info("Open orders:")
        logger.info(
            pd.DataFrame(open_orders).to_string(index=False)
            if open_orders
            else "None"
        )

        stale_cancelled = execution_engine.cancel_stale_orders(
            max_age_days=3
        )

        if stale_cancelled:
            logger.info(
                "Cancelled stale orders: %s",
                stale_cancelled,
            )

        order_plan = build_reconciled_order_plan(
            signals=signals_df,
            positions=positions,
            execution_engine=execution_engine,
            state_store=state_store,
            config=config,
        )

        logger.info("Order plan:")
        logger.info(
            order_plan.to_string(index=False)
            if not order_plan.empty
            else "Empty"
        )

        submit_orders = bool(
            config["execution"].get("submit_orders", False)
        )

        if submit_orders and not order_plan.empty:
            submitted = execution_engine.submit_orders(order_plan)

            if submitted:
                state_store.save()

                submitted_df = pd.DataFrame(submitted)
                save_csv(
                    submitted_df,
                    "submitted_orders",
                )

        state_store.save()

        signals_path = save_csv(
            signals_df,
            "daily_signals",
        )

        order_plan_path = save_csv(
            order_plan,
            "order_plan",
        )

        logger.info("Saved daily signals: %s", signals_path)
        logger.info("Saved order plan: %s", order_plan_path)

    finally:
        execution_engine.disconnect()

    logger.info("Daily paper-trading pipeline finished")
    return 0


def run_daily(config_path: str | Path) -> int:
    """Run the daily pipeline from a YAML config path."""

    config = load_config(config_path)
    return run_daily_from_config(config)


def run_daily_pipeline(config: str | dict[str, Any]) -> int:
    """Compatibility entry point used by pipeline.py.

    Accepts either:
    - a YAML config file path, or
    - an already-loaded configuration dictionary.
    """

    if isinstance(config, dict):
        return run_daily_from_config(config)

    return run_daily(config)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the daily paper-trading pipeline"
    )

    parser.add_argument(
        "--config",
        default="config/strategy.yaml",
        help="Path to YAML configuration file",
    )

    args = parser.parse_args()

    try:
        return run_daily(args.config)
    except Exception as exc:
        logger.exception("Daily pipeline failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())