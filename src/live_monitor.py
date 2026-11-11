"""Continuous intraday monitor for the paper-trading pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from logging.handlers import RotatingFileHandler

from src.reconciliation import (
    reconcile_pending_entries,
    reconcile_exits,
    build_exit_orders,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.state import StateStore
from src.live_quotes import LiveQuoteUpdater
from src.daily import (
    load_prices,
    select_pairs,
    calculate_pair_signal,
    build_reconciled_order_plan,
)
from src.execution_paper import PaperExecutionEngine
from src.reconciliation import (
    reconcile_pending_entries,
    reconcile_exits,
)

logger = logging.getLogger(__name__)

STATE_FILE = Path("data/state/paper_state.json")


def is_us_market_open() -> bool:
    """Approximate US regular-hours check in Hong Kong time."""

    now = datetime.now()
    weekday = now.weekday()

    if weekday >= 5:
        return False

    hour = now.hour
    minute = now.minute

    if 22 <= hour < 24:
        return True

    if 0 <= hour < 5:
        return True

    if hour == 21 and minute >= 30:
        return True

    if hour == 5 and minute == 0:
        return True

    return False


def run_monitor_cycle(
    config: dict[str, Any],
    execution_engine: PaperExecutionEngine,
    quote_updater: LiveQuoteUpdater,
    state_store: StateStore,
) -> None:
    """Run one live monitoring cycle."""

    prices = load_prices(config)

    pairs = select_pairs(prices, config)

    tickers = sorted(
        {
            ticker
            for pair in pairs
            for ticker in pair
        }
    )

    latest_prices = quote_updater.get_latest_prices(tickers)

    prices = quote_updater.overlay_latest_prices(
        prices=prices,
        latest_prices=latest_prices,
    )

    signals = []

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
        return

    positions = execution_engine.get_positions()
    open_orders = execution_engine.get_open_orders()

    reconcile_pending_entries(
        state_store=state_store,
        positions=positions,
        execution_engine=execution_engine,
    )

    reconcile_exits(
        state_store=state_store,
        positions=positions,
        cooldown_days=int(
            config["strategy"].get("cooldown_days", 10)
        ),
    )


    reconcile_pending_entries(
        state_store=state_store,
        positions=positions,
        execution_engine=execution_engine,
    )

    reconcile_exits(
        state_store=state_store,
        positions=positions,
        execution_engine=execution_engine,
        cooldown_days=int(
            config["strategy"].get("cooldown_days", 10)
        ),
    )
    logger.info(
        "Monitor cycle | positions=%s | open_orders=%s",
        len(positions),
        len(open_orders),
    )

    order_plan = build_reconciled_order_plan(
        signals=signals_df,
        positions=positions,
        execution_engine=execution_engine,
        state_store=state_store,
        config=config,
    )

    exit_orders = build_exit_orders(
        state_store=state_store,
        positions=positions,
        signals=signals_df,
        config=config,
    )

    if exit_orders:
        logger.info(
            "Exit orders required:\n%s",
            pd.DataFrame(exit_orders).to_string(index=False),
        )

        submitted_exits = execution_engine.submit_exit_orders(
            exit_orders=exit_orders,
        )

        if submitted_exits:
            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            output_path = Path("data/results")
            output_path.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(submitted_exits).to_csv(
                output_path / f"submitted_exit_orders_{timestamp}.csv",
                index=False,
            )

    exit_orders = build_exit_orders(
        state_store=state_store,
        positions=positions,
        signals=signals_df,
        config=config,
    )

    if exit_orders:
        logger.info(
            "Exit orders required:\n%s",
            pd.DataFrame(exit_orders).to_string(index=False),
        )

        submitted_exits = execution_engine.submit_exit_orders(
            exit_orders=exit_orders,
        )

        if submitted_exits:
            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            output_path = Path("data/results")
            output_path.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(submitted_exits).to_csv(
                output_path / f"submitted_exit_orders_{timestamp}.csv",
                index=False,
            )

    exit_orders = build_exit_orders(
        state_store=state_store,
        positions=positions,
        signals=signals_df,
        execution_engine=execution_engine,
        config=config,
    )

    if exit_orders:
        logger.info(
            "Exit orders required:\n%s",
            pd.DataFrame(exit_orders).to_string(index=False),
        )

    if order_plan.empty:
        logger.info("No new orders required")
        state_store.save()
        return

    logger.info(
        "New order plan:\n%s",
        order_plan.to_string(index=False),
    )

    if not execution_engine.submit_orders_enabled:
        logger.info("Order submission is disabled")
        state_store.save()
        return

    submitted = execution_engine.submit_orders(order_plan)

    if submitted:
        state_store.save()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path("data/results")
        output_path.mkdir(parents=True, exist_ok=True)

        submitted_df = pd.DataFrame(submitted)
        submitted_df.to_csv(
            output_path / f"submitted_orders_{timestamp}.csv",
            index=False,
        )


def run_continuous_monitor(
    config_path: str,
    interval_seconds: int = 60,
) -> int:
    """Run the monitor continuously during US market hours."""

    with Path(config_path).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

        log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
        handlers=[
            logging.StreamHandler(),
            RotatingFileHandler(
                log_dir / "live_monitor.log",
                maxBytes=5_000_000,
                backupCount=5,
                encoding="utf-8",
            ),
        ],
    )

    state_store = StateStore(STATE_FILE)

    execution_engine = PaperExecutionEngine(
        config=config,
        state_store=state_store,
        prices=None,
    )

    quote_updater = LiveQuoteUpdater(
        host=config["execution"]["host"],
        port=config["execution"]["port"],
    )

    execution_engine.connect()
    quote_updater.connect()

    logger.info(
        "Starting continuous monitor every %s seconds",
        interval_seconds,
    )

    try:
        while True:
            try:
                if is_us_market_open():
                    run_monitor_cycle(
                        config=config,
                        execution_engine=execution_engine,
                        quote_updater=quote_updater,
                        state_store=state_store,
                    )
                else:
                    logger.info("US market is closed; waiting")

            except Exception as exc:
                logger.exception("Monitor cycle failed: %s", exc)

                try:
                    execution_engine.disconnect()
                    quote_updater.disconnect()
                except Exception:
                    logger.warning(
                        "Failed to clean up connections after error"
                    )

                time.sleep(10)

                try:
                    execution_engine.connect()
                    quote_updater.connect()
                    logger.info("Reconnected to Futu OpenD")
                except Exception as reconnect_error:
                    logger.exception(
                        "Reconnection failed: %s",
                        reconnect_error,
                    )

            time.sleep(interval_seconds)

    except KeyboardInterrupt:
        logger.info("Continuous monitor stopped by user")

    finally:
        quote_updater.disconnect()
        execution_engine.disconnect()

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the continuous paper-trading monitor"
    )

    parser.add_argument(
        "--config",
        default="config/strategy.yaml",
        help="Path to YAML configuration file",
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Seconds between monitoring cycles",
    )

    args = parser.parse_args()

    return run_continuous_monitor(
        config_path=args.config,
        interval_seconds=args.interval,
    )


if __name__ == "__main__":
    sys.exit(main())