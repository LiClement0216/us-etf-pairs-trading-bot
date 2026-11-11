import argparse

import pandas as pd

from src.config import load_config
from src.data_loader import (
    download_price_history,
    align_close_prices,
)
from src.pair_selector import select_pairs
from src.signals import generate_pair_signals
from src.backtest import (
    backtest_pair_strategy,
    calculate_performance_metrics,
)
from src.portfolio import combine_pair_backtests
from src.report import save_results
from src.daily import run_daily_pipeline


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run the pairs-trading research pipeline."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/strategy.yaml",
        help="Path to the YAML configuration file.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="backtest",
        choices=["backtest", "daily", "check"],
        help="Pipeline execution mode.",
    )

    return parser.parse_args()


def run_backtest_pipeline(config: dict) -> None:
    tickers = config["universe"]["tickers"]

    print("Loading price data...")

    price_data = download_price_history(
        tickers=tickers,
        start=config["data"]["start_date"],
        end=config["data"]["end_date"],
        refresh=config["data"]["refresh"],
    )

    close_prices = align_close_prices(price_data)

    print(f"Loaded {close_prices.shape[1]} tickers "
          f"and {close_prices.shape[0]} trading days.")

    print("Selecting eligible pairs...")

    selected_pairs = select_pairs(
        close_prices=close_prices,
        pair_config=config["pair_selection"],
    )

    if not selected_pairs:
        raise ValueError("No eligible pairs were selected.")

    print(f"Selected {len(selected_pairs)} eligible pairs.")

    pair_backtests = []
    pair_metrics = []

    for pair in selected_pairs:
        asset_a = pair["asset_a"]
        asset_b = pair["asset_b"]

        print(f"Backtesting {asset_a} / {asset_b}...")

        signals = generate_pair_signals(
            spread=pair["spread"],
            zscore=pair["zscore"],
            entry_zscore=config["strategy"]["entry_zscore"],
            exit_zscore=config["strategy"]["exit_zscore"],
            stop_zscore=config["strategy"]["stop_zscore"],
            max_holding_days=config["strategy"]["max_holding_days"],
            cooldown_days=config["strategy"]["cooldown_days"],
            market_filter=pair["market_filter"],
        )

        backtest = backtest_pair_strategy(
            close_prices=close_prices,
            asset_a=asset_a,
            asset_b=asset_b,
            signals=signals,
            capital_per_pair=config["backtest"]["capital_per_pair"],
            commission_bps=config["backtest"]["commission_bps"],
            slippage_bps=config["backtest"]["slippage_bps"],
        )

        backtest["asset_a"] = asset_a
        backtest["asset_b"] = asset_b

        metrics = calculate_performance_metrics(backtest)
        metrics["asset_a"] = asset_a
        metrics["asset_b"] = asset_b

        pair_backtests.append(backtest)
        pair_metrics.append(metrics)

    metrics_df = pd.DataFrame(pair_metrics)

    print("\nPair backtest results:")
    print(
        metrics_df[
            [
                "asset_a",
                "asset_b",
                "total_return",
                "annualized_return",
                "sharpe_ratio",
                "max_drawdown",
                "position_changes",
            ]
        ].to_string(index=False)
    )

    portfolio = None

    try:
        portfolio = combine_pair_backtests(
            backtests=pair_backtests,
            portfolio_config=config["portfolio"],
            backtest_config=config["backtest"],
        )

        print("\nPortfolio backtest created.")

    except ValueError as error:
        print(f"\nPortfolio not created: {error}")

    output_paths = save_results(
        config=config,
        selected_pairs=selected_pairs,
        metrics=metrics_df,
        portfolio=portfolio,
    )

    print("\nPipeline completed.")
    print(f"Selected pairs: {len(selected_pairs)}")

    for name, path in output_paths.items():
        if path is not None:
            print(f"{name}: {path}")

def run_connection_check(config: dict) -> None:
    from src.execution_paper import FutuPaperExecutor

    executor = FutuPaperExecutor(
        host=config["execution"]["host"],
        port=config["execution"]["port"],
    )

    try:
        account_id = executor.get_paper_account()
        positions = executor.get_positions()
        orders = executor.get_open_orders()

        print("Paper account ID:", account_id)
        print("\nPositions:")
        print(positions.to_string(index=False))
        print("\nOpen orders:")
        print(orders.to_string(index=False))
    finally:
        executor.close()

def main():
    args = parse_arguments()
    config = load_config(args.config)
    print("Loaded entry_zscore:", config["strategy"]["entry_zscore"])
    print("Loaded submit_orders:", config["execution"]["submit_orders"])

    if args.mode == "backtest":
        run_backtest_pipeline(config)

    elif args.mode == "daily":
        run_daily_pipeline(config)

    elif args.mode == "check":
        run_connection_check(config)

if __name__ == "__main__":
    main()