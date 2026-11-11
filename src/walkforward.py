import pandas as pd
import numpy as np

from src.cointegration import calculate_rolling_hedge_ratio
from src.signals import generate_pair_signals
from src.backtest import (
    backtest_pair_strategy,
    calculate_performance_metrics,
)


def create_walk_forward_windows(
    dates: pd.DatetimeIndex,
    train_years: int = 2,
    test_months: int = 6,
    step_months: int = 6,
    min_test_days: int = 60,
):
    start_date = dates[0]
    end_date = dates[-1]

    train_end = start_date + pd.DateOffset(years=train_years)
    test_end = train_end + pd.DateOffset(months=test_months)

    windows = []

    while True:
        test_dates = dates[
            (dates > train_end)
            & (dates <= test_end)
        ]

        if len(test_dates) < min_test_days:
            break

        windows.append(
            {
                "train_start": start_date,
                "train_end": train_end,
                "test_start": train_end,
                "test_end": test_end,
                "test_days": len(test_dates),
            }
        )

        train_end += pd.DateOffset(months=step_months)
        test_end += pd.DateOffset(months=step_months)

    return windows


def run_walk_forward_backtest(
    close_prices: pd.DataFrame,
    asset_a: str,
    asset_b: str,
    hedge_window: int = 120,
    zscore_window: int = 60,
    entry_zscore: float = 2.0,
    exit_zscore: float = 0.0,
    stop_zscore: float = 3.5,
    capital_per_pair: float = 1000.0,
):
    windows = create_walk_forward_windows(
        dates=close_prices.index,
        train_years=2,
        test_months=6,
        step_months=6,
    )

    all_results = []

    for window in windows:
        train_prices = close_prices.loc[
            window["train_start"]:window["train_end"]
        ]

        test_prices = close_prices.loc[
            (close_prices.index > window["train_end"])
            & (close_prices.index <= window["test_end"])
        ]
        if len(test_prices) < 60:
            continue

        if len(train_prices) < hedge_window:
            continue

        from src.cointegration import estimate_hedge_ratio

        train_log_prices = np.log(
            train_prices[[asset_a, asset_b]].dropna()
        )

        hedge_ratio = estimate_hedge_ratio(
            log_price_a=train_log_prices[asset_a],
            log_price_b=train_log_prices[asset_b],
        )

        log_prices = np.log(
            test_prices[[asset_a, asset_b]].dropna()
        )

        spread = (
            log_prices[asset_a]
            - hedge_ratio * log_prices[asset_b]
        )

        zscore = (
            (spread - spread.rolling(zscore_window).mean())
            / spread.rolling(zscore_window).std()
        )

        signals = generate_pair_signals(
            spread=spread,
            zscore=zscore,
            entry_zscore=entry_zscore,
            exit_zscore=exit_zscore,
            stop_zscore=stop_zscore,
        )

        backtest = backtest_pair_strategy(
            close_prices=test_prices,
            asset_a=asset_a,
            asset_b=asset_b,
            signals=signals,
            capital_per_pair=capital_per_pair,
        )

        metrics = calculate_performance_metrics(backtest)
        metrics["test_start"] = window["test_start"]
        metrics["test_end"] = window["test_end"]

        all_results.append(metrics)

    return pd.DataFrame(all_results)