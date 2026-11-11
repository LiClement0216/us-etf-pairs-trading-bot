import numpy as np
import pandas as pd

from src.cointegration import (
    test_cointegration,
    calculate_hedge_ratio_stability,
    calculate_spread,
    calculate_spread_zscore,
)


def calculate_correlation(
    close_prices: pd.DataFrame,
    asset_a: str,
    asset_b: str,
    lookback_days: int,
) -> float:
    returns = np.log(
        close_prices[[asset_a, asset_b]]
    ).diff().dropna()

    recent_returns = returns.tail(lookback_days)

    return float(
        recent_returns[asset_a].corr(recent_returns[asset_b])
    )


def count_zscore_crossings(
    zscore: pd.Series,
    threshold: float,
) -> int:
    crossings = 0
    above = False
    below = False

    for value in zscore.dropna():
        if value >= threshold and not above:
            crossings += 1
            above = True
            below = False
        elif value <= -threshold and not below:
            crossings += 1
            above = False
            below = True

    return crossings


def build_market_filter(
    close_prices: pd.DataFrame,
    ticker: str,
    lookback_days: int,
    max_drawdown: float,
) -> pd.Series:
    return (
        close_prices[ticker]
        .pct_change(lookback_days)
        .shift(1)
        > max_drawdown
    )


def select_pairs(
    close_prices: pd.DataFrame,
    pair_config: dict,
) -> list[dict]:
    selected = []

    economic_pairs = pair_config["economic_pairs"]

    for asset_a, asset_b in economic_pairs:
        try:
            correlation = calculate_correlation(
                close_prices=close_prices,
                asset_a=asset_a,
                asset_b=asset_b,
                lookback_days=pair_config["correlation_lookback_days"],
            )

            if correlation < pair_config["min_correlation"]:
                continue

            stability = calculate_hedge_ratio_stability(
                close_prices=close_prices,
                asset_a=asset_a,
                asset_b=asset_b,
                window=pair_config["cointegration_lookback_days"],
                step_size=63,
            )

            coint_result = test_cointegration(
                close_prices=close_prices,
                asset_a=asset_a,
                asset_b=asset_b,
                lookback_days=pair_config["cointegration_lookback_days"],
            )

            spread = calculate_spread(
                close_prices=close_prices,
                asset_a=asset_a,
                asset_b=asset_b,
                hedge_ratio=coint_result["hedge_ratio"],
            )

            zscore = calculate_spread_zscore(
                spread=spread,
                lookback_days=60,
            )

            crossings = count_zscore_crossings(
                zscore=zscore,
                threshold=2.0,
            )

            is_eligible = (
                coint_result["coint_p_value"]
                < pair_config["cointegration_p_value"]
                and coint_result["adf_p_value"]
                < pair_config["adf_p_value"]
                and stability["hedge_ratio_cv"]
                <= pair_config["max_hedge_ratio_cv"]
                and stability["hedge_ratio_range"]
                <= pair_config["max_hedge_ratio_range"]
                and crossings
                >= pair_config["min_zscore_crossings"]
            )

            if not is_eligible:
                continue

            market_filter = build_market_filter(
                close_prices=close_prices,
                ticker="SPY",
                lookback_days=20,
                max_drawdown=-0.10,
            )

            selected.append(
                {
                    "asset_a": asset_a,
                    "asset_b": asset_b,
                    "correlation": correlation,
                    "coint_p_value": coint_result["coint_p_value"],
                    "adf_p_value": coint_result["adf_p_value"],
                    "hedge_ratio": coint_result["hedge_ratio"],
                    "hedge_ratio_cv": stability["hedge_ratio_cv"],
                    "zscore_crossings": crossings,
                    "spread": spread,
                    "zscore": zscore,
                    "market_filter": market_filter,
                }
            )

        except Exception as error:
            print(f"Skipped {asset_a} / {asset_b}: {error}")

    selected = sorted(
        selected,
        key=lambda pair: (
            -pair["correlation"],
            pair["coint_p_value"],
        ),
    )

    return selected[: pair_config["max_pairs"]]