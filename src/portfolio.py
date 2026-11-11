import numpy as np
import pandas as pd

from src.backtest import calculate_performance_metrics


def combine_pair_backtests(
    backtests: list[pd.DataFrame],
    portfolio_config: dict,
    backtest_config: dict,
) -> pd.DataFrame:
    """Combine eligible pair backtests into an equal-weight portfolio."""

    if not backtests:
        raise ValueError("No eligible pairs to combine.")

    selected = []

    for backtest in backtests:
        metrics = calculate_performance_metrics(backtest)

        min_return = portfolio_config["min_pair_return"]

        if (
            metrics["total_return"] >= min_return
            and not np.isnan(metrics["total_return"])
        ):
            selected.append(backtest)

    if not selected:
        raise ValueError("No pair met the minimum return requirement.")

    max_pairs = backtest_config["max_active_pairs"]
    selected = selected[:max_pairs]

    aligned_returns = {}
    if not selected:
        failed_pairs = []

        for backtest in backtests:
            asset_a = backtest["asset_a"].iloc[0]
            asset_b = backtest["asset_b"].iloc[0]

            metrics = calculate_performance_metrics(backtest)

            failed_pairs.append(
                f"{asset_a}/{asset_b}: "
                f"return={metrics['total_return']:.4%}"
            )

        raise ValueError(
            "No pair met the minimum return requirement. "
            f"Minimum required: {portfolio_config['min_pair_return']}. "
            f"Candidates: {', '.join(failed_pairs)}"
        )
    for backtest in selected:
        asset_a = backtest["asset_a"].iloc[0]
        asset_b = backtest["asset_b"].iloc[0]

        pair_name = f"{asset_a}_{asset_b}"

        aligned_returns[pair_name] = backtest["net_return"]

    aligned_returns = pd.DataFrame(aligned_returns)

    aligned_returns = aligned_returns.fillna(0)

    weights = pd.Series(
        1 / len(selected),
        index=aligned_returns.columns,
    )

    portfolio_returns = aligned_returns.mul(weights).sum(axis=1)

    capital_per_pair = backtest_config["capital_per_pair"]
    total_capital = capital_per_pair * len(selected)

    portfolio_equity = total_capital * (
        1 + portfolio_returns
    ).cumprod()

    return pd.DataFrame(
        {
            "portfolio_return": portfolio_returns,
            "portfolio_equity": portfolio_equity,
        },
        index=aligned_returns.index,
    )