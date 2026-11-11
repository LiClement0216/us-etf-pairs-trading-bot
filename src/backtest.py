import numpy as np
import pandas as pd


def calculate_pair_returns(
    close_prices: pd.DataFrame,
    asset_a: str,
    asset_b: str,
) -> pd.DataFrame:
    returns = close_prices[[asset_a, asset_b]].pct_change()

    return returns.rename(
        columns={
            asset_a: "return_a",
            asset_b: "return_b",
        }
    )


def backtest_pair_strategy(
    close_prices: pd.DataFrame,
    asset_a: str,
    asset_b: str,
    signals: pd.DataFrame,
    capital_per_pair: float = 1000.0,
    commission_bps: float = 5.0,
    slippage_bps: float = 5.0,
) -> pd.DataFrame:
    """Backtest a dollar-neutral pair strategy.

    Each leg receives half of the pair capital.
    Signals are executed on the next trading day.
    """

    returns = calculate_pair_returns(
        close_prices=close_prices,
        asset_a=asset_a,
        asset_b=asset_b,
    )

    positions = signals["signal"].shift(1).fillna(0)

    leg_capital = capital_per_pair / 2

    gross_return = 0.5 * positions * (
        returns["return_a"] - returns["return_b"]
    )

    position_change = positions.diff().abs().fillna(0)

    trading_cost = position_change * (
        (commission_bps + slippage_bps) / 10000
    ) * 2

    net_return = gross_return - trading_cost

    equity = capital_per_pair * (1 + net_return.fillna(0)).cumprod()

    return pd.DataFrame(
        {
            "signal": signals["signal"],
            "position": positions,
            "gross_return": gross_return,
            "trading_cost": trading_cost,
            "net_return": net_return,
            "equity": equity,
        }
    )


import numpy as np
import pandas as pd


def calculate_performance_metrics(
    backtest: pd.DataFrame,
    trading_days_per_year: int = 252,
) -> dict:
    returns = backtest["net_return"].dropna()

    if len(returns) < 2:
        return {
            "total_return": np.nan,
            "annualized_return": np.nan,
            "annualized_volatility": np.nan,
            "sharpe_ratio": np.nan,
            "max_drawdown": np.nan,
            "position_changes": 0,
            "valid_days": len(returns),
        }

    equity = backtest["equity"]

    if len(equity) < 2:
        return {
            "total_return": np.nan,
            "annualized_return": np.nan,
            "annualized_volatility": np.nan,
            "sharpe_ratio": np.nan,
            "max_drawdown": np.nan,
            "position_changes": 0,
            "valid_days": len(returns),
        }

    total_return = equity.iloc[-1] / equity.iloc[0] - 1

    annualized_return = (1 + total_return) ** (
        trading_days_per_year / len(returns)
    ) - 1

    annualized_volatility = returns.std() * np.sqrt(
        trading_days_per_year
    )

    sharpe_ratio = (
        annualized_return / annualized_volatility
        if annualized_volatility > 0
        else np.nan
    )

    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_drawdown = drawdown.min()

    position_changes = (
        backtest["position"]
        .diff()
        .abs()
        .fillna(0)
        .sum()
    )

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "position_changes": position_changes,
        "valid_days": len(returns),
    }