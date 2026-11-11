import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm


def calculate_rolling_hedge_ratio(
    close_prices: pd.DataFrame,
    asset_a: str,
    asset_b: str,
    window: int = 120,
    min_periods: int = 100,
) -> pd.Series:
    """Estimate a walk-forward hedge ratio using only past data."""

    log_prices = np.log(
        close_prices[[asset_a, asset_b]].dropna()
    )

    y = log_prices[asset_a]
    x = log_prices[asset_b]

    x_with_constant = sm.add_constant(x)

    model = RollingOLS(
        endog=y,
        exog=x_with_constant,
        window=window,
        min_nobs=min_periods,
    )

    fitted = model.fit()

    hedge_ratios = fitted.params[asset_b]

    return hedge_ratios.shift(1)

def calculate_log_prices(close_prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(close_prices)


def test_cointegration(
    close_prices: pd.DataFrame,
    asset_a: str,
    asset_b: str,
    lookback_days: int = 500,
) -> dict:
    """Run an Engle-Granger cointegration test on two price series."""

    prices = close_prices[[asset_a, asset_b]].dropna().tail(lookback_days)

    if len(prices) < lookback_days:
        raise ValueError(
            f"Not enough history for {asset_a} / {asset_b}: "
            f"{len(prices)} rows available, {lookback_days} required."
        )

    log_prices = np.log(prices)

    coint_stat, p_value, critical_values = coint(
        log_prices[asset_a],
        log_prices[asset_b],
        trend="c",
        autolag="aic",
    )

    hedge_ratio = estimate_hedge_ratio(
        log_prices[asset_a],
        log_prices[asset_b],
    )

    spread = log_prices[asset_a] - hedge_ratio * log_prices[asset_b]

    adf_result = adfuller(
        spread,
        autolag="aic",
        result_object=True,
    )

    adf_stat = adf_result.statistic
    adf_p_value = adf_result.pvalue

    return {
        "asset_a": asset_a,
        "asset_b": asset_b,
        "coint_stat": coint_stat,
        "coint_p_value": p_value,
        "coint_critical_5pct": critical_values[1],
        "hedge_ratio": hedge_ratio,
        "adf_stat": adf_stat,
        "adf_p_value": adf_p_value,
        "is_cointegrated_5pct": p_value < 0.05,
        "spread_stationary_5pct": adf_p_value < 0.05,
        "spread": spread,
    }


def estimate_hedge_ratio(
    log_price_a: pd.Series,
    log_price_b: pd.Series,
) -> float:
    """Estimate beta in: log(A) = alpha + beta * log(B)."""

    x = log_price_b.to_numpy()
    y = log_price_a.to_numpy()

    x_matrix = np.column_stack([np.ones(len(x)), x])
    coefficients = np.linalg.lstsq(x_matrix, y, rcond=None)[0]

    return float(coefficients[1])


def calculate_spread(
    close_prices: pd.DataFrame,
    asset_a: str,
    asset_b: str,
    hedge_ratio: float,
) -> pd.Series:
    log_prices = np.log(close_prices[[asset_a, asset_b]].dropna())

    return log_prices[asset_a] - hedge_ratio * log_prices[asset_b]


def calculate_spread_zscore(
    spread: pd.Series,
    lookback_days: int = 60,
) -> pd.Series:
    rolling_mean = spread.shift(1).rolling(window=lookback_days).mean()
    rolling_std = spread.shift(1).rolling(window=lookback_days).std()

    return (spread - rolling_mean) / rolling_std



def calculate_spread_quality(
    spread: pd.Series,
    lookback_days: int = 60,
) -> dict:
    rolling_std = spread.rolling(lookback_days).std()

    return {
        "latest_spread": spread.iloc[-1],
        "latest_rolling_std": rolling_std.iloc[-1],
        "spread_std_median": rolling_std.median(),
    }

def calculate_pair_score(
    correlation: float,
    coint_pass_rate: float,
    adf_p_value: float,
    hedge_ratio_cv: float,
    zscore_crossings: int,
    rolling_spread_std: float,
) -> float:
    score = 0.0

    if correlation >= 0.40:
        score += 1

    if coint_pass_rate >= 0.60:
        score += 1

    if adf_p_value < 0.10:
        score += 1

    if hedge_ratio_cv <= 0.20:
        score += 1

    if zscore_crossings >= 4:
        score += 1

    if rolling_spread_std >= 0.01:
        score += 1

    return score

def calculate_hedge_ratio_stability(
    close_prices: pd.DataFrame,
    asset_a: str,
    asset_b: str,
    window: int = 252,
    step_size: int = 63,
) -> dict:
    prices = close_prices[[asset_a, asset_b]].dropna()

    hedge_ratios = []

    for start in range(0, len(prices) - window, step_size):
        sample = prices.iloc[start:start + window]

        log_prices = np.log(sample)

        hedge_ratio = estimate_hedge_ratio(
            log_price_a=log_prices[asset_a],
            log_price_b=log_prices[asset_b],
        )

        hedge_ratios.append(hedge_ratio)

    hedge_series = pd.Series(hedge_ratios)

    return {
        "hedge_ratio_mean": hedge_series.mean(),
        "hedge_ratio_std": hedge_series.std(),
        "hedge_ratio_min": hedge_series.min(),
        "hedge_ratio_max": hedge_series.max(),
        "hedge_ratio_range": hedge_series.max() - hedge_series.min(),
        "hedge_ratio_cv": (
            hedge_series.std() / hedge_series.mean()
            if hedge_series.mean() != 0
            else np.nan
        ),
    }