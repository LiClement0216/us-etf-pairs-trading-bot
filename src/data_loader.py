from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf


DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def get_raw_path(ticker: str) -> Path:
    safe_ticker = ticker.replace("^", "_").replace("/", "_")
    return RAW_DIR / f"{safe_ticker}.parquet"


def download_price_history(
    tickers: Iterable[str],
    start: str,
    end: str,
    interval: str = "1d",
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Download daily OHLCV data, using local Parquet caching."""

    tickers = list(tickers)

    if not tickers:
        raise ValueError("At least one ticker is required.")

    if start >= end:
        raise ValueError("start date must be earlier than end date.")

    downloaded = {}

    for ticker in tickers:
        raw_path = get_raw_path(ticker)

        if raw_path.exists() and not refresh:
            df = pd.read_parquet(raw_path)
            downloaded[ticker] = df
            continue

        df = yf.Ticker(ticker).history(
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,
            actions=False,
        )

        if df.empty:
            print(f"WARNING: no data returned for {ticker}")
            continue

        df = clean_price_data(df, ticker)
        df.to_parquet(raw_path)
        downloaded[ticker] = df

    if not downloaded:
        raise RuntimeError("No price data was downloaded.")

    return downloaded


def clean_price_data(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Clean a single ticker's OHLCV DataFrame."""

    required_columns = {"Open", "High", "Low", "Close", "Volume"}

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"{ticker} is missing columns: {missing}")

    df = df.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"

    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    numeric_columns = ["Open", "High", "Low", "Close", "Volume"]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")

    invalid_rows = (
        df["Close"].isna()
        | (df["Close"] <= 0)
        | (df["High"] < df["Low"])
    )

    if invalid_rows.any():
        print(f"WARNING: removed {invalid_rows.sum()} invalid rows for {ticker}")
        df = df[~invalid_rows]

    return df


def save_price_data(
    price_data: dict[str, pd.DataFrame],
    filename: str,
) -> Path:
    """Save all ticker DataFrames into one Parquet file."""

    if not price_data:
        raise ValueError("price_data is empty.")

    output_path = PROCESSED_DIR / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    combined = pd.concat(
        {ticker: df["Close"] for ticker, df in price_data.items()},
        axis=1,
    )

    combined.to_parquet(output_path)
    return output_path


def load_price_data(filename: str) -> pd.DataFrame:
    """Load a saved close-price matrix."""

    path = PROCESSED_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Price file not found: {path}")

    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def align_close_prices(
    price_data: dict[str, pd.DataFrame],
    min_history_days: int = 250,
    max_ffill_days: int = 5,
) -> pd.DataFrame:
    """Create an aligned DataFrame of adjusted closing prices."""

    close_prices = pd.concat(
        {ticker: df["Close"] for ticker, df in price_data.items()},
        axis=1,
    )

    close_prices = close_prices.sort_index().ffill(limit=max_ffill_days)

    valid_columns = close_prices.columns[
        close_prices.notna().sum() >= min_history_days
    ]

    if len(valid_columns) < 2:
        raise ValueError("Need at least two tickers with enough history.")

    return close_prices[valid_columns].dropna()


def get_price_summary(close_prices: pd.DataFrame) -> pd.DataFrame:
    """Return basic data-quality statistics for each ticker."""

    return pd.DataFrame(
        {
            "first_date": close_prices.apply(lambda s: s.first_valid_index()),
            "last_date": close_prices.apply(lambda s: s.last_valid_index()),
            "rows": close_prices.notna().sum(),
            "missing_rows": close_prices.isna().sum(),
            "start_price": close_prices.apply(lambda s: s.dropna().iloc[0]),
            "last_price": close_prices.apply(lambda s: s.dropna().iloc[-1]),
        }
    )