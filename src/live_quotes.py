"""Live quote updater for the paper-trading monitor."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from futu import OpenQuoteContext, RET_OK

logger = logging.getLogger(__name__)


class LiveQuoteUpdater:
    """Fetches real-time quotes and overlays them onto daily price data."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 11111,
    ) -> None:
        self.host = host
        self.port = port
        self.quote_ctx: OpenQuoteContext | None = None

    def connect(self) -> None:
        if self.quote_ctx is not None:
            return

        self.quote_ctx = OpenQuoteContext(
            host=self.host,
            port=self.port,
        )

    def disconnect(self) -> None:
        if self.quote_ctx is not None:
            self.quote_ctx.close()
            self.quote_ctx = None

    def _require_connection(self) -> None:
        if self.quote_ctx is None:
            raise RuntimeError(
                "Live quote updater is not connected."
            )

    def get_latest_prices(
        self,
        tickers: list[str],
    ) -> dict[str, float]:
        """Return the latest available price for each ticker."""

        self._require_connection()

        if not tickers:
            return {}

        code_list = [f"US.{ticker}" for ticker in tickers]

        ret, data = self.quote_ctx.get_market_snapshot(code_list)

        if ret != RET_OK:
            raise RuntimeError(f"Futu snapshot request failed: {data}")

        latest_prices: dict[str, float] = {}

        for _, row in data.iterrows():
            ticker = str(row.get("code", "")).split(".")[-1]

            price = row.get("last_price", 0)

            if pd.isna(price) or float(price) <= 0:
                price = row.get("close_price", 0)

            if pd.isna(price) or float(price) <= 0:
                continue

            latest_prices[ticker] = float(price)

        return latest_prices

    def overlay_latest_prices(
        self,
        prices: pd.DataFrame,
        latest_prices: dict[str, float],
    ) -> pd.DataFrame:
        """Replace or add today's provisional close for available tickers."""

        if prices.empty or not latest_prices:
            return prices

        latest_date = prices["date"].max()
        today = pd.Timestamp.now().normalize()

        overlay_rows = []

        for ticker, price in latest_prices.items():
            overlay_rows.append(
                {
                    "date": today,
                    "ticker": ticker,
                    "close": price,
                }
            )

        overlay = pd.DataFrame(overlay_rows)

        historical = prices[prices["date"] < today]

        updated = pd.concat(
            [historical, overlay],
            ignore_index=True,
        )

        updated["date"] = pd.to_datetime(updated["date"])
        updated["close"] = pd.to_numeric(
            updated["close"],
            errors="coerce",
        )

        return updated.dropna(
            subset=["date", "ticker", "close"]
        )