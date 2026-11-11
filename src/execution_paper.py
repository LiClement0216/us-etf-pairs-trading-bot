"""Futu paper-trading execution engine.

This module handles:
- Connecting to OpenD
- Reading paper account positions
- Reading and reconciling orders
- Building pair orders
- Submitting paper limit orders
- Cancelling stale orders
- Persisting execution state
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any
from src.trade_history import record_order_submitted
import pandas as pd
from futu import (
    OpenSecTradeContext,
    TrdEnv,
    TrdMarket,
    TrdSide,
    OrderType,
    TimeInForce,
    ModifyOrderOp,
    RET_OK,
)

from src.state import StateStore

logger = logging.getLogger(__name__)


ACTIVE_ORDER_STATUSES = {
    "SUBMITTING",
    "SUBMITTED",
    "FILLED_PART",
    "PART_FILLED",
    "CANCELLED_PART",
}

FINAL_ORDER_STATUSES = {
    "FILLED_ALL",
    "CANCELLED_ALL",
    "FAILED",
    "DISABLED",
    "DELETED",
}


@dataclass
class OrderInstruction:
    ticker: str
    side: str
    quantity: int
    limit_price: float
    pair_key: str
    target_signal: int
    entry_zscore: float
    signal_date: str


class PaperExecutionEngine:
    """Executes and reconciles paper trades through Futu OpenD."""

    def __init__(
        self,
        config: dict[str, Any],
        state_store: StateStore,
        prices: pd.DataFrame | None = None,
    ) -> None:
        self.config = config
        self.state_store = state_store
        self.prices = prices

        execution_cfg = config["execution"]
        self.mode = execution_cfg.get("mode", "paper")
        self.host = execution_cfg.get("host", "127.0.0.1")
        self.port = int(execution_cfg.get("port", 11111))
        self.order_type = execution_cfg.get("order_type", "limit")
        self.price_buffer_bps = float(execution_cfg.get("price_buffer_bps", 5))
        self.max_order_notional = float(
            execution_cfg.get("max_order_notional", 500)
        )
        self.max_orders_per_day = int(
            execution_cfg.get("max_orders_per_day", 10)
        )
        self.cancel_stale_orders_enabled = bool(
            execution_cfg.get("cancel_stale_orders", True)
        )
        self.submit_orders_enabled = bool(
            execution_cfg.get("submit_orders", False)
        )
        self.require_market_open = bool(
            execution_cfg.get("require_market_open", True)
        )

        if self.mode != "paper":
            raise ValueError(
                "This execution engine supports paper trading only. "
                "Set execution.mode to 'paper'."
            )

        self.trd_env = TrdEnv.SIMULATE
        self.trade_ctx: OpenSecTradeContext | None = None
        self.account_id: int | None = None

    def connect(self) -> None:
        """Connect to OpenD and locate the paper trading account."""

        if self.trade_ctx is not None:
            return

        logger.info("Connecting to Futu OpenD at %s:%s", self.host, self.port)

        self.trade_ctx = OpenSecTradeContext(
            filter_trdmarket=TrdMarket.US,
            host=self.host,
            port=self.port,
        )

        ret, accounts = self.trade_ctx.get_acc_list()

        if ret != RET_OK:
            raise RuntimeError(f"Failed to get Futu accounts: {accounts}")

        paper_accounts = accounts[
            accounts["trd_env"].astype(str).str.upper() == "SIMULATE"
        ]

        if paper_accounts.empty:
            raise RuntimeError(
                "No Futu paper trading account found. "
                "Open a simulated trading account in the Futu app first."
            )

        self.account_id = int(paper_accounts.iloc[0]["acc_id"])
        logger.info(
            "Connected to Futu paper account %s",
            self.account_id,
        )

    def disconnect(self) -> None:
        if self.trade_ctx is not None:
            self.trade_ctx.close()
            self.trade_ctx = None
            self.account_id = None

    def _require_connection(self) -> None:
        if self.trade_ctx is None or self.account_id is None:
            raise RuntimeError(
                "Execution engine is not connected. Call connect() first."
            )

    def get_positions(self) -> pd.DataFrame:
        """Return normalized current paper positions."""

        self._require_connection()

        ret, data = self.trade_ctx.position_list_query(
            trd_env=self.trd_env,
            acc_id=self.account_id,
            currency="USD",
        )

        if ret != RET_OK:
            raise RuntimeError(f"Futu position query failed: {data}")

        rows = []

        for _, row in data.iterrows():
            ticker = str(row.get("code", "")).split(".")[-1]
            quantity = float(row.get("qty", 0) or 0)

            if quantity == 0:
                continue

            rows.append(
                {
                    "ticker": ticker,
                    "signed_quantity": quantity,
                    "position_side": "LONG" if quantity > 0 else "SHORT",
                    "cost_price": float(row.get("cost_price", 0) or 0),
                    "market_value": float(row.get("market_val", 0) or 0),
                }
            )

        return pd.DataFrame(
            rows,
            columns=[
                "ticker",
                "signed_quantity",
                "position_side",
                "cost_price",
                "market_value",
            ],
        )

    def get_orders(self) -> list[dict[str, Any]]:
        """Return normalized recent and open orders."""

        self._require_connection()

        ret, data = self.trade_ctx.order_list_query(
            trd_env=self.trd_env,
            acc_id=self.account_id,
        )

        if ret != RET_OK:
            raise RuntimeError(f"Futu order query failed: {data}")

        orders = []

        for _, row in data.iterrows():
            orders.append(
                {
                    "order_id": str(row.get("order_id", "")),
                    "ticker": str(row.get("code", "")).split(".")[-1],
                    "status": str(row.get("order_status", "")),
                    "quantity": float(row.get("qty", 0) or 0),
                    "filled_quantity": float(
                        row.get("filled_qty", 0) or 0
                    ),
                    "price": float(row.get("price", 0) or 0),
                    "side": str(row.get("trd_side", "")),
                    "create_time": str(row.get("create_time", "")),
                    "update_time": str(row.get("update_time", "")),
                }
            )

        return orders

    def get_open_orders(self) -> list[dict[str, Any]]:
        """Return orders that still represent live exposure or pending fills."""

        return [
            order
            for order in self.get_orders()
            if order["status"] in ACTIVE_ORDER_STATUSES
        ]

    def get_orders_for_ticker(self, ticker: str) -> list[dict[str, Any]]:
        return [
            order
            for order in self.get_open_orders()
            if order["ticker"] == ticker
        ]

    def has_pending_orders(self, tickers: list[str]) -> bool:
        open_orders = self.get_open_orders()
        return any(order["ticker"] in tickers for order in open_orders)

    def cancel_order(self, order_id: str) -> None:
        """Cancel an unfilled order."""

        self._require_connection()

        ret, data = self.trade_ctx.modify_order(
            modify_order_op=ModifyOrderOp.CANCEL,
            order_id=order_id,
            qty=0,
            price=0,
            trd_env=self.trd_env,
            acc_id=self.account_id,
        )

        if ret != RET_OK:
            raise RuntimeError(
                f"Futu cancel failed for order {order_id}: {data}"
            )

        logger.info("Cancelled Futu order %s", order_id)

    def cancel_stale_orders(self, max_age_days: int = 3) -> list[str]:
        """Cancel open orders older than the configured age."""

        if not self.cancel_stale_orders_enabled:
            return []

        cancelled: list[str] = []
        today = date.today()

        for order in self.get_open_orders():
            created = order.get("create_time", "")

            if not created:
                continue

            try:
                order_date = datetime.fromisoformat(
                    created.replace("Z", "+00:00")
                ).date()
            except ValueError:
                continue

            if (today - order_date).days >= max_age_days:
                try:
                    self.cancel_order(order["order_id"])
                    cancelled.append(order["order_id"])
                except Exception as exc:
                    logger.warning(
                        "Failed to cancel stale order %s: %s",
                        order["order_id"],
                        exc,
                    )

        return cancelled

    def get_latest_price(self, ticker: str) -> float:
        """Get the latest close price from normalized price data."""

        if self.prices is None or self.prices.empty:
            raise RuntimeError(
                "Normalized price data was not provided to "
                "PaperExecutionEngine."
            )

        required_columns = {"date", "ticker", "close"}

        if not required_columns.issubset(self.prices.columns):
            raise ValueError(
                "Normalized price data must contain "
                f"{sorted(required_columns)}. "
                f"Actual columns: {list(self.prices.columns)}"
            )

        ticker_prices = self.prices[
            self.prices["ticker"] == ticker
        ].sort_values("date")

        if ticker_prices.empty:
            raise ValueError(f"No price data found for {ticker}")

        latest_price = float(ticker_prices["close"].iloc[-1])

        if not math.isfinite(latest_price) or latest_price <= 0:
            raise ValueError(
                f"Invalid latest price for {ticker}: {latest_price}"
            )

        return latest_price

    def build_limit_price(
        self,
        ticker: str,
        side: str,
    ) -> float:
        """Build a marketable limit price with a configurable buffer."""

        latest_price = self.get_latest_price(ticker)
        buffer = latest_price * (self.price_buffer_bps / 10_000)

        if side == "BUY":
            return round(latest_price + buffer, 2)

        if side == "SELL":
            return round(latest_price - buffer, 2)

        raise ValueError(f"Unsupported order side: {side}")

    def calculate_quantity(
        self,
        ticker: str,
        capital: float,
    ) -> int:
        """Calculate a whole-share quantity from a notional budget."""

        price = self.get_latest_price(ticker)

        if price <= 0:
            return 0

        quantity = int(math.floor(capital / price))

        return max(quantity, 0)

    def build_order(
        self,
        ticker: str,
        target_signal: int,
        capital: float,
        pair_key: str,
        entry_zscore: float,
        signal_date: str,
    ) -> OrderInstruction | None:
        """Build one limit-order instruction."""

        if target_signal == 0:
            return None

        side = "BUY" if target_signal > 0 else "SELL"
        quantity = self.calculate_quantity(ticker, capital)

        if quantity <= 0:
            logger.warning(
                "Skipping %s because calculated quantity is zero",
                ticker,
            )
            return None

        price = self.build_limit_price(ticker, side)

        return OrderInstruction(
            ticker=ticker,
            side=side,
            quantity=quantity,
            limit_price=price,
            pair_key=pair_key,
            target_signal=target_signal,
            entry_zscore=entry_zscore,
            signal_date=signal_date,
        )

    def submit_order(self, instruction: OrderInstruction) -> dict[str, Any]:
        """Submit one paper limit order."""

        self._require_connection()

        side = (
            TrdSide.BUY
            if instruction.side == "BUY"
            else TrdSide.SELL
        )

        ret, data = self.trade_ctx.place_order(
            price=instruction.limit_price,
            qty=instruction.quantity,
            code=f"US.{instruction.ticker}",
            trd_side=side,
            order_type=OrderType.NORMAL,
            trd_env=self.trd_env,
            acc_id=self.account_id,
            time_in_force=TimeInForce.DAY,
        )

        if ret != RET_OK:
            raise RuntimeError(
                f"Futu order submission failed for "
                f"{instruction.ticker}: {data}"
            )

        raw_order_id = data.get("order_id", "")

        if isinstance(raw_order_id, pd.Series):
            raw_order_id = raw_order_id.iloc[0]

        order_id = str(raw_order_id)

        logger.info(
            "Submitted paper order: %s %s %s shares @ %s | order_id=%s",
            instruction.side,
            instruction.ticker,
            instruction.quantity,
            instruction.limit_price,
            order_id,
        )
        record_order_submitted(
            {
                "order_id": order_id,
                "ticker": instruction.ticker,
                "side": instruction.side,
                "quantity": instruction.quantity,
                "limit_price": instruction.limit_price,
                "pair_key": instruction.pair_key,
                "target_signal": instruction.target_signal,
                "entry_zscore": instruction.entry_zscore,
                "signal_date": instruction.signal_date,
            }
        )
        return {
            "order_id": order_id,
            "ticker": instruction.ticker,
            "side": instruction.side,
            "quantity": instruction.quantity,
            "limit_price": instruction.limit_price,
            "pair_key": instruction.pair_key,
            "target_signal": instruction.target_signal,
            "entry_zscore": instruction.entry_zscore,
            "signal_date": instruction.signal_date,
        }

    def submit_orders(
        self,
        order_plan: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """Submit a DataFrame of order instructions."""

        submitted: list[dict[str, Any]] = []

        if not self.submit_orders_enabled:
            return submitted
    
        if order_plan.empty:
            return submitted

        if len(order_plan) > self.max_orders_per_day:
            raise ValueError(
                f"Order plan has {len(order_plan)} orders, exceeding "
                f"max_orders_per_day={self.max_orders_per_day}"
            )

        for _, row in order_plan.iterrows():
            instruction = OrderInstruction(
                ticker=str(row["ticker"]),
                side=str(row["side"]),
                quantity=int(row["quantity"]),
                limit_price=float(row["limit_price"]),
                pair_key=str(row["pair_key"]),
                target_signal=int(row["target_signal"]),
                entry_zscore=float(row["entry_zscore"]),
                signal_date=str(row["signal_date"]),
            )

            result = self.submit_order(instruction)
            submitted.append(result)

        return submitted

    def submit_exit_orders(
        self,
        exit_orders: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Submit closing orders for active pairs."""

        submitted: list[dict[str, Any]] = []

        if not exit_orders:
            return submitted

        if not self.submit_orders_enabled:
            logger.info("Order submission is disabled")
            return submitted

        for order in exit_orders:
            ticker = str(order["ticker"])
            side = str(order["side"])
            quantity = int(order["quantity"])
            pair_key = str(order["pair_key"])
            exit_reason = str(order["exit_reason"])

            if quantity <= 0:
                continue

            price = self.build_limit_price(ticker, side)

            instruction = OrderInstruction(
                ticker=ticker,
                side=side,
                quantity=quantity,
                limit_price=price,
                pair_key=pair_key,
                target_signal=0,
                entry_zscore=0.0,
                signal_date=date.today().isoformat(),
            )

            result = self.submit_order(instruction)

            result["exit_reason"] = exit_reason
            submitted.append(result)

        return submitted

    def submit_exit_orders(
        self,
        exit_orders: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Submit closing orders for active pairs."""

        submitted: list[dict[str, Any]] = []

        if not exit_orders:
            return submitted

        if not self.submit_orders_enabled:
            logger.info("Order submission is disabled")
            return submitted

        for order in exit_orders:
            ticker = str(order["ticker"])
            side = str(order["side"])
            quantity = int(order["quantity"])
            pair_key = str(order["pair_key"])
            exit_reason = str(order["exit_reason"])

            if quantity <= 0:
                continue

            price = self.build_limit_price(ticker, side)

            instruction = OrderInstruction(
                ticker=ticker,
                side=side,
                quantity=quantity,
                limit_price=price,
                pair_key=pair_key,
                target_signal=0,
                entry_zscore=0.0,
                signal_date=date.today().isoformat(),
            )

            result = self.submit_order(instruction)

            result["exit_reason"] = exit_reason
            submitted.append(result)

        return submitted