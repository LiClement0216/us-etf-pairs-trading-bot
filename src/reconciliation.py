"""Reconciliation and exit-order generation for paper trading."""


from __future__ import annotations


import logging
from typing import Any


import numpy as np
import pandas as pd


from src.execution_paper import PaperExecutionEngine
from src.state import StateStore


from src.trade_history import (
    record_pair_activated,
    record_pair_closed,
)


logger = logging.getLogger(__name__)



def reconcile_pending_entries(
    state_store: StateStore,
    positions: pd.DataFrame,
    execution_engine: PaperExecutionEngine,
) -> None:
    """Promote pending pairs to active once both legs have filled."""


    position_map = _build_position_map(positions)
    open_orders = execution_engine.get_open_orders()


    for pair_key, pair_state in list(
        state_store.state["pairs"].items()
    ):
        if pair_state.get("status") != "pending_entry":
            continue


        asset_a, asset_b = pair_key.split("_")


        saved_order_ids = set(
            int(order_id)
            for order_id in pair_state.get("open_order_ids", [])
        )


        remaining_orders = [
            order
            for order in open_orders
            if int(order["order_id"]) in saved_order_ids
        ]


        if remaining_orders:
            continue


        asset_a_position = position_map.get(asset_a)
        asset_b_position = position_map.get(asset_b)


        if asset_a_position is None or asset_b_position is None:
            continue


        asset_a_quantity = float(
            asset_a_position["signed_quantity"]
        )


        asset_b_quantity = float(
            asset_b_position["signed_quantity"]
        )


        target_signal = int(pair_state.get("target_signal", 0))


        if target_signal == 0:
            continue


        expected_asset_a_direction = np.sign(target_signal)
        expected_asset_b_direction = np.sign(-target_signal)


        actual_asset_a_direction = np.sign(asset_a_quantity)
        actual_asset_b_direction = np.sign(asset_b_quantity)


        if actual_asset_a_direction != expected_asset_a_direction:
            continue


        if actual_asset_b_direction != expected_asset_b_direction:
            continue


        state_store.mark_active(
            pair_key=pair_key,
            target_signal=target_signal,
            entry_date=str(pair_state["entry_date"]),
            entry_zscore=float(pair_state["entry_zscore"]),
        )


        record_pair_activated(
            pair_key=pair_key,
            pair_state=state_store.get_pair(pair_key),
        )


        logger.info(
            "Pair %s is fully filled and now active",
            pair_key,
        )



def reconcile_exits(
    state_store: StateStore,
    positions: pd.DataFrame,
    cooldown_days: int = 10,
    execution_engine: PaperExecutionEngine | None = None,
) -> None:
    """Return active pairs to flat once both legs are closed."""


    position_map = _build_position_map(positions)


    for pair_key, pair_state in list(
        state_store.state["pairs"].items()
    ):
        if pair_state.get("status") != "active":
            continue


        asset_a, asset_b = pair_key.split("_")


        asset_a_position = position_map.get(asset_a)
        asset_b_position = position_map.get(asset_b)


        asset_a_quantity = (
            float(asset_a_position["signed_quantity"])
            if asset_a_position is not None
            else 0.0
        )


        asset_b_quantity = (
            float(asset_b_position["signed_quantity"])
            if asset_b_position is not None
            else 0.0
        )


        if asset_a_quantity == 0 and asset_b_quantity == 0:
            state_store.mark_flat(
                pair_key=pair_key,
                cooldown_days=cooldown_days,
            )


            record_pair_closed(
                pair_key=pair_key,
                pair_state=state_store.get_pair(pair_key),
            )


            logger.info(
                "Pair %s is fully closed and now flat",
                pair_key,
            )



def build_exit_orders(
    state_store: StateStore,
    positions: pd.DataFrame,
    signals: pd.DataFrame,
    config: dict[str, Any],
    execution_engine: PaperExecutionEngine | None = None,
) -> list[dict[str, Any]]:
    """Build closing orders for active pairs meeting exit conditions."""


    exit_orders: list[dict[str, Any]] = []


    strategy_cfg = config["strategy"]


    exit_zscore = float(strategy_cfg["exit_zscore"])
    stop_zscore = float(strategy_cfg["stop_zscore"])
    max_holding_days = int(strategy_cfg["max_holding_days"])


    position_map = _build_position_map(positions)
    signal_map = _build_signal_map(signals)


    for pair_key, pair_state in list(
        state_store.state["pairs"].items()
    ):
        if pair_state.get("status") != "active":
            continue


        asset_a, asset_b = pair_key.split("_")


        asset_a_position = position_map.get(asset_a)
        asset_b_position = position_map.get(asset_b)


        if asset_a_position is None or asset_b_position is None:
            continue


        asset_a_quantity = float(
            asset_a_position["signed_quantity"]
        )


        asset_b_quantity = float(
            asset_b_position["signed_quantity"]
        )


        if asset_a_quantity == 0 or asset_b_quantity == 0:
            continue


        target_signal = int(pair_state.get("target_signal", 0))


        if target_signal == 0:
            continue


        signal = signal_map.get(pair_key)


        if signal is None:
            continue


        latest_zscore = float(signal["latest_zscore"])
        entry_date = pair_state.get("entry_date")


        should_exit = False
        exit_reason = ""


        if target_signal == 1 and latest_zscore >= exit_zscore:
            should_exit = True
            exit_reason = "zscore_exit"


        elif target_signal == -1 and latest_zscore <= -exit_zscore:
            should_exit = True
            exit_reason = "zscore_exit"


        if abs(latest_zscore) >= stop_zscore:
            should_exit = True
            exit_reason = "stop_loss"


        if entry_date:
            holding_days = (
                pd.Timestamp.now().normalize()
                - pd.Timestamp(entry_date)
            ).days


            if holding_days >= max_holding_days:
                should_exit = True
                exit_reason = "max_holding_days"


        if not should_exit:
            continue


        asset_a_exit_side = (
            "SELL"
            if asset_a_quantity > 0
            else "BUY"
        )


        asset_b_exit_side = (
            "SELL"
            if asset_b_quantity > 0
            else "BUY"
        )


        exit_orders.append(
            {
                "ticker": asset_a,
                "side": asset_a_exit_side,
                "quantity": int(abs(asset_a_quantity)),
                "pair_key": pair_key,
                "exit_reason": exit_reason,
            }
        )


        exit_orders.append(
            {
                "ticker": asset_b,
                "side": asset_b_exit_side,
                "quantity": int(abs(asset_b_quantity)),
                "pair_key": pair_key,
                "exit_reason": exit_reason,
            }
        )


    return exit_orders



def _build_position_map(
    positions: pd.DataFrame,
) -> dict[str, pd.Series]:
    if positions.empty:
        return {}


    return {
        str(row["ticker"]): row
        for _, row in positions.iterrows()
    }



def _build_signal_map(
    signals: pd.DataFrame,
) -> dict[str, pd.Series]:
    if signals.empty:
        return {}


    return {
        f"{row['asset_a']}_{row['asset_b']}": row
        for _, row in signals.iterrows()
    }