"""Persistent trade history for the paper-trading pipeline."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


HISTORY_FILE = Path("data/results/trade_history.jsonl")


def append_trade_event(event: dict[str, Any]) -> None:
    """Append one trade event to the persistent JSONL history."""

    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

    event = {
        **event,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    with HISTORY_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, default=str) + "\n")


def load_trade_history() -> pd.DataFrame:
    """Load all recorded trade events."""

    if not HISTORY_FILE.exists():
        return pd.DataFrame()

    records = []

    with HISTORY_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return pd.DataFrame(records)


def record_order_submitted(
    order: dict[str, Any],
) -> None:
    append_trade_event(
        {
            "event_type": "order_submitted",
            **order,
        }
    )


def record_pair_activated(
    pair_key: str,
    pair_state: dict[str, Any],
) -> None:
    append_trade_event(
        {
            "event_type": "pair_activated",
            "pair_key": pair_key,
            **pair_state,
        }
    )


def record_pair_closed(
    pair_key: str,
    pair_state: dict[str, Any],
) -> None:
    append_trade_event(
        {
            "event_type": "pair_closed",
            "pair_key": pair_key,
            **pair_state,
        }
    )