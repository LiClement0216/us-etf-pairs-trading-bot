"""Persistent state for the paper-trading pipeline."""

from __future__ import annotations

import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any


DEFAULT_STATE = {
    "version": 1,
    "pairs": {},
    "last_run": None,
}


class StateStore:
    """Stores pair-level execution state in a JSON file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return json.loads(json.dumps(DEFAULT_STATE))

        try:
            with self.path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
        except (json.JSONDecodeError, OSError):
            backup = self.path.with_suffix(".corrupt.json")

            if self.path.exists():
                self.path.replace(backup)

            return json.loads(json.dumps(DEFAULT_STATE))

        if not isinstance(loaded, dict):
            return json.loads(json.dumps(DEFAULT_STATE))

        loaded.setdefault("version", 1)
        loaded.setdefault("pairs", {})
        loaded.setdefault("last_run", None)

        return loaded

    def save(self) -> None:
        self.state["last_run"] = datetime.now().isoformat(
            timespec="seconds"
        )

        tmp_path = self.path.with_suffix(".tmp")

        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(
                self.state,
                f,
                indent=2,
                default=str,
            )

        os.replace(tmp_path, self.path)

    def get_pair(self, pair_key: str) -> dict[str, Any]:
        return self.state["pairs"].setdefault(pair_key, {})

    def set_pair(
        self,
        pair_key: str,
        data: dict[str, Any],
    ) -> None:
        self.state["pairs"][pair_key] = data

    def remove_pair(self, pair_key: str) -> None:
        self.state["pairs"].pop(pair_key, None)

    def mark_pending_entry(
        self,
        pair_key: str,
        target_signal: int,
        entry_zscore: float,
        order_ids: list[int],
    ) -> None:
        self.set_pair(
            pair_key,
            {
                "target_signal": target_signal,
                "status": "pending_entry",
                "entry_date": date.today().isoformat(),
                "entry_zscore": entry_zscore,
                "cooldown_remaining": 0,
                "last_order_date": date.today().isoformat(),
                "open_order_ids": order_ids,
            },
        )

    def mark_active(
        self,
        pair_key: str,
        target_signal: int,
        entry_date: str,
        entry_zscore: float,
    ) -> None:
        self.set_pair(
            pair_key,
            {
                "target_signal": target_signal,
                "status": "active",
                "entry_date": entry_date,
                "entry_zscore": entry_zscore,
                "cooldown_remaining": 0,
                "last_order_date": date.today().isoformat(),
                "open_order_ids": [],
            },
        )

    def mark_flat(
        self,
        pair_key: str,
        cooldown_days: int,
    ) -> None:
        self.set_pair(
            pair_key,
            {
                "target_signal": 0,
                "status": "flat",
                "entry_date": None,
                "entry_zscore": None,
                "cooldown_remaining": cooldown_days,
                "last_order_date": date.today().isoformat(),
                "open_order_ids": [],
            },
        )

    def decrement_cooldowns(self) -> None:
        for pair in self.state["pairs"].values():
            remaining = pair.get("cooldown_remaining", 0)

            if remaining > 0:
                pair["cooldown_remaining"] = remaining - 1