"""Daily scheduler for the pairs-trading pipeline."""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "strategy.yaml"
PYTHON = sys.executable


def run_daily():
    command = [
        PYTHON,
        "-B",
        str(PROJECT_ROOT / "pipeline.py"),
        "--config",
        str(CONFIG_PATH),
        "--mode",
        "daily",
    ]

    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Running daily pipeline")

    result = subprocess.run(command, cwd=PROJECT_ROOT)

    print(
        f"[{datetime.now():%Y-%m-%d %H:%M:%S}] "
        f"Daily pipeline finished with exit code {result.returncode}"
    )

    return result.returncode


def run_forever(interval_seconds: int = 3600):
    while True:
        try:
            run_daily()
        except Exception as exc:
            print(f"Daily pipeline failed: {exc}")

        time.sleep(interval_seconds)


if __name__ == "__main__":
    run_daily()