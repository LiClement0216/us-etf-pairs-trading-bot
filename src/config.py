from pathlib import Path

import yaml


def load_config(config_path: str) -> dict:
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    required_sections = [
        "project",
        "data",
        "universe",
        "pair_selection",
        "strategy",
        "backtest",
        "portfolio",
        "execution",
    ]

    missing = [
        section for section in required_sections
        if section not in config
    ]

    if missing:
        raise ValueError(f"Missing config sections: {missing}")

    return config