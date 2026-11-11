from pathlib import Path

import pandas as pd


from pathlib import Path

import pandas as pd


def create_timestamp() -> str:
    return pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")


def save_results(
    config: dict,
    selected_pairs: list[dict],
    metrics: pd.DataFrame,
    portfolio: pd.DataFrame | None,
) -> dict[str, Path | None]:
    results_dir = Path(config["project"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = create_timestamp()
    project_name = config["project"]["name"]

    pair_rows = []

    for pair in selected_pairs:
        pair_rows.append(
            {
                "asset_a": pair["asset_a"],
                "asset_b": pair["asset_b"],
                "correlation": pair["correlation"],
                "coint_p_value": pair["coint_p_value"],
                "adf_p_value": pair["adf_p_value"],
                "hedge_ratio": pair["hedge_ratio"],
                "hedge_ratio_cv": pair["hedge_ratio_cv"],
                "zscore_crossings": pair["zscore_crossings"],
            }
        )

    selected_pairs_path = results_dir / (
        f"{project_name}_selected_pairs_{timestamp}.csv"
    )

    metrics_path = results_dir / (
        f"{project_name}_pair_metrics_{timestamp}.csv"
    )

    portfolio_path = None

    pd.DataFrame(pair_rows).to_csv(
        selected_pairs_path,
        index=False,
    )

    metrics.to_csv(
        metrics_path,
        index=False,
    )

    if portfolio is not None:
        portfolio_path = results_dir / (
            f"{project_name}_portfolio_backtest_{timestamp}.csv"
        )

        portfolio.to_csv(
            portfolio_path,
            index=True,
        )

    return {
        "selected_pairs": selected_pairs_path,
        "pair_metrics": metrics_path,
        "portfolio_backtest": portfolio_path,
    }

def save_rejected_pairs(
    config: dict,
    rejected_pairs: list[dict],
) -> Path:
    results_dir = Path(config["project"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    project_name = config["project"]["name"]

    path = results_dir / (
        f"{project_name}_rejected_pairs_{timestamp}.csv"
    )

    pd.DataFrame(rejected_pairs).to_csv(path, index=False)

    return path