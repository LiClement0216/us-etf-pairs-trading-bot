import pandas as pd


def generate_pair_signals(
    spread: pd.Series,
    zscore: pd.Series,
    entry_zscore: float = 2.0,
    exit_zscore: float = 0.0,
    stop_zscore: float = 3.5,
    max_holding_days: int = 30,
    cooldown_days: int = 10,
    market_filter: pd.Series | None = None,
) -> pd.DataFrame:
    signal = pd.Series(0, index=spread.index, dtype=int)

    position = 0
    entry_date = None
    last_exit_date = None

    for date in spread.index:
        current_z = zscore.loc[date]

        if pd.isna(current_z):
            signal.loc[date] = position
            continue

        can_enter = (
            last_exit_date is None
            or (date - last_exit_date).days >= cooldown_days
        )

        if market_filter is not None:
            market_ok = market_filter.reindex(spread.index).loc[date]

            if pd.notna(market_ok) and not market_ok:
                can_enter = False

        if position == 0 and can_enter:
            if current_z <= -entry_zscore:
                position = 1
                entry_date = date

            elif current_z >= entry_zscore:
                position = -1
                entry_date = date

        elif position == 1:
            holding_days = (
                spread.index.get_loc(date)
                - spread.index.get_loc(entry_date)
            )

            exit_now = (
                current_z >= exit_zscore
                or current_z <= -stop_zscore
                or holding_days >= max_holding_days
            )

            if exit_now:
                position = 0
                entry_date = None
                last_exit_date = date

        elif position == -1:
            holding_days = (
                spread.index.get_loc(date)
                - spread.index.get_loc(entry_date)
            )

            exit_now = (
                current_z <= exit_zscore
                or current_z >= stop_zscore
                or holding_days >= max_holding_days
            )

            if exit_now:
                position = 0
                entry_date = None
                last_exit_date = date

        signal.loc[date] = position

    return pd.DataFrame(
        {
            "spread": spread,
            "zscore": zscore,
            "signal": signal,
        }
    )