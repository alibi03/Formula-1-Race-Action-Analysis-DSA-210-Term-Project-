from __future__ import annotations

import pandas as pd


def compute_position_change_proxy(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a simple position-change proxy from race result fields.

    Expected columns (typical): `GridPosition` (start), `Position` (finish), `Driver`.
    This will evolve as we settle on the best definition using lap-level data.
    """
    df = results_df.copy()
    if "GridPosition" in df.columns and "Position" in df.columns:
        df["pos_delta"] = df["GridPosition"].astype("float") - df["Position"].astype("float")
    return df


def label_high_action_races(race_level_df: pd.DataFrame, action_col: str, quantile: float = 0.75) -> pd.DataFrame:
    df = race_level_df.copy()
    threshold = df[action_col].quantile(quantile)
    df["high_action"] = (df[action_col] >= threshold).astype(int)
    df.attrs["high_action_threshold"] = float(threshold)
    return df

