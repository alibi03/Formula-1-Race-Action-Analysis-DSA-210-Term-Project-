from __future__ import annotations

import pandas as pd


def basic_missingness(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.isna()
        .mean()
        .sort_values(ascending=False)
        .rename("missing_frac")
        .to_frame()
    )

