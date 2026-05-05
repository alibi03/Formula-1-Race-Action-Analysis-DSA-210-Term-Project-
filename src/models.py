from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Outcome-derived columns: do not use as predictors for high_action / mean_abs_pos_change.
_OUTCOME_LEAK_COLS = frozenset(
    {
        "mean_abs_pos_change",
        "total_abs_pos_change",
        "max_pos_gained",
        "pos_change_std",
        "high_action",
    }
)

# Predictors available without using post-race position summaries.
DEFAULT_LEAKFREE_FEATURES: tuple[str, ...] = (
    "year",
    "round",
    "n_starters",
    "n_classified",
    "n_dnf",
    "n_laps",
    "avg_air_temp",
    "avg_track_temp",
    "avg_humidity",
    "avg_wind_speed",
    "rain_pct",
    "any_rain",
    "circuit_length_km",
    "circuit_turns",
    "circuit_type",
    "location",
    "country",
)

OPTIONAL_IN_RACE_FEATURES: tuple[str, ...] = ("n_pit_stops", "n_compounds_used")


@dataclass(frozen=True)
class TrainTestData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def modeling_columns(
    include_in_race_features: bool = False,
) -> list[str]:
    cols = list(DEFAULT_LEAKFREE_FEATURES)
    if include_in_race_features:
        cols.extend(OPTIONAL_IN_RACE_FEATURES)
    return cols


def modeling_matrix(
    df: pd.DataFrame,
    target: str,
    *,
    include_in_race_features: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    feats = modeling_columns(include_in_race_features=include_in_race_features)
    missing = [c for c in feats + [target] if c not in df.columns]
    if missing:
        raise KeyError(f"DataFrame missing columns: {missing}")
    frame = df[feats + [target]].copy()
    for c in ("circuit_type", "location", "country"):
        if c in frame.columns:
            frame[c] = frame[c].fillna("unknown").astype(str)
    frame = frame.dropna(subset=[target])
    X = frame[feats]
    y = frame[target]
    return X, y


def make_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.2,
    seed: int = 42,
    stratify: bool = False,
) -> TrainTestData:
    strat = y if stratify and y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=strat
    )
    return TrainTestData(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    from pandas.api.types import is_numeric_dtype, is_bool_dtype

    cat_cols = [c for c in X.columns if not is_numeric_dtype(X[c]) and not is_bool_dtype(X[c])]
    num_cols = [c for c in X.columns if c not in cat_cols]
    transformers: list[tuple[str, Pipeline | SimpleImputer, list[str]]] = []
    if num_cols:
        transformers.append(("num", SimpleImputer(strategy="median"), num_cols))
    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            )
        )
    if not transformers:
        raise ValueError("No feature columns to preprocess.")
    return ColumnTransformer(transformers=transformers)


def fit_rf_classifier(train_test: TrainTestData) -> tuple[Pipeline, dict]:
    pre = _build_preprocessor(train_test.X_train)
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(train_test.X_train, train_test.y_train)
    proba = pipe.predict_proba(train_test.X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(train_test.y_test, proba)),
        "report": classification_report(
            train_test.y_test, preds, output_dict=True, zero_division=0
        ),
    }
    return pipe, metrics


def fit_rf_regressor(train_test: TrainTestData) -> tuple[Pipeline, dict]:
    pre = _build_preprocessor(train_test.X_train)
    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(train_test.X_train, train_test.y_train)
    pred = pipe.predict(train_test.X_test)
    metrics = {
        "r2": float(r2_score(train_test.y_test, pred)),
        "mae": float(mean_absolute_error(train_test.y_test, pred)),
    }
    return pipe, metrics


def cross_val_roc_auc_classifier(X: pd.DataFrame, y: pd.Series, *, n_splits: int = 5, seed: int = 42) -> float:
    pre = _build_preprocessor(X)
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=3,
        random_state=seed,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    pipe = Pipeline([("pre", pre), ("model", model)])
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
    return float(np.mean(scores))


def cross_val_r2_regressor(X: pd.DataFrame, y: pd.Series, *, n_splits: int = 5, seed: int = 42) -> float:
    from sklearn.model_selection import KFold

    pre = _build_preprocessor(X)
    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=3,
        random_state=seed,
        n_jobs=-1,
    )
    pipe = Pipeline([("pre", pre), ("model", model)])
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="r2", n_jobs=1)
    return float(np.mean(scores))


def rf_feature_importance_df(pipe: Pipeline, *, top: int = 20) -> pd.DataFrame:
    pre: ColumnTransformer = pipe.named_steps["pre"]
    model = pipe.named_steps["model"]
    names = pre.get_feature_names_out()
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        raise ValueError("Model has no feature_importances_.")
    df = pd.DataFrame({"feature": names, "importance": imp}).sort_values("importance", ascending=False)
    return df.head(top).reset_index(drop=True)


def assert_no_outcome_leak(feature_cols: Iterable[str]) -> None:
    bad = set(feature_cols) & _OUTCOME_LEAK_COLS
    if bad:
        raise ValueError(f"Outcome leakage in features: {sorted(bad)}")
