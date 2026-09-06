"""Data cleaning and preprocessing pipelines."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    ESSENTIAL_FEATURES,
    IDENTIFIER_COLUMNS,
    METADATA_COLUMNS,
    RANDOM_STATE,
    TARGET_COLUMN,
    TARGET_COLUMNS,
    TEST_START_DATE,
    TIMESTAMP_COLUMN,
    TRAIN_END_DATE,
    VALID_ROW_COLUMN,
)


# ---------------------------------------------------------------------------
# Temporal split
# ---------------------------------------------------------------------------

def create_temporal_split(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (train_df, test_df) using a strict chronological cut.

    Only valid-training rows are included (is_valid_training_row == True).

    Training : 2026-01-01 00:00 – TRAIN_END_DATE   (first 8 days)
    Testing  : TEST_START_DATE  – end of dataset   (final ~2.1 days)

    The distributional shift (train ~20.6 % congested, test ~45.2 % congested)
    is real and reflects queue build-up over the simulation window.  It is
    documented rather than masked.
    """
    valid_df = df[df[VALID_ROW_COLUMN]].copy()

    train_df = valid_df[valid_df[TIMESTAMP_COLUMN] <= TRAIN_END_DATE].copy()
    test_df  = valid_df[valid_df[TIMESTAMP_COLUMN] >= TEST_START_DATE].copy()

    assert len(train_df) + len(test_df) == len(valid_df), (
        "Train and test sets do not cover all valid rows."
    )
    assert train_df[TIMESTAMP_COLUMN].max() < test_df[TIMESTAMP_COLUMN].min(), (
        "Temporal overlap detected between train and test sets."
    )

    print("Temporal split")
    print(f"  Train : {len(train_df):>5,} rows  "
          f"({train_df[TARGET_COLUMN].mean()*100:.1f}% congested)  "
          f"{train_df[TIMESTAMP_COLUMN].min().date()} → "
          f"{train_df[TIMESTAMP_COLUMN].max().date()}")
    print(f"  Test  : {len(test_df):>5,} rows  "
          f"({test_df[TARGET_COLUMN].mean()*100:.1f}% congested)  "
          f"{test_df[TIMESTAMP_COLUMN].min().date()} → "
          f"{test_df[TIMESTAMP_COLUMN].max().date()}")

    return train_df, test_df


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------

def _encode_categoricals(
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
    cat_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Ordinal-encode categorical columns fitted on training data only."""
    train_out = train_df.copy()
    test_out  = test_df.copy()

    for col in cat_cols:
        categories = sorted(train_out[col].astype(str).unique())
        mapping = {v: i for i, v in enumerate(categories)}

        train_out[col] = train_out[col].astype(str).map(mapping)
        # Unseen test values → -1 (safe for tree models; rare for synthetic data)
        test_out[col]  = test_out[col].astype(str).map(mapping).fillna(-1).astype(int)

    return train_out, test_out


def prepare_features(
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """
    Build feature matrices and target vectors.

    Returns
    -------
    X_train, X_test, y_train, y_test, feature_names
    """
    # Columns that must never be model inputs
    excluded = set(IDENTIFIER_COLUMNS + TARGET_COLUMNS + METADATA_COLUMNS)

    # Start from the essential feature list; keep only columns that exist
    base_features = [f for f in ESSENTIAL_FEATURES if f in train_df.columns]

    # Add categorical features that are in the dataset and not already listed
    extra_cat = [
        f for f in CATEGORICAL_FEATURES
        if f in train_df.columns and f not in base_features
    ]

    all_feature_cols = base_features + extra_cat

    # Safety check – no target or metadata columns should sneak in
    leaking = [c for c in all_feature_cols if c in excluded]
    if leaking:
        raise ValueError(f"Feature list contains excluded columns: {leaking}")

    # Encode categoricals (fit on train only)
    cat_present = [c for c in extra_cat if c in train_df.columns]
    if cat_present:
        train_df, test_df = _encode_categoricals(train_df, test_df, cat_present)
        print(f"  Encoded {len(cat_present)} categorical feature(s): {cat_present}")

    X_train = train_df[all_feature_cols].copy()
    X_test  = test_df[all_feature_cols].copy()
    y_train = train_df[TARGET_COLUMN].copy().astype(int)
    y_test  = test_df[TARGET_COLUMN].copy().astype(int)

    # Sanity checks
    assert X_train.isnull().sum().sum() == 0, "Missing values in X_train"
    assert X_test.isnull().sum().sum()  == 0, "Missing values in X_test"
    assert set(y_train.unique()).issubset({0, 1}), "y_train has non-binary values"
    assert set(y_test.unique()).issubset({0, 1}),  "y_test has non-binary values"

    feature_names = list(X_train.columns)

    print(f"  Features : {len(feature_names)}")
    print(f"  X_train  : {X_train.shape}  — y_train positive rate: {y_train.mean():.1%}")
    print(f"  X_test   : {X_test.shape}  — y_test  positive rate: {y_test.mean():.1%}")

    return X_train, X_test, y_train, y_test, feature_names


# ---------------------------------------------------------------------------
# Scaling (used for Logistic Regression only)
# ---------------------------------------------------------------------------

def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """Fit a StandardScaler on training data and return it."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def apply_scaler(
    scaler: StandardScaler,
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform train and test sets with a pre-fitted scaler."""
    return scaler.transform(X_train), scaler.transform(X_test)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def get_excluded_columns() -> List[str]:
    """Columns that must never be used as model inputs."""
    return IDENTIFIER_COLUMNS + TARGET_COLUMNS + METADATA_COLUMNS
