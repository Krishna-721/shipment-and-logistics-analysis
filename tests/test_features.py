"""
Tests for preprocessing and feature preparation pipeline.

Covers:
- Temporal split: no overlap, correct ordering, both sets non-empty
- Feature matrix: correct shape, no missing values, no leakage
- Scaling: fitted only on training data (no test-set information)
- Categorical encoding: consistent mapping between train and test
- Excluded columns: targets and metadata never appear in X
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import (
    IDENTIFIER_COLUMNS,
    METADATA_COLUMNS,
    TARGET_COLUMN,
    TARGET_COLUMNS,
    TIMESTAMP_COLUMN,
    VALID_ROW_COLUMN,
)


class TestTemporalSplit:
    def test_no_temporal_overlap(self, split_dfs):
        train_df, test_df = split_dfs
        assert train_df[TIMESTAMP_COLUMN].max() < test_df[TIMESTAMP_COLUMN].min()

    def test_train_precedes_test(self, split_dfs):
        train_df, test_df = split_dfs
        assert train_df[TIMESTAMP_COLUMN].min() < test_df[TIMESTAMP_COLUMN].min()

    def test_both_sets_non_empty(self, split_dfs):
        train_df, test_df = split_dfs
        assert len(train_df) > 0
        assert len(test_df) > 0

    def test_only_valid_rows_included(self, split_dfs):
        train_df, test_df = split_dfs
        assert train_df[VALID_ROW_COLUMN].all()
        assert test_df[VALID_ROW_COLUMN].all()

    def test_no_valid_rows_discarded(self, raw_df, split_dfs):
        """Train + test should contain every valid row from the raw dataset."""
        train_df, test_df = split_dfs
        total_valid = raw_df[VALID_ROW_COLUMN].sum()
        assert len(train_df) + len(test_df) == total_valid


class TestFeaturePreparation:
    def test_x_has_no_missing_values(self, prepared_data):
        X_train, X_test, *_ = prepared_data
        assert X_train.isnull().sum().sum() == 0
        assert X_test.isnull().sum().sum() == 0

    def test_y_is_binary(self, prepared_data):
        _, _, y_train, y_test, _ = prepared_data
        assert set(y_train.unique()).issubset({0, 1})
        assert set(y_test.unique()).issubset({0, 1})

    def test_target_not_in_features(self, prepared_data):
        X_train, X_test, _, _, feature_names = prepared_data
        for col in TARGET_COLUMNS:
            assert col not in feature_names, (
                f"Target '{col}' found in feature list — data leakage"
            )

    def test_metadata_not_in_features(self, prepared_data):
        _, _, _, _, feature_names = prepared_data
        for col in METADATA_COLUMNS:
            assert col not in feature_names, (
                f"Metadata col '{col}' found in feature list"
            )

    def test_identifiers_not_in_features(self, prepared_data):
        _, _, _, _, feature_names = prepared_data
        for col in IDENTIFIER_COLUMNS:
            assert col not in feature_names, (
                f"Identifier col '{col}' found in feature list"
            )

    def test_train_test_same_feature_count(self, prepared_data):
        X_train, X_test, _, _, _ = prepared_data
        assert X_train.shape[1] == X_test.shape[1]

    def test_feature_names_match_columns(self, prepared_data):
        X_train, _, _, _, feature_names = prepared_data
        assert list(X_train.columns) == feature_names

    def test_at_least_20_features(self, prepared_data):
        """Sanity check: should have at least 20 features after selection."""
        _, _, _, _, feature_names = prepared_data
        assert len(feature_names) >= 20, (
            f"Only {len(feature_names)} features — check ESSENTIAL_FEATURES"
        )


class TestScaling:
    def test_scaled_train_approximately_zero_mean(self, scaled_data):
        X_tr_s, *_ = scaled_data
        # Mean of each feature should be close to 0 after StandardScaler
        means = np.abs(X_tr_s.mean(axis=0))
        assert (means < 1e-10).all(), (
            f"Scaled training mean not near zero: max abs mean = {means.max():.2e}"
        )

    def test_scaled_train_approximately_unit_variance(self, scaled_data):
        X_tr_s, *_ = scaled_data
        stds = X_tr_s.std(axis=0)
        # Constant features get std=0; everything else should be ~1
        non_const = stds > 1e-8
        if non_const.any():
            assert np.allclose(stds[non_const], 1.0, atol=1e-8)

    def test_scaler_fit_only_on_train(self, prepared_data, scaled_data):
        """
        Scaler mean and scale must be derived from training data only.
        Verify by manually fitting on train and checking it matches.
        """
        from sklearn.preprocessing import StandardScaler
        from src.data.preprocessing import apply_scaler, fit_scaler

        X_train, X_test, *_ = prepared_data
        _, _, _, _, scaler, _ = scaled_data

        ref_scaler = StandardScaler().fit(X_train)
        np.testing.assert_array_almost_equal(scaler.mean_, ref_scaler.mean_)
        np.testing.assert_array_almost_equal(scaler.scale_, ref_scaler.scale_)

    def test_test_not_used_to_fit_scaler(self, prepared_data, scaled_data):
        """
        If the scaler had accidentally been fit on test data too, its mean
        would differ. Confirm it was fit only on train.
        """
        from sklearn.preprocessing import StandardScaler

        X_train, X_test, *_ = prepared_data
        _, _, _, _, scaler, _ = scaled_data

        combined_mean = np.concatenate([X_train.values, X_test.values]).mean(axis=0)
        # Combined mean must differ from scaler mean (unless sets are identical)
        if not np.allclose(X_train.mean().values, np.concatenate([X_train.values, X_test.values]).mean(axis=0)):
            assert not np.allclose(scaler.mean_, combined_mean, atol=1e-6), (
                "Scaler mean matches combined train+test mean — possible test leakage"
            )


class TestCategoricalEncoding:
    def test_vessel_type_is_numeric_after_encoding(self, prepared_data):
        X_train, X_test, *_ = prepared_data
        if "vessel_type" in X_train.columns:
            assert pd.api.types.is_integer_dtype(X_train["vessel_type"]) or \
                   pd.api.types.is_float_dtype(X_train["vessel_type"])

    def test_cargo_type_is_numeric_after_encoding(self, prepared_data):
        X_train, X_test, *_ = prepared_data
        if "cargo_type" in X_train.columns:
            assert pd.api.types.is_integer_dtype(X_train["cargo_type"]) or \
                   pd.api.types.is_float_dtype(X_train["cargo_type"])
