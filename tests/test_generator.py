"""
Tests for the synthetic data generator.

Covers:
- Reproducibility with fixed seed
- Correct number of rows and ports
- No missing values in valid training rows
- No negative physical quantities
- Valid training row marker
- Temporal ordering within each port
- Target is binary
- Target separation from features
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import (
    METADATA_COLUMNS,
    TARGET_COLUMNS,
    TIMESTAMP_COLUMN,
    VALID_ROW_COLUMN,
)


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def generated_df():
    """Run the real generator with default parameters (seed=42, small scale)."""
    from src.data.generator import generate_dataset
    return generate_dataset(n_ports=4, hours_per_port=30, seed=42)


# ── tests ─────────────────────────────────────────────────────────────────────

class TestReproducibility:
    def test_same_seed_gives_identical_output(self):
        from src.data.generator import generate_dataset
        df1 = generate_dataset(n_ports=4, hours_per_port=20, seed=42)
        df2 = generate_dataset(n_ports=4, hours_per_port=20, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seed_gives_different_output(self):
        from src.data.generator import generate_dataset
        df1 = generate_dataset(n_ports=4, hours_per_port=20, seed=42)
        df2 = generate_dataset(n_ports=4, hours_per_port=20, seed=99)
        assert not df1["queue_length"].equals(df2["queue_length"])


class TestShape:
    def test_row_count(self, generated_df):
        """Expect exactly n_ports * hours_per_port rows."""
        assert len(generated_df) == 4 * 30

    def test_port_count(self, generated_df):
        assert generated_df["port_id"].nunique() == 4

    def test_all_ports_present(self, generated_df):
        expected = {f"PORT_{i:02d}" for i in range(1, 5)}
        assert set(generated_df["port_id"].unique()) == expected


class TestMissingValues:
    def test_no_missing_in_valid_rows(self, generated_df):
        valid = generated_df[generated_df[VALID_ROW_COLUMN]]
        assert valid.isnull().sum().sum() == 0, (
            f"Missing values found in valid rows:\n"
            f"{valid.isnull().sum()[valid.isnull().sum() > 0]}"
        )

    def test_no_missing_in_full_dataset(self, generated_df):
        # Even incomplete-horizon rows should not have NaN in feature columns
        # (only targets may be absent, but generator fills them anyway)
        non_target = [c for c in generated_df.columns if c not in TARGET_COLUMNS + METADATA_COLUMNS]
        assert generated_df[non_target].isnull().sum().sum() == 0


class TestPhysicalConstraints:
    def test_queue_length_non_negative(self, generated_df):
        assert (generated_df["queue_length"] >= 0).all()

    def test_waiting_time_non_negative(self, generated_df):
        assert (generated_df["avg_waiting_time_hours"] >= 0).all()

    def test_future_delay_non_negative(self, generated_df):
        assert (generated_df["future_delay_hours"] >= 0).all()

    def test_berth_utilization_bounded(self, generated_df):
        assert (generated_df["berth_utilization"] >= 0).all()
        assert (generated_df["berth_utilization"] <= 1.0).all()

    def test_arrivals_non_negative(self, generated_df):
        assert (generated_df["arrivals_last_1h"] >= 0).all()

    def test_available_berths_non_negative(self, generated_df):
        assert (generated_df["available_berths"] >= 0).all()


class TestValidTrainingRows:
    def test_invalid_rows_count(self, generated_df):
        """Last 6 rows per port should be marked invalid."""
        invalid = generated_df[~generated_df[VALID_ROW_COLUMN]]
        assert len(invalid) == 4 * 6, (
            f"Expected {4 * 6} invalid rows, got {len(invalid)}"
        )

    def test_invalid_rows_are_at_end_of_each_port(self, generated_df):
        for port in generated_df["port_id"].unique():
            port_data = generated_df[generated_df["port_id"] == port].sort_values(TIMESTAMP_COLUMN)
            valid_flags = port_data[VALID_ROW_COLUMN].values
            # All False values should be at the tail
            first_false = next((i for i, v in enumerate(valid_flags) if not v), None)
            if first_false is not None:
                assert all(not v for v in valid_flags[first_false:]), (
                    f"Invalid rows not contiguous at end for {port}"
                )


class TestTemporalOrdering:
    def test_timestamps_monotonic_per_port(self, generated_df):
        for port in generated_df["port_id"].unique():
            port_ts = generated_df[generated_df["port_id"] == port][TIMESTAMP_COLUMN]
            assert port_ts.is_monotonic_increasing, (
                f"Timestamps not monotonic for {port}"
            )

    def test_hourly_frequency_per_port(self, generated_df):
        for port in generated_df["port_id"].unique():
            port_ts = generated_df[generated_df["port_id"] == port][TIMESTAMP_COLUMN].sort_values()
            diffs = port_ts.diff().dropna()
            assert (diffs == pd.Timedelta(hours=1)).all(), (
                f"Non-hourly intervals found for {port}"
            )


class TestTargets:
    def test_congestion_label_is_binary(self, generated_df):
        assert set(generated_df["congestion_label"].unique()).issubset({0, 1})

    def test_future_delay_is_numeric(self, generated_df):
        assert pd.api.types.is_float_dtype(generated_df["future_delay_hours"])

    def test_targets_not_in_feature_list(self):
        from src.config import ESSENTIAL_FEATURES, TARGET_COLUMNS
        for col in TARGET_COLUMNS:
            assert col not in ESSENTIAL_FEATURES, (
                f"Target column '{col}' found in ESSENTIAL_FEATURES — data leakage risk"
            )

    def test_metadata_not_in_feature_list(self):
        from src.config import ESSENTIAL_FEATURES, METADATA_COLUMNS
        for col in METADATA_COLUMNS:
            assert col not in ESSENTIAL_FEATURES, (
                f"Metadata column '{col}' found in ESSENTIAL_FEATURES"
            )

    def test_identifier_not_in_feature_list(self):
        from src.config import ESSENTIAL_FEATURES, IDENTIFIER_COLUMNS
        for col in IDENTIFIER_COLUMNS:
            assert col not in ESSENTIAL_FEATURES, (
                f"Identifier column '{col}' found in ESSENTIAL_FEATURES"
            )
