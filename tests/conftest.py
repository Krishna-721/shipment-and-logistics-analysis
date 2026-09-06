"""
Shared pytest fixtures for the Safiri PortPulse test suite.

Fixtures are scoped at session level where possible so the dataset and
trained models are built once per test run rather than once per test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import (
    ESSENTIAL_FEATURES,
    QUEUE_THRESHOLD,
    RANDOM_STATE,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
    VALID_ROW_COLUMN,
)


# ─────────────────────────────────────────────────────────────────────────────
# Minimal synthetic DataFrame — no file I/O required
# ─────────────────────────────────────────────────────────────────────────────

N_PORTS = 4
HOURS_PER_PORT = 30   # tiny; enough for all unit tests


def _make_synthetic_df(seed: int = 42) -> pd.DataFrame:
    """
    Build a small but structurally complete synthetic DataFrame that mirrors
    the columns produced by the real generator without running the generator.
    """
    rng = np.random.default_rng(seed)
    n = N_PORTS * HOURS_PER_PORT
    timestamps = pd.date_range("2026-01-01", periods=HOURS_PER_PORT, freq="h")

    rows = []
    for port_idx in range(N_PORTS):
        port_id = f"PORT_{port_idx + 1:02d}"
        total_berths = rng.integers(4, 12)
        for ts in timestamps:
            queue = float(rng.uniform(0, total_berths * 2))
            berth_util = min(1.0, queue / total_berths)
            available = max(0, total_berths - int(round(queue * 0.25)))
            rows.append({
                "port_id":                    port_id,
                "timestamp":                  ts,
                "total_berths":               int(total_berths),
                "available_berths":           int(available),
                "berth_utilization":          float(berth_util),
                "vessels_currently_in_port":  int(max(0, round(queue))),
                "vessels_anchored":           int(max(0, round(queue * 0.7))),
                "vessels_approaching":        int(rng.poisson(1.5)),
                "arrivals_last_1h":           int(rng.poisson(2.0)),
                "arrivals_last_6h":           int(rng.poisson(12.0)),
                "arrivals_last_24h":          int(rng.poisson(48.0)),
                "arrival_rate":               float(rng.uniform(0.5, 3.5)),
                "queue_length":               queue,
                "avg_waiting_time_hours":     float(max(0, rng.normal(2, 1))),
                "cranes_operational":         int(rng.integers(1, 8)),
                "crane_utilization":          float(rng.uniform(0.3, 0.9)),
                "equipment_failure_count":    int(rng.poisson(0.1)),
                "labor_availability_pct":     float(rng.uniform(0.7, 1.0)),
                "weather_severity":           float(rng.uniform(0, 0.5)),
                "storm_flag":                 int(rng.uniform(0, 1) > 0.9),
                "hour":                       int(ts.hour),
                "day_of_week":                int(ts.dayofweek),
                "is_weekend":                 int(ts.dayofweek >= 5),
                "historical_avg_wait_time":   float(rng.uniform(1, 4)),
                "historical_congestion_rate": float(rng.uniform(0.05, 0.40)),
                "traffic_pressure":           float(rng.uniform(0, 1)),
                "capacity_pressure":          float(berth_util),
                "queue_pressure":             float(queue / total_berths),
                "equipment_pressure":         float(rng.uniform(0, 0.2)),
                "weather_pressure":           float(rng.uniform(0, 0.5)),
                "queue_capacity_interaction": float(queue / total_berths * berth_util),
                "traffic_weather_interaction": float(rng.uniform(0, 0.5)),
                "vessel_type":                rng.choice(["container", "bulk", "tanker", "general_cargo"]),
                "cargo_type":                 rng.choice(["containerized", "dry_bulk", "liquid_bulk", "general"]),
                # Targets
                "congestion_label":           int(queue > QUEUE_THRESHOLD * total_berths),
                "future_delay_hours":         float(max(0, rng.normal(3, 1))),
                # Metadata
                "is_valid_training_row":      True,
            })

    df = pd.DataFrame(rows)
    # Last 6 rows per port have incomplete horizon
    for port_id in df["port_id"].unique():
        mask = df["port_id"] == port_id
        idx = df[mask].index[-6:]
        df.loc[idx, "is_valid_training_row"] = False

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    """Full synthetic DataFrame (session-scoped, built once)."""
    return _make_synthetic_df()


@pytest.fixture(scope="session")
def split_dfs(raw_df):
    """Return (train_df, test_df) using the real temporal split function."""
    from src.data.preprocessing import create_temporal_split
    # Override the config dates to match the small fixture (24 hours span)
    # by splitting at the midpoint timestamp instead.
    valid = raw_df[raw_df[VALID_ROW_COLUMN]].copy()
    mid_ts = valid[TIMESTAMP_COLUMN].sort_values().iloc[len(valid) // 2]
    train = valid[valid[TIMESTAMP_COLUMN] <= mid_ts].copy()
    test  = valid[valid[TIMESTAMP_COLUMN] >  mid_ts].copy()
    return train, test


@pytest.fixture(scope="session")
def prepared_data(split_dfs):
    """Return (X_train, X_test, y_train, y_test, feature_names)."""
    from src.data.preprocessing import prepare_features
    train_df, test_df = split_dfs
    return prepare_features(train_df, test_df)


@pytest.fixture(scope="session")
def scaled_data(prepared_data):
    """Return scaled arrays and the fitted scaler."""
    from src.data.preprocessing import apply_scaler, fit_scaler
    X_train, X_test, y_train, y_test, feature_names = prepared_data
    scaler = fit_scaler(X_train)
    X_tr_s, X_te_s = apply_scaler(scaler, X_train, X_test)
    return X_tr_s, X_te_s, y_train, y_test, scaler, feature_names


@pytest.fixture(scope="session")
def trained_lr(scaled_data):
    """Trained Logistic Regression model."""
    from src.models.congestion import CongestionClassifier
    X_tr, _, y_tr, *_ = scaled_data
    clf = CongestionClassifier("logistic")
    clf.fit(X_tr, y_tr.values)
    return clf


@pytest.fixture(scope="session")
def trained_rf(prepared_data):
    """Trained Random Forest model."""
    from src.models.congestion import CongestionClassifier
    X_tr, _, y_tr, *_ = prepared_data
    clf = CongestionClassifier("random_forest", n_estimators=10)
    clf.fit(X_tr.values, y_tr.values)
    return clf


@pytest.fixture(scope="session")
def queue_baseline(prepared_data):
    """Fitted QueueOnlyBaseline."""
    from src.models.baseline import QueueOnlyBaseline
    X_tr, _, y_tr, *_ = prepared_data
    b = QueueOnlyBaseline()
    b.fit(X_tr, y_tr)
    return b
