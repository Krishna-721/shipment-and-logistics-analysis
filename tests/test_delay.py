"""
Tests for E-007 — Future Delay Prediction.

Covers:
- DelayBaseline: fit/predict_mean/predict_median, not-fitted guard
- DelayRegressor: fit/predict interface for all model types
- Predict output: non-negative (clipped), shape, dtype
- Negative-target guard on fit
- evaluate_delay_model: metric types, plausible ranges, perfect case
- format_delay_results_table: shape and columns
- No leakage: future_delay_hours and congestion_label absent from features
- Target separation: delay target extracted from DataFrame, not feature matrix
- Reproducibility: same seed → same predictions
- Model persistence: save/load round-trip preserves predictions
- Invalid model type raises ValueError
- Predict before fit raises RuntimeError
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import TARGET_COLUMNS, METADATA_COLUMNS, IDENTIFIER_COLUMNS


# ─────────────────────────────────────────────────────────────────────────────
# Small deterministic regression fixtures (no file I/O)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def reg_data():
    """
    Tiny regression dataset aligned with the conftest synthetic DataFrame.
    Uses the session-level prepared_data fixture indirectly via conftest,
    but builds its own y from the fixture's delay column.
    """
    rng = np.random.default_rng(42)
    n_train, n_test, n_feat = 80, 30, 10

    X_train = rng.uniform(0, 5, (n_train, n_feat)).astype(np.float32)
    X_test  = rng.uniform(0, 5, (n_test,  n_feat)).astype(np.float32)

    # Target: linear combination of first 3 features + noise, clipped to ≥ 0
    w = np.array([1.5, -0.8, 2.0] + [0.0] * (n_feat - 3))
    y_train = np.clip(X_train @ w + rng.normal(0, 0.2, n_train), 0, None)
    y_test  = np.clip(X_test  @ w + rng.normal(0, 0.2, n_test),  0, None)

    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────────────────────
# DelayBaseline
# ─────────────────────────────────────────────────────────────────────────────

class TestDelayBaseline:
    def test_fit_stores_mean_and_median(self, reg_data):
        from src.models.delay import DelayBaseline
        _, _, y_tr, _ = reg_data
        b = DelayBaseline()
        b.fit(y_tr)
        assert b.train_mean   == pytest.approx(float(y_tr.mean()),       rel=1e-6)
        assert b.train_median == pytest.approx(float(np.median(y_tr)),   rel=1e-6)

    def test_predict_mean_shape_and_value(self, reg_data):
        from src.models.delay import DelayBaseline
        _, X_te, y_tr, _ = reg_data
        b = DelayBaseline()
        b.fit(y_tr)
        preds = b.predict_mean(len(X_te))
        assert preds.shape == (len(X_te),)
        assert (preds == b.train_mean).all()

    def test_predict_median_shape_and_value(self, reg_data):
        from src.models.delay import DelayBaseline
        _, X_te, y_tr, _ = reg_data
        b = DelayBaseline()
        b.fit(y_tr)
        preds = b.predict_median(len(X_te))
        assert preds.shape == (len(X_te),)
        assert (preds == b.train_median).all()

    def test_predict_before_fit_raises(self):
        from src.models.delay import DelayBaseline
        b = DelayBaseline()
        with pytest.raises(RuntimeError, match="fitted"):
            b.predict_mean(5)

    def test_median_predict_before_fit_raises(self):
        from src.models.delay import DelayBaseline
        b = DelayBaseline()
        with pytest.raises(RuntimeError, match="fitted"):
            b.predict_median(5)

    def test_fitted_flag(self, reg_data):
        from src.models.delay import DelayBaseline
        _, _, y_tr, _ = reg_data
        b = DelayBaseline()
        assert not b.is_fitted
        b.fit(y_tr)
        assert b.is_fitted

    def test_repr_changes_after_fit(self, reg_data):
        from src.models.delay import DelayBaseline
        _, _, y_tr, _ = reg_data
        b = DelayBaseline()
        assert "unfitted" in repr(b)
        b.fit(y_tr)
        assert "mean=" in repr(b)


# ─────────────────────────────────────────────────────────────────────────────
# DelayRegressor — interface
# ─────────────────────────────────────────────────────────────────────────────

class TestDelayRegressorInterface:
    @pytest.mark.parametrize("model_type", ["ridge", "random_forest", "xgboost"])
    def test_fit_predict_pipeline(self, model_type, reg_data):
        from src.models.delay import DelayRegressor
        X_tr, X_te, y_tr, _ = reg_data
        kwargs = {"n_estimators": 5} if model_type != "ridge" else {}
        m = DelayRegressor(model_type, **kwargs)
        m.fit(X_tr, y_tr)
        preds = m.predict(X_te)
        assert preds.shape == (len(X_te),)
        assert preds.dtype in (np.float32, np.float64)

    @pytest.mark.parametrize("model_type", ["ridge", "random_forest", "xgboost"])
    def test_predictions_non_negative(self, model_type, reg_data):
        from src.models.delay import DelayRegressor
        X_tr, X_te, y_tr, _ = reg_data
        kwargs = {"n_estimators": 5} if model_type != "ridge" else {}
        m = DelayRegressor(model_type, **kwargs)
        m.fit(X_tr, y_tr)
        preds = m.predict(X_te)
        assert (preds >= 0).all(), f"Negative predictions from {model_type}"

    def test_invalid_model_type_raises(self):
        from src.models.delay import DelayRegressor
        with pytest.raises(ValueError, match="model_type"):
            DelayRegressor("neural_net")

    def test_predict_before_fit_raises(self):
        from src.models.delay import DelayRegressor
        m = DelayRegressor("ridge")
        with pytest.raises(RuntimeError, match="fitted"):
            m.predict(np.array([[1.0, 2.0]]))

    def test_fit_rejects_negative_target(self, reg_data):
        from src.models.delay import DelayRegressor
        X_tr, _, _, _ = reg_data
        y_bad = np.full(len(X_tr), -1.0)
        m = DelayRegressor("ridge")
        with pytest.raises(ValueError, match="negative"):
            m.fit(X_tr, y_bad)

    def test_fitted_flag(self, reg_data):
        from src.models.delay import DelayRegressor
        X_tr, _, y_tr, _ = reg_data
        m = DelayRegressor("ridge")
        assert not m.is_fitted
        m.fit(X_tr, y_tr)
        assert m.is_fitted


# ─────────────────────────────────────────────────────────────────────────────
# Clipping: model should never return negative predictions
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictionClipping:
    def test_ridge_clips_to_zero(self):
        """
        Train Ridge on targets near zero so it predicts slightly negative on
        unseen data; the wrapper must clip those to 0.
        """
        from src.models.delay import DelayRegressor
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 3)).astype(np.float64)
        y = np.clip(X[:, 0] * 0.01, 0, None)   # near-zero targets
        m = DelayRegressor("ridge")
        m.fit(X, y)
        # Force a test point that raw Ridge would predict as negative
        X_neg = np.array([[-10.0, 0.0, 0.0]])
        pred = m.predict(X_neg)
        assert pred[0] >= 0.0

    def test_random_forest_output_non_negative(self, reg_data):
        from src.models.delay import DelayRegressor
        X_tr, X_te, y_tr, _ = reg_data
        m = DelayRegressor("random_forest", n_estimators=5)
        m.fit(X_tr, y_tr)
        assert (m.predict(X_te) >= 0).all()


# ─────────────────────────────────────────────────────────────────────────────
# Feature importance
# ─────────────────────────────────────────────────────────────────────────────

class TestDelayFeatureImportance:
    def test_ridge_importance_shape(self, reg_data):
        from src.models.delay import DelayRegressor
        X_tr, _, y_tr, _ = reg_data
        m = DelayRegressor("ridge")
        m.fit(X_tr, y_tr)
        imp = m.get_feature_importance()
        assert imp is not None
        assert len(imp) == X_tr.shape[1]

    def test_ridge_importance_non_negative(self, reg_data):
        from src.models.delay import DelayRegressor
        X_tr, _, y_tr, _ = reg_data
        m = DelayRegressor("ridge")
        m.fit(X_tr, y_tr)
        imp = m.get_feature_importance()
        assert (imp >= 0).all()

    def test_random_forest_importance_sums_to_one(self, reg_data):
        from src.models.delay import DelayRegressor
        X_tr, _, y_tr, _ = reg_data
        m = DelayRegressor("random_forest", n_estimators=10)
        m.fit(X_tr, y_tr)
        imp = m.get_feature_importance()
        assert imp is not None
        assert abs(imp.sum() - 1.0) < 1e-6

    def test_unfitted_importance_returns_none(self):
        from src.models.delay import DelayRegressor
        m = DelayRegressor("ridge")
        assert m.get_feature_importance() is None


# ─────────────────────────────────────────────────────────────────────────────
# Regression metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluateDelayModel:
    def test_perfect_prediction_metrics(self):
        from src.models.delay import evaluate_delay_model
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r = evaluate_delay_model(y, y, "Perfect")
        assert r["mae"]  == pytest.approx(0.0, abs=1e-9)
        assert r["rmse"] == pytest.approx(0.0, abs=1e-9)
        assert r["r2"]   == pytest.approx(1.0, abs=1e-9)

    def test_constant_prediction_r2_non_positive(self):
        from src.models.delay import evaluate_delay_model
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.full(5, y_true.mean())
        r = evaluate_delay_model(y_true, y_pred, "Constant")
        assert r["r2"] <= 0.0

    def test_result_has_required_keys(self):
        from src.models.delay import evaluate_delay_model
        y = np.array([0.0, 1.0, 2.0])
        r = evaluate_delay_model(y, y * 1.1, "Test")
        for key in ["model_name", "n_samples", "target_mean", "target_std",
                    "target_min", "target_max", "mae", "rmse", "r2"]:
            assert key in r, f"Missing key: {key}"

    def test_mae_non_negative(self):
        from src.models.delay import evaluate_delay_model
        rng = np.random.default_rng(7)
        y = rng.uniform(0, 10, 50)
        p = rng.uniform(0, 10, 50)
        r = evaluate_delay_model(y, p)
        assert r["mae"]  >= 0
        assert r["rmse"] >= 0

    def test_n_samples_correct(self):
        from src.models.delay import evaluate_delay_model
        y = np.arange(20, dtype=float)
        r = evaluate_delay_model(y, y)
        assert r["n_samples"] == 20

    def test_target_stats_correct(self):
        from src.models.delay import evaluate_delay_model
        y = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
        r = evaluate_delay_model(y, y)
        assert r["target_mean"] == pytest.approx(4.0, rel=1e-6)
        assert r["target_min"]  == pytest.approx(0.0, abs=1e-9)
        assert r["target_max"]  == pytest.approx(8.0, rel=1e-6)

    def test_mae_matches_manual_calculation(self):
        from src.models.delay import evaluate_delay_model
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 2.0])
        r = evaluate_delay_model(y_true, y_pred)
        expected_mae = (abs(1-2) + abs(2-2) + abs(3-2)) / 3
        assert r["mae"] == pytest.approx(expected_mae, rel=1e-9)

    def test_rmse_matches_manual_calculation(self):
        from src.models.delay import evaluate_delay_model
        y_true = np.array([0.0, 0.0, 3.0, 4.0])
        y_pred = np.array([0.0, 0.0, 0.0, 0.0])
        r = evaluate_delay_model(y_true, y_pred)
        expected_rmse = np.sqrt((9 + 16) / 4)
        assert r["rmse"] == pytest.approx(expected_rmse, rel=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# format_delay_results_table
# ─────────────────────────────────────────────────────────────────────────────

class TestFormatDelayResultsTable:
    def test_shape(self):
        from src.models.delay import evaluate_delay_model, format_delay_results_table
        y = np.arange(10, dtype=float)
        results = [
            evaluate_delay_model(y, y * 1.0, "M1"),
            evaluate_delay_model(y, y * 1.1, "M2"),
            evaluate_delay_model(y, y * 1.5, "M3"),
        ]
        df = format_delay_results_table(results)
        assert len(df) == 3
        assert list(df.columns) == ["Model", "MAE", "RMSE", "R2"]

    def test_model_names_preserved(self):
        from src.models.delay import evaluate_delay_model, format_delay_results_table
        y = np.array([1.0, 2.0, 3.0])
        results = [evaluate_delay_model(y, y, name) for name in ["A", "B"]]
        df = format_delay_results_table(results)
        assert list(df["Model"]) == ["A", "B"]


# ─────────────────────────────────────────────────────────────────────────────
# Leakage prevention
# ─────────────────────────────────────────────────────────────────────────────

class TestLeakagePrevention:
    """
    Confirm that target and metadata columns are absent from the
    feature matrix used for delay model training.
    """

    def test_future_delay_not_in_feature_list(self):
        from src.config import ESSENTIAL_FEATURES, TARGET_COLUMNS
        assert "future_delay_hours" not in ESSENTIAL_FEATURES

    def test_congestion_label_not_in_feature_list(self):
        from src.config import ESSENTIAL_FEATURES
        assert "congestion_label" not in ESSENTIAL_FEATURES

    def test_metadata_not_in_feature_list(self):
        from src.config import ESSENTIAL_FEATURES, METADATA_COLUMNS
        for col in METADATA_COLUMNS:
            assert col not in ESSENTIAL_FEATURES

    def test_identifiers_not_in_feature_list(self):
        from src.config import ESSENTIAL_FEATURES, IDENTIFIER_COLUMNS
        for col in IDENTIFIER_COLUMNS:
            assert col not in ESSENTIAL_FEATURES

    def test_prepared_features_exclude_delay_target(self, prepared_data):
        """
        The prepared X_train DataFrame must not contain future_delay_hours.
        """
        X_train, X_test, _, _, feature_names = prepared_data
        assert "future_delay_hours" not in feature_names
        assert "future_delay_hours" not in X_train.columns
        assert "future_delay_hours" not in X_test.columns

    def test_prepared_features_exclude_congestion_label(self, prepared_data):
        X_train, X_test, _, _, feature_names = prepared_data
        assert "congestion_label" not in feature_names
        assert "congestion_label" not in X_train.columns

    def test_y_train_is_congestion_label_not_delay(self, prepared_data):
        """
        The y_train returned by prepare_features must be congestion_label
        (binary), not future_delay_hours (continuous).
        This is intentional — delay experiments extract y from the raw
        DataFrame separately.
        """
        _, _, y_train, y_test, _ = prepared_data
        assert set(y_train.unique()).issubset({0, 1}), (
            "y_train contains non-binary values — delay target may have leaked"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

class TestReproducibility:
    @pytest.mark.parametrize("model_type", ["ridge", "random_forest"])
    def test_same_predictions_across_runs(self, model_type, reg_data):
        from src.models.delay import DelayRegressor
        X_tr, X_te, y_tr, _ = reg_data
        kwargs = {"n_estimators": 10} if model_type != "ridge" else {}

        m1 = DelayRegressor(model_type, **kwargs)
        m1.fit(X_tr, y_tr)

        m2 = DelayRegressor(model_type, **kwargs)
        m2.fit(X_tr, y_tr)

        # Use allclose (atol=1e-12) rather than array_equal — floating-point
        # parallel aggregation in RF may differ at machine-epsilon precision.
        np.testing.assert_allclose(
            m1.predict(X_te), m2.predict(X_te), atol=1e-12,
            err_msg=f"Predictions not reproducible for {model_type}",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

class TestDelayModelPersistence:
    @pytest.mark.parametrize("model_type", ["ridge", "random_forest"])
    def test_save_load_preserves_predictions(self, model_type, reg_data):
        from src.models.delay import DelayRegressor
        X_tr, X_te, y_tr, _ = reg_data
        kwargs = {"n_estimators": 10} if model_type != "ridge" else {}
        m = DelayRegressor(model_type, **kwargs)
        m.fit(X_tr, y_tr)
        preds_before = m.predict(X_te)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "delay_model.joblib"
            m.save(path)
            loaded = DelayRegressor.load(path)

        preds_after = loaded.predict(X_te)
        # allclose covers any floating-point aggregation differences in RF
        np.testing.assert_allclose(preds_before, preds_after, atol=1e-12)

    def test_save_unfitted_raises(self):
        from src.models.delay import DelayRegressor
        m = DelayRegressor("ridge")
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(RuntimeError, match="fitted"):
                m.save(Path(tmpdir) / "model.joblib")

    def test_loaded_model_is_fitted(self, reg_data):
        from src.models.delay import DelayRegressor
        X_tr, _, y_tr, _ = reg_data
        m = DelayRegressor("ridge")
        m.fit(X_tr, y_tr)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.joblib"
            m.save(path)
            loaded = DelayRegressor.load(path)
        assert loaded.is_fitted
        assert loaded.model_type == "ridge"


# ─────────────────────────────────────────────────────────────────────────────
# Delay target extraction from fixture data
# ─────────────────────────────────────────────────────────────────────────────

class TestDelayTargetExtraction:
    """
    Verify that future_delay_hours can be extracted from the conftest
    synthetic DataFrame correctly — non-negative, numeric, right shape.
    """

    def test_delay_column_exists_in_raw_df(self, raw_df):
        assert "future_delay_hours" in raw_df.columns

    def test_delay_values_non_negative(self, raw_df):
        assert (raw_df["future_delay_hours"] >= 0).all()

    def test_delay_values_numeric(self, raw_df):
        assert pd.api.types.is_float_dtype(raw_df["future_delay_hours"])

    def test_delay_extraction_from_split(self, split_dfs):
        train_df, test_df = split_dfs
        y_tr = train_df["future_delay_hours"].values
        y_te = test_df["future_delay_hours"].values
        assert len(y_tr) > 0
        assert len(y_te) > 0
        assert (y_tr >= 0).all()
        assert (y_te >= 0).all()

    def test_delay_and_feature_lengths_match(self, split_dfs, prepared_data):
        train_df, test_df = split_dfs
        X_train, X_test, _, _, _ = prepared_data
        assert len(train_df["future_delay_hours"]) == len(X_train)
        assert len(test_df["future_delay_hours"])  == len(X_test)
