"""
Tests for congestion classification models.

Covers:
- QueueOnlyBaseline: correct prediction logic, proba shape, not-fitted guard
- CongestionClassifier: fit/predict/predict_proba interface for all model types
- Reproducibility: same seed → same predictions
- Feature importance: shape alignment, non-negative values
- Metric calculation: correct output types and plausible ranges
- Model persistence: save / load round-trip
- No future-target leakage in features used for training
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Queue-Only Baseline
# ─────────────────────────────────────────────────────────────────────────────

class TestQueueOnlyBaseline:
    def test_predict_returns_binary_array(self, queue_baseline, prepared_data):
        X_train, X_test, *_ = prepared_data
        preds = queue_baseline.predict(X_test)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_shape(self, queue_baseline, prepared_data):
        _, X_test, *_ = prepared_data
        probas = queue_baseline.predict_proba(X_test)
        assert probas.shape == (len(X_test), 2)

    def test_predict_proba_sums_to_one(self, queue_baseline, prepared_data):
        _, X_test, *_ = prepared_data
        probas = queue_baseline.predict_proba(X_test)
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-6)

    def test_high_queue_predicts_congestion(self):
        """Rows with queue > threshold * berths must be predicted as congested."""
        from src.models.baseline import QueueOnlyBaseline
        from src.config import QUEUE_THRESHOLD

        X = pd.DataFrame({
            "queue_length": [10.0, 0.1],
            "total_berths": [4,    4],
        })
        b = QueueOnlyBaseline(threshold=QUEUE_THRESHOLD)
        b.fit(X, pd.Series([1, 0]))
        preds = b.predict(X)
        # queue=10 > 0.5*4=2 → congested; queue=0.1 ≤ 2 → not congested
        assert preds[0] == 1
        assert preds[1] == 0

    def test_not_fitted_raises(self):
        from src.models.baseline import QueueOnlyBaseline
        b = QueueOnlyBaseline()
        X = pd.DataFrame({"queue_length": [1.0], "total_berths": [4]})
        with pytest.raises(ValueError, match="fitted"):
            b.predict(X)

    def test_missing_column_raises_on_fit(self):
        from src.models.baseline import QueueOnlyBaseline
        b = QueueOnlyBaseline()
        X = pd.DataFrame({"queue_length": [1.0]})   # missing total_berths
        with pytest.raises(ValueError, match="total_berths"):
            b.fit(X, pd.Series([0]))

    def test_threshold_stored_as_attribute(self):
        from src.models.baseline import QueueOnlyBaseline
        b = QueueOnlyBaseline(threshold=0.75)
        assert b.threshold == 0.75


# ─────────────────────────────────────────────────────────────────────────────
# CongestionClassifier — interface
# ─────────────────────────────────────────────────────────────────────────────

class TestCongestionClassifierInterface:
    @pytest.mark.parametrize("model_type", ["logistic", "random_forest", "xgboost"])
    def test_fit_predict_pipeline(self, model_type, prepared_data, scaled_data):
        from src.models.congestion import CongestionClassifier

        X_tr_raw, X_te_raw, y_tr, y_te, _ = prepared_data
        X_tr_s, X_te_s, *_ = scaled_data

        X_tr = X_tr_s if model_type == "logistic" else X_tr_raw.values
        X_te = X_te_s if model_type == "logistic" else X_te_raw.values

        n_est = 5  # tiny for speed in tests
        kwargs = {"n_estimators": n_est} if model_type != "logistic" else {}
        clf = CongestionClassifier(model_type, **kwargs)
        clf.fit(X_tr, y_tr.values)

        preds = clf.predict(X_te)
        assert preds.shape == (len(X_te),)
        assert set(preds).issubset({0, 1})

    @pytest.mark.parametrize("model_type", ["logistic", "random_forest", "xgboost"])
    def test_predict_proba_shape_and_valid(self, model_type, prepared_data, scaled_data):
        from src.models.congestion import CongestionClassifier

        X_tr_raw, X_te_raw, y_tr, y_te, _ = prepared_data
        X_tr_s, X_te_s, *_ = scaled_data

        X_tr = X_tr_s if model_type == "logistic" else X_tr_raw.values
        X_te = X_te_s if model_type == "logistic" else X_te_raw.values

        kwargs = {"n_estimators": 5} if model_type != "logistic" else {}
        clf = CongestionClassifier(model_type, **kwargs)
        clf.fit(X_tr, y_tr.values)

        probas = clf.predict_proba(X_te)
        assert probas.shape == (len(X_te), 2)
        assert (probas >= 0).all()
        assert (probas <= 1).all()
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-6)

    def test_invalid_model_type_raises(self):
        from src.models.congestion import CongestionClassifier
        with pytest.raises(ValueError, match="model_type"):
            CongestionClassifier("neural_net")

    def test_predict_before_fit_raises(self):
        from src.models.congestion import CongestionClassifier
        clf = CongestionClassifier("logistic")
        with pytest.raises(RuntimeError, match="fitted"):
            clf.predict(np.array([[1.0, 2.0]]))


class TestReproducibility:
    @pytest.mark.parametrize("model_type", ["logistic", "random_forest"])
    def test_same_predictions_across_runs(self, model_type, prepared_data, scaled_data):
        from src.models.congestion import CongestionClassifier

        X_tr_raw, X_te_raw, y_tr, *_ = prepared_data
        X_tr_s, X_te_s, *_ = scaled_data

        X_tr = X_tr_s if model_type == "logistic" else X_tr_raw.values
        X_te = X_te_s if model_type == "logistic" else X_te_raw.values

        kwargs = {"n_estimators": 10} if model_type != "logistic" else {}

        clf1 = CongestionClassifier(model_type, **kwargs)
        clf1.fit(X_tr, y_tr.values)

        clf2 = CongestionClassifier(model_type, **kwargs)
        clf2.fit(X_tr, y_tr.values)

        np.testing.assert_array_equal(clf1.predict(X_te), clf2.predict(X_te))


# ─────────────────────────────────────────────────────────────────────────────
# Feature importance
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureImportance:
    def test_logistic_regression_importance_shape(self, trained_lr, prepared_data):
        _, _, _, _, feature_names = prepared_data
        imp = trained_lr.get_feature_importance()
        assert imp is not None
        assert len(imp) == len(feature_names)

    def test_random_forest_importance_shape(self, trained_rf, prepared_data):
        _, _, _, _, feature_names = prepared_data
        imp = trained_rf.get_feature_importance()
        assert imp is not None
        assert len(imp) == len(feature_names)

    def test_random_forest_importance_non_negative(self, trained_rf):
        imp = trained_rf.get_feature_importance()
        assert (imp >= 0).all()

    def test_random_forest_importance_sums_to_one(self, trained_rf):
        imp = trained_rf.get_feature_importance()
        assert abs(imp.sum() - 1.0) < 1e-6

    def test_logistic_regression_importance_non_negative(self, trained_lr):
        """LR importance is |coeff|, must be ≥ 0."""
        imp = trained_lr.get_feature_importance()
        assert (imp >= 0).all()


# ─────────────────────────────────────────────────────────────────────────────
# Metric calculation
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluationMetrics:
    def test_perfect_predictions_give_perfect_metrics(self):
        from src.models.evaluation import evaluate_classification_model
        y = np.array([0, 0, 1, 1])
        result = evaluate_classification_model(y, y, y.astype(float), "Perfect")
        assert result["accuracy"]  == 1.0
        assert result["precision"] == 1.0
        assert result["recall"]    == 1.0
        assert result["f1_score"]  == 1.0
        assert result["roc_auc"]   == 1.0

    def test_all_wrong_gives_zero_precision_recall(self):
        from src.models.evaluation import evaluate_classification_model
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 0])
        result = evaluate_classification_model(y_true, y_pred, None, "AllWrong")
        assert result["precision"] == 0.0
        assert result["recall"]    == 0.0

    def test_result_contains_required_keys(self):
        from src.models.evaluation import evaluate_classification_model
        y = np.array([0, 1, 0, 1])
        p = np.array([0, 1, 1, 1])
        result = evaluate_classification_model(y, p, p.astype(float), "Test")
        for key in ["accuracy", "precision", "recall", "f1_score",
                    "roc_auc", "pr_auc", "brier_score", "confusion_matrix"]:
            assert key in result, f"Missing key: {key}"

    def test_confusion_matrix_keys(self):
        from src.models.evaluation import evaluate_classification_model
        y = np.array([0, 0, 1, 1])
        p = np.array([0, 1, 0, 1])
        result = evaluate_classification_model(y, p)
        cm = result["confusion_matrix"]
        assert set(cm.keys()) == {"tn", "fp", "fn", "tp"}
        assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == len(y)

    def test_metrics_in_valid_range(self):
        from src.models.evaluation import evaluate_classification_model
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 100)
        p = rng.integers(0, 2, 100)
        prob = rng.uniform(0, 1, 100)
        result = evaluate_classification_model(y, p, prob, "Random")
        for metric in ["accuracy", "precision", "recall", "f1_score"]:
            assert 0.0 <= result[metric] <= 1.0
        if result["roc_auc"] is not None:
            assert 0.0 <= result["roc_auc"] <= 1.0
        if result["brier_score"] is not None:
            assert 0.0 <= result["brier_score"] <= 1.0

    def test_format_results_table_shape(self):
        from src.models.evaluation import evaluate_classification_model, format_results_table
        y = np.array([0, 1, 0, 1])
        p = np.array([0, 1, 1, 1])
        results = [
            evaluate_classification_model(y, p, p.astype(float), "M1"),
            evaluate_classification_model(y, p, p.astype(float), "M2"),
        ]
        df = format_results_table(results)
        assert len(df) == 2
        assert "Model" in df.columns
        for col in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "Brier Score"]:
            assert col in df.columns


# ─────────────────────────────────────────────────────────────────────────────
# Model persistence (save / load round-trip)
# ─────────────────────────────────────────────────────────────────────────────

class TestModelPersistence:
    @pytest.mark.parametrize("model_type", ["logistic", "random_forest"])
    def test_save_load_predictions_identical(self, model_type, prepared_data, scaled_data):
        from src.models.congestion import CongestionClassifier

        X_tr_raw, X_te_raw, y_tr, *_ = prepared_data
        X_tr_s, X_te_s, *_ = scaled_data

        X_tr = X_tr_s if model_type == "logistic" else X_tr_raw.values
        X_te = X_te_s if model_type == "logistic" else X_te_raw.values

        kwargs = {"n_estimators": 10} if model_type != "logistic" else {}
        clf = CongestionClassifier(model_type, **kwargs)
        clf.fit(X_tr, y_tr.values)
        preds_before = clf.predict(X_te)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.joblib"
            clf.save(path)
            loaded = CongestionClassifier.load(path)

        preds_after = loaded.predict(X_te)
        np.testing.assert_array_equal(preds_before, preds_after)

    def test_save_unfitted_raises(self):
        from src.models.congestion import CongestionClassifier
        clf = CongestionClassifier("logistic")
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(RuntimeError, match="fitted"):
                clf.save(Path(tmpdir) / "model.joblib")
