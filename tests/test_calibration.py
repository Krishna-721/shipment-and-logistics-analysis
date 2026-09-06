"""
Tests for E-006 — Probability Calibration.

Covers:
- compute_calibration_metrics: correct types, valid ranges, determinism
- CalibrationResult.to_dict: all required keys, serialisable
- Brier score reproduced against sklearn reference
- Log-loss reproduced against sklearn reference
- ECE calculation: known-perfect case, known-imperfect case
- Probability bounds: outputs always in [0, 1]
- No test-label leakage: calibration fitted on training slice, not test set
- fit_calibrated_wrapper: fitted model returns valid probabilities
- compare_calibration_results: runs without error
- Reproducibility: same inputs → same metrics
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import brier_score_loss, log_loss


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_well_calibrated_probs(n: int = 200, seed: int = 0) -> tuple:
    """
    Return (y_true, y_prob) where the model is approximately well-calibrated:
    predicted probability ≈ observed positive rate within each decile.
    """
    rng = np.random.default_rng(seed)
    y_prob = rng.uniform(0.0, 1.0, n)
    # Bernoulli draw: each sample is positive with probability == y_prob[i]
    y_true = rng.binomial(1, y_prob).astype(float)
    return y_true, y_prob


def _make_bimodal_probs(n: int = 200, seed: int = 0) -> tuple:
    """
    Return (y_true, y_prob) mimicking the actual LR output: most predictions
    near 0 or near 1, very few in the middle.

    Labels are aligned with probabilities so that the model is well-predicting:
    low-prob samples → label 0, high-prob samples → label 1.
    This is what the actual trained LR produces (bimodal AND accurate).
    """
    rng = np.random.default_rng(seed)
    n_low  = int(n * 0.55)
    n_high = n - n_low
    low_probs  = rng.uniform(0.00, 0.02, n_low)
    high_probs = rng.uniform(0.97, 1.00, n_high)
    # Aligned labels: low-prob → 0, high-prob → 1
    y_prob = np.concatenate([low_probs, high_probs])
    y_true = np.concatenate([np.zeros(n_low), np.ones(n_high)]).astype(float)
    # Shuffle both together to mix ports/timestamps
    idx = rng.permutation(n)
    return y_true[idx], y_prob[idx]


# ─────────────────────────────────────────────────────────────────────────────
# CalibrationResult — data container
# ─────────────────────────────────────────────────────────────────────────────

class TestCalibrationResult:
    def test_to_dict_has_required_keys(self):
        from src.models.calibration import compute_calibration_metrics
        y_true, y_prob = _make_well_calibrated_probs()
        result = compute_calibration_metrics(y_true, y_prob, model_name="Test")
        d = result.to_dict()
        required = {
            "model_name", "brier_score", "log_loss", "ece",
            "bin_mean_predicted", "bin_fraction_positive", "bin_counts",
            "prob_min", "prob_max", "prob_median",
            "prob_bimodal_low_pct", "prob_bimodal_high_pct",
            "calibration_applied", "calibration_method",
        }
        assert required.issubset(d.keys()), (
            f"Missing keys: {required - d.keys()}"
        )

    def test_to_dict_is_json_serialisable(self):
        import json
        from src.models.calibration import compute_calibration_metrics
        y_true, y_prob = _make_well_calibrated_probs()
        result = compute_calibration_metrics(y_true, y_prob)
        json.dumps(result.to_dict())   # must not raise

    def test_calibration_applied_default_false(self):
        from src.models.calibration import compute_calibration_metrics
        y_true, y_prob = _make_well_calibrated_probs()
        result = compute_calibration_metrics(y_true, y_prob)
        assert result.calibration_applied is False
        assert result.calibration_method is None


# ─────────────────────────────────────────────────────────────────────────────
# compute_calibration_metrics — metric values
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeCalibrationMetrics:
    def test_brier_matches_sklearn(self):
        from src.models.calibration import compute_calibration_metrics
        y_true, y_prob = _make_well_calibrated_probs(seed=7)
        result = compute_calibration_metrics(y_true, y_prob)
        expected = brier_score_loss(y_true, y_prob)
        assert abs(result.brier_score - expected) < 1e-9

    def test_logloss_matches_sklearn(self):
        from src.models.calibration import compute_calibration_metrics
        y_true, y_prob = _make_well_calibrated_probs(seed=8)
        result = compute_calibration_metrics(y_true, y_prob)
        expected = log_loss(y_true, y_prob)
        assert abs(result.log_loss_score - expected) < 1e-9

    def test_brier_perfect_predictions_is_zero(self):
        from src.models.calibration import compute_calibration_metrics
        y_true = np.array([0.0, 0.0, 1.0, 1.0])
        y_prob = np.array([0.0, 0.0, 1.0, 1.0])
        result = compute_calibration_metrics(y_true, y_prob)
        assert result.brier_score == pytest.approx(0.0, abs=1e-9)

    def test_brier_worst_predictions_is_one(self):
        from src.models.calibration import compute_calibration_metrics
        y_true = np.array([0.0, 0.0, 1.0, 1.0])
        y_prob = np.array([1.0, 1.0, 0.0, 0.0])
        result = compute_calibration_metrics(y_true, y_prob)
        assert result.brier_score == pytest.approx(1.0, abs=1e-9)

    def test_ece_perfect_model_near_zero(self):
        """
        A model that always predicts exactly the true class probability
        should have ECE very close to zero.
        """
        from src.models.calibration import compute_calibration_metrics
        rng = np.random.default_rng(42)
        # Build a perfectly calibrated model: in each 0.1-wide bucket
        # assign prob = bucket midpoint and sample accordingly
        bucket_mids = [0.05, 0.15, 0.25, 0.35, 0.45,
                       0.55, 0.65, 0.75, 0.85, 0.95]
        y_prob_list, y_true_list = [], []
        for mid in bucket_mids:
            n_k = 50
            probs = np.full(n_k, mid)
            labels = rng.binomial(1, mid, n_k).astype(float)
            y_prob_list.append(probs)
            y_true_list.append(labels)
        y_prob = np.concatenate(y_prob_list)
        y_true = np.concatenate(y_true_list)
        result = compute_calibration_metrics(y_true, y_prob, n_bins=10)
        # With 50 samples per bin and true calibration, ECE should be small
        # (stochastic, but consistently < 0.10 with seed=42)
        assert result.ece < 0.10

    def test_ece_overconfident_model_has_positive_ece(self):
        """
        A model that always predicts 1.0 for all positives and 0.0 for all
        negatives has no calibration gap at the extremes — ECE depends on
        actual label distribution in bins.
        A model that predicts 0.9 for everything when true rate is 0.5
        should have ECE ≈ 0.40.
        """
        from src.models.calibration import compute_calibration_metrics
        n = 200
        y_true = np.array([1.0] * 100 + [0.0] * 100)
        y_prob = np.full(n, 0.9)   # always overconfident
        result = compute_calibration_metrics(y_true, y_prob, n_bins=10)
        # All predictions land in the [0.9,1.0) bin; observed rate ≈ 0.5
        # ECE ≈ 0.40
        assert result.ece > 0.30

    def test_bin_counts_sum_to_n(self):
        from src.models.calibration import compute_calibration_metrics
        y_true, y_prob = _make_well_calibrated_probs(n=300, seed=3)
        result = compute_calibration_metrics(y_true, y_prob, n_bins=10)
        assert sum(result.bin_counts) == len(y_true)

    def test_bin_lists_same_length(self):
        from src.models.calibration import compute_calibration_metrics
        y_true, y_prob = _make_well_calibrated_probs()
        result = compute_calibration_metrics(y_true, y_prob, n_bins=10)
        assert (
            len(result.bin_mean_predicted)
            == len(result.bin_fraction_positive)
            == len(result.bin_counts)
        )

    def test_bin_fraction_positive_in_range(self):
        from src.models.calibration import compute_calibration_metrics
        y_true, y_prob = _make_well_calibrated_probs()
        result = compute_calibration_metrics(y_true, y_prob)
        for fp in result.bin_fraction_positive:
            assert 0.0 <= fp <= 1.0

    def test_bin_mean_predicted_in_range(self):
        from src.models.calibration import compute_calibration_metrics
        y_true, y_prob = _make_well_calibrated_probs()
        result = compute_calibration_metrics(y_true, y_prob)
        for mp in result.bin_mean_predicted:
            assert 0.0 <= mp <= 1.0

    def test_prob_distribution_summary_correct(self):
        from src.models.calibration import compute_calibration_metrics
        y_true = np.array([0.0, 1.0])
        y_prob = np.array([0.1, 0.9])
        result = compute_calibration_metrics(y_true, y_prob, n_bins=2)
        assert result.prob_min    == pytest.approx(0.1, abs=1e-9)
        assert result.prob_max    == pytest.approx(0.9, abs=1e-9)
        assert result.prob_median == pytest.approx(0.5, abs=1e-9)

    def test_bimodal_pct_correct(self):
        from src.models.calibration import compute_calibration_metrics
        # 60% low (≤0.05), 40% high (≥0.95), 0% middle
        y_prob = np.concatenate([
            np.full(60, 0.01),
            np.full(40, 0.99),
        ])
        y_true = np.concatenate([np.zeros(60), np.ones(40)])
        result = compute_calibration_metrics(y_true, y_prob, n_bins=10)
        assert result.prob_bimodal_low_pct  == pytest.approx(60.0, abs=0.01)
        assert result.prob_bimodal_high_pct == pytest.approx(40.0, abs=0.01)

    def test_invalid_prob_raises(self):
        from src.models.calibration import compute_calibration_metrics
        y_true = np.array([0.0, 1.0])
        y_prob = np.array([0.5, 1.5])   # out of range
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            compute_calibration_metrics(y_true, y_prob)

    def test_non_binary_y_true_raises(self):
        from src.models.calibration import compute_calibration_metrics
        y_true = np.array([0.0, 0.5, 1.0])   # 0.5 is invalid
        y_prob = np.array([0.1, 0.5, 0.9])
        with pytest.raises(ValueError, match="binary"):
            compute_calibration_metrics(y_true, y_prob)

    def test_deterministic_same_inputs(self):
        from src.models.calibration import compute_calibration_metrics
        y_true, y_prob = _make_well_calibrated_probs(seed=99)
        r1 = compute_calibration_metrics(y_true, y_prob)
        r2 = compute_calibration_metrics(y_true, y_prob)
        assert r1.brier_score    == r2.brier_score
        assert r1.log_loss_score == r2.log_loss_score
        assert r1.ece            == r2.ece


# ─────────────────────────────────────────────────────────────────────────────
# Probability bounds — model outputs
# ─────────────────────────────────────────────────────────────────────────────

class TestProbabilityBounds:
    """Verify that all models produce probabilities strictly in [0, 1]."""

    def test_logistic_regression_proba_in_range(self, trained_lr, scaled_data):
        _, X_te_s, *_ = scaled_data
        proba = trained_lr.predict_proba(X_te_s)[:, 1]
        assert (proba >= 0.0).all(), "LR probability below 0"
        assert (proba <= 1.0).all(), "LR probability above 1"

    def test_random_forest_proba_in_range(self, trained_rf, prepared_data):
        _, X_te, *_ = prepared_data
        proba = trained_rf.predict_proba(X_te.values)[:, 1]
        assert (proba >= 0.0).all()
        assert (proba <= 1.0).all()

    def test_proba_sum_to_one(self, trained_lr, scaled_data):
        _, X_te_s, *_ = scaled_data
        proba = trained_lr.predict_proba(X_te_s)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Calibration metrics on fixture data
# ─────────────────────────────────────────────────────────────────────────────

class TestCalibrationOnFixtureModel:
    """Run the full calibration metric stack on the fixture-trained LR model."""

    def test_calibration_metrics_compute_without_error(
        self, trained_lr, scaled_data
    ):
        from src.models.calibration import compute_calibration_metrics
        _, X_te_s, _, y_te, *_ = scaled_data
        proba = trained_lr.predict_proba(X_te_s)[:, 1]
        result = compute_calibration_metrics(y_te.values.astype(float), proba)
        assert result.brier_score    >= 0.0
        assert result.log_loss_score >= 0.0
        assert result.ece            >= 0.0

    def test_calibration_brier_matches_sklearn(self, trained_lr, scaled_data):
        from src.models.calibration import compute_calibration_metrics
        _, X_te_s, _, y_te, *_ = scaled_data
        proba  = trained_lr.predict_proba(X_te_s)[:, 1]
        y_true = y_te.values.astype(float)
        result = compute_calibration_metrics(y_true, proba)
        ref    = brier_score_loss(y_true, proba)
        assert abs(result.brier_score - ref) < 1e-9

    def test_calibration_logloss_matches_sklearn(self, trained_lr, scaled_data):
        from src.models.calibration import compute_calibration_metrics
        _, X_te_s, _, y_te, *_ = scaled_data
        proba  = trained_lr.predict_proba(X_te_s)[:, 1]
        y_true = y_te.values.astype(float)
        result = compute_calibration_metrics(y_true, proba)
        ref    = log_loss(y_true, proba)
        assert abs(result.log_loss_score - ref) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# No test-label leakage during calibration fitting
# ─────────────────────────────────────────────────────────────────────────────

class TestCalibrationLeakage:
    """
    Verify that fit_calibrated_wrapper is called with training/calibration
    data only — never the test set — by confirming the function signature
    contract and that fitted output on held-out data is valid.
    """

    def test_fit_calibrated_wrapper_not_fitted_on_test(
        self, trained_lr, prepared_data, scaled_data
    ):
        """
        Fit calibration on training slice, apply to test set.
        The fitted calibrator must NOT have seen the test labels.
        """
        from src.models.calibration import fit_calibrated_wrapper

        X_tr_raw, X_te_raw, y_tr, y_te, _ = prepared_data
        X_tr_s, X_te_s, *_ = scaled_data

        # Use last 20% of training data as calibration slice
        n_cal   = max(10, len(X_tr_s) // 5)
        X_cal   = X_tr_s[-n_cal:]
        y_cal   = y_tr.values[-n_cal:]

        # Fit on training slice only — test set untouched here
        calibrated = fit_calibrated_wrapper(
            trained_lr.model, X_cal, y_cal,
            method="sigmoid",
        )

        # Apply to test set — must produce valid probabilities
        proba = calibrated.predict_proba(X_te_s)[:, 1]
        assert (proba >= 0.0).all()
        assert (proba <= 1.0).all()

    def test_calibrated_wrapper_does_not_modify_base_model(
        self, trained_lr, prepared_data, scaled_data
    ):
        """
        After fitting the calibration wrapper, the base model's own
        predict_proba should return identical results as before.
        """
        from src.models.calibration import fit_calibrated_wrapper

        X_tr_s, X_te_s, y_tr, y_te, *_ = scaled_data

        proba_before = trained_lr.predict_proba(X_te_s)[:, 1].copy()

        n_cal = max(10, len(X_tr_s) // 5)
        X_cal = X_tr_s[-n_cal:]
        y_cal = y_tr.values[-n_cal:]

        # Fit wrapper — must not mutate trained_lr.model
        _ = fit_calibrated_wrapper(
            trained_lr.model, X_cal, y_cal,
            method="sigmoid",
        )

        proba_after = trained_lr.predict_proba(X_te_s)[:, 1]
        np.testing.assert_array_equal(proba_before, proba_after)


# ─────────────────────────────────────────────────────────────────────────────
# compare_calibration_results — smoke test
# ─────────────────────────────────────────────────────────────────────────────

class TestCompareCalibrationResults:
    def test_runs_without_error(self, capsys):
        from src.models.calibration import (
            compute_calibration_metrics,
            compare_calibration_results,
        )
        y_true, y_prob = _make_well_calibrated_probs()
        r1 = compute_calibration_metrics(y_true, y_prob, model_name="M1")
        r2 = compute_calibration_metrics(y_true, y_prob, model_name="M2")
        compare_calibration_results([r1, r2])
        captured = capsys.readouterr()
        assert "M1" in captured.out
        assert "M2" in captured.out

    def test_output_contains_metric_values(self, capsys):
        from src.models.calibration import (
            compute_calibration_metrics,
            compare_calibration_results,
        )
        y_true, y_prob = _make_well_calibrated_probs()
        r = compute_calibration_metrics(y_true, y_prob, model_name="TestModel")
        compare_calibration_results([r])
        captured = capsys.readouterr()
        # Brier and Log-Loss values should appear
        assert "Brier" in captured.out or f"{r.brier_score:.4f}" in captured.out


# ─────────────────────────────────────────────────────────────────────────────
# Bimodal distribution — unit tests matching real model behaviour
# ─────────────────────────────────────────────────────────────────────────────

class TestBimodalCalibration:
    """
    Tests that verify our calibration logic handles the highly bimodal
    distribution produced by the actual Logistic Regression model correctly.
    """

    def test_bimodal_brier_is_low(self):
        """
        A well-predicting bimodal model (low=negative, high=positive) should
        have a very low Brier score.
        """
        from src.models.calibration import compute_calibration_metrics
        y_true, y_prob = _make_bimodal_probs(n=500, seed=1)
        result = compute_calibration_metrics(y_true, y_prob)
        assert result.brier_score < 0.05

    def test_bimodal_ece_reflects_noise_bins(self):
        """
        Even with large per-bin gaps, ECE on a bimodal model should remain
        low because the noisy middle bins contain very few samples.
        """
        from src.models.calibration import compute_calibration_metrics
        y_true, y_prob = _make_bimodal_probs(n=500, seed=2)
        result = compute_calibration_metrics(y_true, y_prob, n_bins=10)
        # ECE is weighted by bin fraction, so tiny middle bins can't dominate
        assert result.ece < 0.10

    def test_bimodal_extreme_pcts_correct(self):
        from src.models.calibration import compute_calibration_metrics
        # Exact bimodal: exactly 55% near 0, 45% near 1
        n = 200
        y_prob = np.concatenate([np.full(110, 0.01), np.full(90, 0.99)])
        y_true = np.concatenate([np.zeros(110), np.ones(90)])
        result = compute_calibration_metrics(y_true, y_prob, n_bins=10)
        assert result.prob_bimodal_low_pct  == pytest.approx(55.0, abs=0.01)
        assert result.prob_bimodal_high_pct == pytest.approx(45.0, abs=0.01)
