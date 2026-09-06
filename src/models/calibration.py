"""
Probability calibration utilities for Safiri PortPulse.

This module provides:
  - compute_calibration_metrics()  — Brier, Log-Loss, ECE, per-bin detail
  - CalibrationResult              — typed container for all metrics
  - fit_calibrated_wrapper()       — optional Platt/isotonic wrapper
                                     (only call if materially warranted)

Design notes
------------
* All calibration fitting must use training / validation data only.
* The final test set must never be used to fit a calibration model.
* If the base model is already well-calibrated, no calibration is applied.

A safe calibration workflow when a calibration split is needed:

    Training data (3,840 rows)
          ↓  split chronologically
    Cal-fit set  (earlier)  →  fit CalibratedClassifierCV or Platt
    Cal-val set  (later)    →  compare calibrated vs. original
          ↓
    Final test set (1,040 rows)  →  evaluate, never touched during fitting
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss


# ─────────────────────────────────────────────────────────────────────────────
# Data container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CalibrationResult:
    """All calibration metrics for one model on one dataset."""

    model_name: str

    # Scalar metrics
    brier_score: float
    log_loss_score: float
    ece: float                     # Expected Calibration Error (equal-width bins)

    # Per-bin detail (used for the reliability diagram)
    bin_mean_predicted: List[float]   # mean predicted probability per bin
    bin_fraction_positive: List[float] # observed positive rate per bin
    bin_counts: List[int]             # n samples per bin

    # Probability distribution summary
    prob_min: float
    prob_max: float
    prob_median: float
    prob_bimodal_low_pct: float   # % of predictions ≤ 0.05
    prob_bimodal_high_pct: float  # % of predictions ≥ 0.95

    # Optional: calibration method applied
    calibration_method: Optional[str] = None
    calibration_applied: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name":             self.model_name,
            "brier_score":            round(self.brier_score,     6),
            "log_loss":               round(self.log_loss_score,  6),
            "ece":                    round(self.ece,             6),
            "bin_mean_predicted":     [round(v, 4) for v in self.bin_mean_predicted],
            "bin_fraction_positive":  [round(v, 4) for v in self.bin_fraction_positive],
            "bin_counts":             self.bin_counts,
            "prob_min":               round(self.prob_min,   6),
            "prob_max":               round(self.prob_max,   6),
            "prob_median":            round(self.prob_median, 6),
            "prob_bimodal_low_pct":   round(self.prob_bimodal_low_pct,  4),
            "prob_bimodal_high_pct":  round(self.prob_bimodal_high_pct, 4),
            "calibration_method":     self.calibration_method,
            "calibration_applied":    self.calibration_applied,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Core metric computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_calibration_metrics(
    y_true:     np.ndarray,
    y_prob:     np.ndarray,
    model_name: str = "Model",
    n_bins:     int = 10,
) -> CalibrationResult:
    """
    Compute calibration metrics for a set of probability predictions.

    Parameters
    ----------
    y_true     : Binary ground-truth labels (0/1).
    y_prob     : Predicted positive-class probabilities in [0, 1].
    model_name : Label for reporting.
    n_bins     : Number of equal-width bins for the reliability diagram and ECE.
                 Default 10.  With small test sets, some bins may be empty.

    Returns
    -------
    CalibrationResult
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    if not ((y_prob >= 0).all() and (y_prob <= 1).all()):
        raise ValueError("y_prob must be in [0, 1]")
    if not set(np.unique(y_true)).issubset({0.0, 1.0}):
        raise ValueError("y_true must be binary (0/1)")

    # Scalar metrics
    brier  = float(brier_score_loss(y_true, y_prob))
    ll     = float(log_loss(y_true, y_prob))

    # Equal-width calibration curve (sklearn)
    frac_pos, mean_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )

    # ECE — equal-width bins (manual, so we also capture bin counts)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_mean_pred_list: List[float] = []
    bin_frac_pos_list:  List[float] = []
    bin_count_list:     List[int]   = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # include right edge in the last bin
        if i < n_bins - 1:
            mask = (y_prob >= lo) & (y_prob < hi)
        else:
            mask = (y_prob >= lo) & (y_prob <= hi)

        n_k = int(mask.sum())
        if n_k == 0:
            continue

        conf_k = float(y_prob[mask].mean())
        acc_k  = float(y_true[mask].mean())
        ece   += (n_k / len(y_true)) * abs(acc_k - conf_k)

        bin_mean_pred_list.append(conf_k)
        bin_frac_pos_list.append(acc_k)
        bin_count_list.append(n_k)

    # Probability distribution summary
    return CalibrationResult(
        model_name            = model_name,
        brier_score           = brier,
        log_loss_score        = ll,
        ece                   = float(ece),
        bin_mean_predicted    = bin_mean_pred_list,
        bin_fraction_positive = bin_frac_pos_list,
        bin_counts            = bin_count_list,
        prob_min              = float(y_prob.min()),
        prob_max              = float(y_prob.max()),
        prob_median           = float(np.median(y_prob)),
        prob_bimodal_low_pct  = float((y_prob <= 0.05).mean() * 100),
        prob_bimodal_high_pct = float((y_prob >= 0.95).mean() * 100),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Optional calibration wrapper
# ─────────────────────────────────────────────────────────────────────────────

def fit_calibrated_wrapper(
    base_estimator: Any,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    method: str = "sigmoid",
) -> Any:
    """
    Wrap a pre-fitted estimator with Platt scaling (method='sigmoid') or
    isotonic regression (method='isotonic').

    Uses sklearn CalibratedClassifierCV with cv=None, which treats the
    estimator as pre-fitted and learns the calibration map on X_cal / y_cal.

    IMPORTANT: X_cal / y_cal must be calibration data that was NOT used
    to fit the original model and is NOT the final test set.

    Parameters
    ----------
    base_estimator : A fitted sklearn-compatible estimator.
    X_cal          : Calibration feature matrix (held-out training slice).
    y_cal          : Calibration labels.
    method         : 'sigmoid' (Platt) or 'isotonic'.

    Returns
    -------
    Fitted CalibratedClassifierCV
    """
    # cv=None with a pre-fitted estimator: sklearn ≥ 1.2 dropped cv="prefit".
    # Passing cv=None is the supported replacement.
    calibrated = CalibratedClassifierCV(
        estimator=base_estimator,
        method=method,
        cv=None,
    )
    calibrated.fit(X_cal, y_cal)
    return calibrated


# ─────────────────────────────────────────────────────────────────────────────
# Comparison helper
# ─────────────────────────────────────────────────────────────────────────────

def compare_calibration_results(
    results: List[CalibrationResult],
) -> None:
    """Print a side-by-side comparison of calibration metrics."""
    header = f"  {'Model':<35} {'Brier':>8} {'Log-Loss':>10} {'ECE':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in results:
        print(
            f"  {r.model_name:<35} "
            f"{r.brier_score:>8.4f} "
            f"{r.log_loss_score:>10.4f} "
            f"{r.ece:>8.4f}"
        )
