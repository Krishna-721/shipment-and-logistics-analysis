"""
SHAP-based model explainability for Safiri PortPulse.

Supports two selected models:
  - Logistic Regression  (congestion classification, E-001/E-006)
  - Ridge Regression     (delay prediction, E-007)

Both are linear models fitted on StandardScaler-transformed features, so
``shap.LinearExplainer`` is the correct and exact explainer — no sampling
approximation required.  The explainer is fitted on the *training* background
distribution (mean + covariance of Xtr_scaled) to avoid leaking test-set
information into the explanation baseline.

Public API
----------
PortPulseExplainer
    Wraps a single fitted model + scaler and provides:
    - global_shap_values(X_scaled)         → (n, n_features) ndarray
    - explain_instance(x_scaled)           → FeatureContribution list
    - expected_value                        → float (model baseline)

FeatureContribution
    Named tuple: feature_name, raw_value, shap_value, direction

build_congestion_explainer(...)  → PortPulseExplainer
build_delay_explainer(...)       → PortPulseExplainer

Leakage guarantee
-----------------
The LinearExplainer is constructed with the *training* data only.  Test
instances are passed only to shap_values(), never to the fit step.  The
background masker is a single (mean, cov) summary of Xtr_scaled — no
individual training rows are stored after construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import shap


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FeatureContribution:
    """
    Explanation for a single feature in a single prediction.

    Attributes
    ----------
    feature_name : str
        Human-readable feature label.
    raw_value : float
        Original (unscaled) feature value for this instance.
    shap_value : float
        SHAP contribution to the model output (log-odds for LR,
        hours for Ridge).
    direction : str
        "increases_risk" / "decreases_risk"   for congestion model.
        "increases_delay" / "decreases_delay"  for delay model.
    """
    feature_name: str
    raw_value: float
    shap_value: float
    direction: str

    def to_dict(self) -> dict:
        return {
            "feature_name": self.feature_name,
            "raw_value":    round(float(self.raw_value), 4),
            "shap_value":   round(float(self.shap_value), 6),
            "direction":    self.direction,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Core explainer
# ─────────────────────────────────────────────────────────────────────────────

class PortPulseExplainer:
    """
    Thin wrapper around shap.LinearExplainer for a single fitted model.

    Parameters
    ----------
    model : fitted sklearn linear estimator (LogisticRegression or Ridge)
    X_train_scaled : np.ndarray
        Scaled training data — used to compute the background distribution.
        Individual rows are NOT stored; only mean and covariance are kept.
    feature_names : list[str]
        Names aligned with the columns of X_train_scaled.
    task : {"classification", "regression"}
        Controls the direction labels on FeatureContribution.
    model_label : str
        Human-readable name for logging and plot titles.
    """

    def __init__(
        self,
        model,
        X_train_scaled: np.ndarray,
        feature_names: list,
        task: str = "classification",
        model_label: str = "Model",
    ) -> None:
        if task not in {"classification", "regression"}:
            raise ValueError(f"task must be 'classification' or 'regression', got {task!r}")

        self.feature_names = list(feature_names)
        self.task          = task
        self.model_label   = model_label
        self.n_features    = len(feature_names)

        # Fit LinearExplainer on training background using Independent masker.
        # shap.maskers.Independent stores the training distribution as the
        # reference so each feature is perturbed independently of the others.
        # This is the correct masker for linear models and avoids the deprecated
        # feature_perturbation="correlation_dependent" keyword in SHAP >= 0.42.
        # The masker holds a copy of X_train_scaled so SHAP can marginalise over
        # it; this is the training set only — no test rows are ever passed here.
        masker = shap.maskers.Independent(X_train_scaled, max_samples=len(X_train_scaled))
        self._explainer = shap.LinearExplainer(model, masker)

        # Baseline output (expected value over training distribution)
        self.expected_value: float = float(self._explainer.expected_value)

    # ── global explanations ───────────────────────────────────────────────────

    def global_shap_values(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Compute SHAP values for a batch of scaled instances.

        Returns
        -------
        np.ndarray of shape (n_samples, n_features)
            For LogisticRegression the values are in log-odds space.
            For Ridge they are in the target unit (hours).
        """
        sv = self._explainer.shap_values(X_scaled)
        arr = np.asarray(sv)
        # LogisticRegression LinearExplainer may return shape (n,) per class
        # or (n, n_features) directly — normalise to 2-D
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[0] == self.n_features and arr.shape != (len(X_scaled), self.n_features):
            # Transposed: (n_features, n_samples) → fix
            arr = arr.T
        return arr  # shape (n_samples, n_features)

    def build_shap_explanation(
        self,
        X_scaled: np.ndarray,
        shap_values: Optional[np.ndarray] = None,
    ) -> shap.Explanation:
        """
        Build a shap.Explanation object suitable for shap.plots.*.

        Parameters
        ----------
        X_scaled : np.ndarray  shape (n, n_features)
        shap_values : optional pre-computed array to avoid recomputation
        """
        if shap_values is None:
            shap_values = self.global_shap_values(X_scaled)

        return shap.Explanation(
            values=shap_values,
            base_values=np.full(len(X_scaled), self.expected_value),
            data=X_scaled,
            feature_names=self.feature_names,
        )

    # ── individual explanations ────────────────────────────────────────────────

    def explain_instance(
        self,
        x_scaled: np.ndarray,
        x_raw: Optional[np.ndarray] = None,
        top_n: int = 10,
    ) -> List[FeatureContribution]:
        """
        Explain a single prediction.

        Parameters
        ----------
        x_scaled : np.ndarray shape (n_features,) or (1, n_features)
            Scaled feature vector fed to the model.
        x_raw : np.ndarray shape (n_features,) or None
            Unscaled feature values for human-readable output.
            If None, x_scaled values are used as raw values.
        top_n : int
            Number of top contributors to return (by |SHAP value|).

        Returns
        -------
        list of FeatureContribution, sorted by |shap_value| descending.
        """
        x_scaled = np.asarray(x_scaled, dtype=float).reshape(1, -1)
        if x_scaled.shape[1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {x_scaled.shape[1]}"
            )

        sv_row = self.global_shap_values(x_scaled)[0]  # shape (n_features,)

        if x_raw is None:
            x_raw_arr = x_scaled[0]
        else:
            x_raw_arr = np.asarray(x_raw, dtype=float).reshape(-1)

        contributions = []
        for i, (feat, sv, rv) in enumerate(
            zip(self.feature_names, sv_row, x_raw_arr)
        ):
            direction = self._direction(sv)
            contributions.append(FeatureContribution(
                feature_name=feat,
                raw_value=float(rv),
                shap_value=float(sv),
                direction=direction,
            ))

        # Sort by absolute SHAP value, largest first
        contributions.sort(key=lambda c: abs(c.shap_value), reverse=True)
        return contributions[:top_n]

    # ── helpers ───────────────────────────────────────────────────────────────

    def _direction(self, shap_value: float) -> str:
        if self.task == "classification":
            return "increases_risk" if shap_value > 0 else "decreases_risk"
        return "increases_delay" if shap_value > 0 else "decreases_delay"

    def mean_abs_shap(self, shap_values: np.ndarray) -> np.ndarray:
        """Mean |SHAP| per feature across all rows — shape (n_features,)."""
        return np.abs(shap_values).mean(axis=0)

    def global_importance_dict(self, shap_values: np.ndarray) -> dict:
        """
        Return {feature_name: mean_abs_shap} sorted descending by importance.
        """
        mean_abs = self.mean_abs_shap(shap_values)
        pairs = sorted(
            zip(self.feature_names, mean_abs.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return dict(pairs)

    def __repr__(self) -> str:
        return (
            f"PortPulseExplainer("
            f"model={self.model_label!r}, "
            f"task={self.task!r}, "
            f"n_features={self.n_features}, "
            f"expected_value={self.expected_value:.4f})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Factory helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_congestion_explainer(
    lr_model,
    scaler,
    X_train: "np.ndarray | pd.DataFrame",  # type: ignore[name-defined]
    feature_names: list,
) -> PortPulseExplainer:
    """
    Build a PortPulseExplainer for the Logistic Regression congestion model.

    The scaler is applied to X_train here so the explainer background
    distribution matches exactly what the model saw during training.

    Parameters
    ----------
    lr_model     : fitted sklearn LogisticRegression (extracted from payload)
    scaler       : fitted StandardScaler (from scaler.joblib)
    X_train      : unscaled training feature DataFrame or ndarray
    feature_names: 32-element list aligned with X_train columns
    """
    X_train_arr = np.asarray(X_train, dtype=float)
    X_train_scaled = scaler.transform(X_train_arr)

    return PortPulseExplainer(
        model=lr_model,
        X_train_scaled=X_train_scaled,
        feature_names=feature_names,
        task="classification",
        model_label="Logistic Regression (Congestion)",
    )


def build_delay_explainer(
    ridge_model,
    delay_scaler,
    X_train: "np.ndarray | pd.DataFrame",  # type: ignore[name-defined]
    feature_names: list,
) -> PortPulseExplainer:
    """
    Build a PortPulseExplainer for the Ridge Regression delay model.

    Parameters
    ----------
    ridge_model  : fitted sklearn Ridge (extracted from payload)
    delay_scaler : fitted StandardScaler (from delay_scaler.joblib)
    X_train      : unscaled training feature DataFrame or ndarray
    feature_names: 32-element list aligned with X_train columns
    """
    X_train_arr = np.asarray(X_train, dtype=float)
    X_train_scaled = delay_scaler.transform(X_train_arr)

    return PortPulseExplainer(
        model=ridge_model,
        X_train_scaled=X_train_scaled,
        feature_names=feature_names,
        task="regression",
        model_label="Ridge Regression (Delay)",
    )
