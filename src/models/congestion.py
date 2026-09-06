"""Port congestion classification models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.config import ARTIFACTS_MODELS, RANDOM_STATE


class CongestionClassifier:
    """
    Unified wrapper around Logistic Regression, Random Forest, and XGBoost
    for congestion classification.

    The wrapper deliberately keeps the sklearn-style fit / predict / predict_proba
    interface consistent so the training script can treat all models identically.

    Parameters
    ----------
    model_type : {"logistic", "random_forest", "xgboost"}
    **kwargs   : Passed to the underlying estimator at construction time.
                 Keys that do not apply to the chosen estimator are silently
                 ignored (e.g. n_estimators for logistic).
    """

    _VALID_TYPES = {"logistic", "random_forest", "xgboost"}

    def __init__(self, model_type: str = "logistic", **kwargs: Any) -> None:
        if model_type not in self._VALID_TYPES:
            raise ValueError(
                f"model_type must be one of {self._VALID_TYPES}, got '{model_type}'"
            )
        self.model_type = model_type
        self.init_kwargs: Dict[str, Any] = kwargs
        self.is_fitted: bool = False
        self.model = self._build_model(**kwargs)

    # ── construction ────────────────────────────────────────────────────────

    def _build_model(self, **kwargs: Any):
        if self.model_type == "logistic":
            return LogisticRegression(
                random_state=RANDOM_STATE,
                max_iter=1000,
                solver="lbfgs",
                C=kwargs.get("C", 1.0),
            )

        if self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=kwargs.get("n_estimators", 100),
                max_depth=kwargs.get("max_depth", None),
                min_samples_leaf=kwargs.get("min_samples_leaf", 1),
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )

        if self.model_type == "xgboost":
            try:
                import xgboost as xgb
            except ImportError as exc:
                raise ImportError(
                    "XGBoost is not installed. "
                    "Run: pip install xgboost==2.1.1"
                ) from exc

            return xgb.XGBClassifier(
                n_estimators=kwargs.get("n_estimators", 100),
                max_depth=kwargs.get("max_depth", 5),
                learning_rate=kwargs.get("learning_rate", 0.1),
                subsample=kwargs.get("subsample", 0.8),
                colsample_bytree=kwargs.get("colsample_bytree", 0.8),
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                verbosity=0,
            )

    # ── sklearn interface ────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CongestionClassifier":
        """Train on X, y."""
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return hard class predictions."""
        self._check_fitted()
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities shaped (n, 2)."""
        self._check_fitted()
        return self.model.predict_proba(X)

    # ── interpretability ─────────────────────────────────────────────────────

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Return feature-level importance as a 1-D array aligned with the
        feature matrix columns.

        * Tree models   → mean decrease in impurity (feature_importances_)
        * Logistic Reg  → |coefficient| after standardised scaling
        * Otherwise     → None
        """
        if not self.is_fitted:
            return None
        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        if hasattr(self.model, "coef_"):
            return np.abs(self.model.coef_[0])
        return None

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, filepath: Path) -> None:
        """Serialise fitted model to disk."""
        self._check_fitted()
        payload = {
            "model":      self.model,
            "model_type": self.model_type,
            "init_kwargs": self.init_kwargs,
        }
        joblib.dump(payload, filepath)
        print(f"  Model saved → {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> "CongestionClassifier":
        """Deserialise a saved model."""
        payload = joblib.load(filepath)
        instance = cls.__new__(cls)
        instance.model_type  = payload["model_type"]
        instance.init_kwargs = payload["init_kwargs"]
        instance.model       = payload["model"]
        instance.is_fitted   = True
        return instance

    # ── helpers ───────────────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                f"CongestionClassifier (type={self.model_type}) has not been "
                "fitted yet. Call .fit(X, y) first."
            )

    def get_params(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {"model_type": self.model_type}
        if self.model is not None and hasattr(self.model, "get_params"):
            params.update(self.model.get_params())
        return params

    def __repr__(self) -> str:
        return (
            f"CongestionClassifier(type={self.model_type!r}, "
            f"fitted={self.is_fitted})"
        )
