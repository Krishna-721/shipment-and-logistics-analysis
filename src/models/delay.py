"""
Future delay prediction models for Safiri PortPulse.

Predicts ``future_delay_hours`` — expected operational delay over the next
6 hours — using only prediction-time features.

Models provided
---------------
- DelayBaseline        : predict train-set mean (constant predictor)
- DelayRegressor       : unified wrapper for Ridge, Random Forest, XGBoost

Design notes
------------
* Target: future_delay_hours  (continuous, ≥ 0, right-skewed)
* Same 32 features as the congestion classifier.
* Ridge uses StandardScaler (fitted on training data only).
* Tree models are scale-invariant; scaler is applied optionally.
* All models clip predictions to [0, ∞) — negative delay is non-physical.
* The diagnostic confirmed a very strong queue-state correlation (r ≈ 0.96),
  so high R² is expected and reflects dataset structure, not model sophistication.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import ARTIFACTS_MODELS, RANDOM_STATE


# ─────────────────────────────────────────────────────────────────────────────
# Constant baseline
# ─────────────────────────────────────────────────────────────────────────────

class DelayBaseline:
    """
    Constant predictor: always predicts the training-set mean delay.

    This is the correct naive baseline for regression.  It deliberately does
    not use the test-set mean so as to reflect realistic deployment conditions
    where only historical (training) statistics are available at prediction time.

    The median variant is also tracked because the delay distribution is
    right-skewed (skew ≈ 1.92), making median a meaningful alternative baseline.
    """

    def __init__(self) -> None:
        self.train_mean:   Optional[float] = None
        self.train_median: Optional[float] = None
        self.is_fitted:    bool = False

    def fit(self, y_train: np.ndarray) -> "DelayBaseline":
        y_train = np.asarray(y_train, dtype=float)
        self.train_mean   = float(y_train.mean())
        self.train_median = float(np.median(y_train))
        self.is_fitted    = True
        return self

    def predict_mean(self, n: int) -> np.ndarray:
        self._check_fitted()
        return np.full(n, self.train_mean)

    def predict_median(self, n: int) -> np.ndarray:
        self._check_fitted()
        return np.full(n, self.train_median)

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("DelayBaseline must be fitted before prediction.")

    def __repr__(self) -> str:
        if self.is_fitted:
            return (f"DelayBaseline(mean={self.train_mean:.3f}, "
                    f"median={self.train_median:.3f})")
        return "DelayBaseline(unfitted)"


# ─────────────────────────────────────────────────────────────────────────────
# Regression wrapper
# ─────────────────────────────────────────────────────────────────────────────

class DelayRegressor:
    """
    Unified wrapper for Ridge, Random Forest, and XGBoost regression.

    Interface mirrors CongestionClassifier so downstream code is consistent.

    Parameters
    ----------
    model_type : {"ridge", "random_forest", "xgboost"}
    **kwargs   : Forwarded to the underlying estimator.
    """

    _VALID_TYPES = {"ridge", "random_forest", "xgboost"}

    def __init__(self, model_type: str = "ridge", **kwargs: Any) -> None:
        if model_type not in self._VALID_TYPES:
            raise ValueError(
                f"model_type must be one of {self._VALID_TYPES}, got '{model_type}'"
            )
        self.model_type  = model_type
        self.init_kwargs: Dict[str, Any] = kwargs
        self.is_fitted:  bool = False
        self.model = self._build_model(**kwargs)

    # ── construction ─────────────────────────────────────────────────────────

    def _build_model(self, **kwargs: Any):
        if self.model_type == "ridge":
            return Ridge(
                alpha=kwargs.get("alpha", 1.0),
                random_state=RANDOM_STATE,
            )

        if self.model_type == "random_forest":
            return RandomForestRegressor(
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
                    "XGBoost is not installed.  Run: pip install xgboost==2.1.1"
                ) from exc

            return xgb.XGBRegressor(
                n_estimators=kwargs.get("n_estimators", 100),
                max_depth=kwargs.get("max_depth", 5),
                learning_rate=kwargs.get("learning_rate", 0.1),
                subsample=kwargs.get("subsample", 0.8),
                colsample_bytree=kwargs.get("colsample_bytree", 0.8),
                random_state=RANDOM_STATE,
                verbosity=0,
            )

    # ── sklearn interface ─────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DelayRegressor":
        """Train on X, y.  y values must be ≥ 0."""
        y = np.asarray(y, dtype=float)
        if (y < 0).any():
            raise ValueError("Delay target contains negative values.")
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted delay in hours, clipped to [0, ∞)."""
        self._check_fitted()
        raw = self.model.predict(X)
        return np.clip(raw, 0.0, None)

    # ── interpretability ─────────────────────────────────────────────────────

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Return 1-D importance array aligned with training feature columns.

        * Ridge             → |coefficient| (requires standardised input)
        * Tree models       → mean decrease in impurity / XGB gain
        """
        if not self.is_fitted:
            return None
        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        if hasattr(self.model, "coef_"):
            return np.abs(self.model.coef_)
        return None

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, filepath: Path) -> None:
        self._check_fitted()
        payload = {
            "model":       self.model,
            "model_type":  self.model_type,
            "init_kwargs": self.init_kwargs,
        }
        joblib.dump(payload, filepath)
        print(f"  Model saved → {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> "DelayRegressor":
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
                f"DelayRegressor (type={self.model_type!r}) has not been "
                "fitted yet. Call .fit(X, y) first."
            )

    def get_params(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {"model_type": self.model_type}
        if self.model is not None and hasattr(self.model, "get_params"):
            params.update(self.model.get_params())
        return params

    def __repr__(self) -> str:
        return (
            f"DelayRegressor(type={self.model_type!r}, "
            f"fitted={self.is_fitted})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Regression metrics
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_delay_model(
    y_true:     np.ndarray,
    y_pred:     np.ndarray,
    model_name: str = "Model",
) -> Dict[str, Any]:
    """
    Compute MAE, RMSE, and R² for a regression prediction.

    Parameters
    ----------
    y_true     : Actual delay values.
    y_pred     : Predicted delay values.
    model_name : Label used in reports and plots.

    Returns
    -------
    dict with keys:
        model_name, n_samples, target_mean, target_std,
        mae, rmse, r2.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))

    return {
        "model_name":   model_name,
        "n_samples":    int(len(y_true)),
        "target_mean":  float(y_true.mean()),
        "target_std":   float(y_true.std()),
        "target_min":   float(y_true.min()),
        "target_max":   float(y_true.max()),
        "mae":          mae,
        "rmse":         rmse,
        "r2":           r2,
    }


def format_delay_results_table(results: list) -> "pd.DataFrame":  # type: ignore[name-defined]
    """Return a wide DataFrame for CSV export and console display."""
    import pandas as pd
    rows = []
    for r in results:
        rows.append({
            "Model": r["model_name"],
            "MAE":   round(r["mae"],  4),
            "RMSE":  round(r["rmse"], 4),
            "R2":    round(r["r2"],   4),
        })
    return pd.DataFrame(rows)
