"""
Service layer for loading model artifacts and serving predictions.

ModelRegistry
    Loaded once at application startup (FastAPI lifespan).
    Holds all four joblib artifacts + training background data for SHAP.
    Raises RuntimeError on any missing/corrupt artifact so the app refuses
    to start rather than silently producing invalid predictions.

PredictionService
    Stateless — receives a ModelRegistry and performs one prediction per call.
    Steps:
      1. Build partial feature dict from the validated PredictionRequest (27 raw fields).
      2. Derive the 5 internal features using the exact generator formulas.
      3. Assemble the full 32-column DataFrame in scaler column order.
      4. Scale with LR scaler → run LR → congestion probability.
      5. Scale with Ridge scaler → run Ridge → predicted delay hours.
      6. Run SHAP LinearExplainer on the scaled row → FeatureContribution lists.
      7. Build a RecommendationInput from all the above.
      8. Call RecommendationEngine.recommend() → RecommendationResult.
      9. Return a dict compatible with PredictionResponse.

Derived feature formulas (from generator._calculate_derived_features)
----------------------------------------------------------------------
  capacity_pressure          = clip(berth_utilization, 0, 1)
  queue_pressure             = clip(queue_length / total_berths, 0, 3)
  equipment_pressure         = clip(1 - cranes_operational /
                                    max(2, ceil(total_berths * 1.2)), 0, 1)
  weather_pressure           = weather_severity
  queue_capacity_interaction = queue_pressure * capacity_pressure

These five are NEVER accepted from the client — they are always computed here.
Reconstruction error against the training CSV is ≤ 1e-15 (floating-point only).

SHAP fallback
    If SHAP fails for any reason the service catches the exception and calls
    RecommendationEngine without SHAP contributions.  The E-009 operational
    fallback handles the rest gracefully.  shap_available=False is set.

No logic from src.recommendations is duplicated here.
No ML-related code appears in main.py.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from src.config import ARTIFACTS_MODELS
from src.explainability.formatter import human_label
from src.explainability.shap_explainer import (
    FeatureContribution,
    build_congestion_explainer,
    build_delay_explainer,
)
from src.recommendations.engine import (
    DEFAULT_CONFIG,
    RecommendationConfig,
    RecommendationEngine,
    RecommendationInput,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Artifact paths
# ─────────────────────────────────────────────────────────────────────────────

_LR_PATH           = ARTIFACTS_MODELS / "logistic_regression.joblib"
_SCALER_PATH       = ARTIFACTS_MODELS / "scaler.joblib"
_RIDGE_PATH        = ARTIFACTS_MODELS / "delay_ridge.joblib"
_DELAY_SCALER_PATH = ARTIFACTS_MODELS / "delay_scaler.joblib"


# ─────────────────────────────────────────────────────────────────────────────
# Derived feature computation
# ─────────────────────────────────────────────────────────────────────────────

import math


def _derive_features(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute the five internally-derived features from raw API inputs.

    Formulas are taken verbatim from
    src/data/generator.py::_calculate_derived_features().
    Reconstruction error against the training CSV is ≤ 1e-15.

    Parameters
    ----------
    raw : dict from PredictionRequest.to_feature_row()
          (27 raw fields — the 5 derived slots are absent)

    Returns
    -------
    dict with the 5 derived features added, ready for DataFrame assembly.
    """
    total_berths      = float(raw["total_berths"])
    berth_utilization = float(raw["berth_utilization"])
    queue_length      = float(raw["queue_length"])
    cranes_op         = float(raw["cranes_operational"])
    weather_sev       = float(raw["weather_severity"])

    # cranes_available = max(2, ceil(total_berths * 1.2))  [exact generator formula]
    cranes_available = float(max(2, math.ceil(total_berths * 1.2)))

    capacity_pressure = float(np.clip(berth_utilization, 0.0, 1.0))
    queue_pressure    = float(np.clip(queue_length / max(total_berths, 1.0), 0.0, 3.0))
    equipment_pressure = float(
        np.clip(1.0 - cranes_op / max(cranes_available, 1.0), 0.0, 1.0)
    )
    weather_pressure          = float(weather_sev)
    queue_capacity_interaction = float(queue_pressure * capacity_pressure)

    return {
        "capacity_pressure":          capacity_pressure,
        "queue_pressure":             queue_pressure,
        "equipment_pressure":         equipment_pressure,
        "weather_pressure":           weather_pressure,
        "queue_capacity_interaction": queue_capacity_interaction,
    }


def _build_feature_row(
    raw: Dict[str, Any],
    feature_names: List[str],
) -> Dict[str, Any]:
    """
    Merge the 27 raw client fields with the 5 derived features and return
    a single dict whose keys are in the exact ``feature_names`` order.

    This guarantees that dict key order matches the scaler's
    feature_names_in_ before the DataFrame is constructed, making
    the column ordering explicit and verifiable in tests.

    The scaler column order is:
      0  total_berths … 23 traffic_pressure
      24 capacity_pressure          ← derived
      25 queue_pressure             ← derived
      26 equipment_pressure         ← derived
      27 weather_pressure           ← derived
      28 queue_capacity_interaction ← derived
      29 traffic_weather_interaction
      30 vessel_type
      31 cargo_type
    """
    derived = _derive_features(raw)
    combined = {**raw, **derived}   # all 32 values keyed by name

    # Verify nothing is missing before assembling
    missing = [n for n in feature_names if n not in combined]
    if missing:
        raise RuntimeError(
            f"Feature vector is missing columns: {missing}. "
            "This is a service bug, not a client error."
        )

    # Return in explicit scaler order so dict key order == column order
    return {name: combined[name] for name in feature_names}

@dataclass
class ModelRegistry:
    """
    Holds all loaded artifacts.  Build with ModelRegistry.load().

    Attributes
    ----------
    lr_model      : fitted LogisticRegression
    scaler        : StandardScaler for LR input
    ridge_model   : fitted Ridge
    delay_scaler  : StandardScaler for Ridge input
    feature_names : ordered list of 32 feature column names
    cong_explainer : PortPulseExplainer for congestion (or None)
    delay_explainer: PortPulseExplainer for delay (or None)
    """
    lr_model:        Any
    scaler:          Any
    ridge_model:     Any
    delay_scaler:    Any
    feature_names:   List[str]

    # SHAP explainers — built lazily from a small background sample
    cong_explainer:  Optional[Any] = field(default=None)
    delay_explainer: Optional[Any] = field(default=None)

    @classmethod
    def load(cls, background_n: int = 200) -> "ModelRegistry":
        """
        Load all four joblib artifacts.
        Raises RuntimeError if any file is missing or unreadable.

        background_n : number of training rows to keep for SHAP background.
            Using a sample (not all 3,840 rows) keeps the explainer small
            and construction fast, while still giving a representative
            background distribution.
        """
        for path in [_LR_PATH, _SCALER_PATH, _RIDGE_PATH, _DELAY_SCALER_PATH]:
            if not Path(path).exists():
                raise RuntimeError(
                    f"Required model artifact not found: {path}\n"
                    "Run 'python scripts/train.py' and 'python scripts/train_delay.py' "
                    "to generate all artifacts."
                )

        try:
            lr_payload    = joblib.load(_LR_PATH)
            scaler        = joblib.load(_SCALER_PATH)
            ridge_payload = joblib.load(_RIDGE_PATH)
            delay_scaler  = joblib.load(_DELAY_SCALER_PATH)
        except Exception as exc:
            raise RuntimeError(f"Failed to load model artifacts: {exc}") from exc

        lr_model    = lr_payload["model"]
        ridge_model = ridge_payload["model"]
        feature_names = list(scaler.feature_names_in_)

        # Build SHAP explainers using a small background sample from the
        # training distribution (stored inside the scaler's feature_names_in_).
        # We reconstruct a background by loading the actual training data;
        # if that fails we disable SHAP gracefully.
        cong_explainer  = None
        delay_explainer = None
        try:
            X_bg = _load_training_background(feature_names, scaler, background_n)
            cong_explainer  = build_congestion_explainer(
                lr_model, scaler, X_bg, feature_names
            )
            delay_explainer = build_delay_explainer(
                ridge_model, delay_scaler, X_bg, feature_names
            )
        except Exception:
            # SHAP explainer construction failed — prediction still works,
            # recommendation engine falls back to operational factors.
            pass

        return cls(
            lr_model       = lr_model,
            scaler         = scaler,
            ridge_model    = ridge_model,
            delay_scaler   = delay_scaler,
            feature_names  = feature_names,
            cong_explainer = cong_explainer,
            delay_explainer= delay_explainer,
        )

    @property
    def shap_available(self) -> bool:
        return self.cong_explainer is not None and self.delay_explainer is not None


def _load_training_background(
    feature_names: List[str],
    scaler: Any,
    n: int,
) -> pd.DataFrame:
    """
    Load a small background sample from the real training data.
    Falls back to random normal data scaled to fit feature_names if the
    dataset file is unavailable (e.g. in isolated test environments).
    """
    try:
        from src.config import load_dataset
        from src.data.preprocessing import create_temporal_split, prepare_features
        df = load_dataset()
        train_df, _ = create_temporal_split(df)
        X_train, _, _, _, _ = prepare_features(train_df, train_df)
        # Subsample for speed
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_train), size=min(n, len(X_train)), replace=False)
        return X_train.iloc[idx].reset_index(drop=True)
    except Exception:
        # Return a minimal background using scaler mean as a single row
        mean_arr = scaler.mean_
        return pd.DataFrame([mean_arr], columns=feature_names)


# ─────────────────────────────────────────────────────────────────────────────
# Prediction service
# ─────────────────────────────────────────────────────────────────────────────

class PredictionService:
    """
    Stateless prediction service.  All mutable state lives in ModelRegistry.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        rec_config: Optional[RecommendationConfig] = None,
    ) -> None:
        self._reg    = registry
        self._engine = RecommendationEngine(rec_config or DEFAULT_CONFIG)

    def predict(self, request: "PredictionRequest") -> Dict[str, Any]:  # type: ignore[name-defined]
        """
        Run the full prediction pipeline for one request.

        Parameters
        ----------
        request : PredictionRequest (validated Pydantic model)

        Returns
        -------
        dict compatible with PredictionResponse
        """
        # 1 — Build the 27-field raw dict from the request, then derive the
        #     5 internally-computed features and assemble all 32 columns in
        #     the exact order the scaler expects.
        raw_row     = request.to_feature_row()
        feature_row = _build_feature_row(raw_row, self._reg.feature_names)
        X_df = pd.DataFrame([feature_row], columns=self._reg.feature_names)

        # 2 — Scale for LR and predict congestion probability
        X_lr = self._reg.scaler.transform(X_df)
        congestion_prob = float(
            self._reg.lr_model.predict_proba(X_lr)[0, 1]
        )

        # 3 — Scale for Ridge and predict delay
        X_ridge = self._reg.delay_scaler.transform(X_df)
        predicted_delay = float(
            np.clip(self._reg.ridge_model.predict(X_ridge)[0], 0.0, None)
        )

        # 4 — SHAP contributions (optional)
        cong_contribs:  List[FeatureContribution] = []
        delay_contribs: List[FeatureContribution] = []
        shap_ok = False

        if self._reg.shap_available:
            try:
                x_raw = X_df.values[0]
                cong_contribs = self._reg.cong_explainer.explain_instance(
                    x_scaled=X_lr[0], x_raw=x_raw, top_n=10
                )
                delay_contribs = self._reg.delay_explainer.explain_instance(
                    x_scaled=X_ridge[0], x_raw=x_raw, top_n=10
                )
                shap_ok = True
            except Exception:
                cong_contribs  = []
                delay_contribs = []

        # 5 — Build RecommendationInput
        # queue_pressure is now derived internally — read from feature_row, not raw_row
        rec_inp = RecommendationInput(
            congestion_probability   = congestion_prob,
            predicted_delay_hours    = predicted_delay,
            queue_pressure           = feature_row.get("queue_pressure"),
            berth_utilization        = feature_row.get("berth_utilization"),
            available_berths         = int(feature_row.get("available_berths", 0)),
            avg_waiting_time_hours   = feature_row.get("avg_waiting_time_hours"),
            congestion_contributions = cong_contribs if cong_contribs else None,
            delay_contributions      = delay_contribs if delay_contribs else None,
        )

        # 6 — Get recommendation
        rec_result = self._engine.recommend(rec_inp)

        # 7 — Serialise factors
        # human_label() is from src.explainability.formatter — the same 32-entry
        # label map used by the SHAP experiment scripts.  It converts snake_case
        # feature names to operator-readable strings (e.g. "queue_length" →
        # "Queue length (vessels)") so the frontend receives meaningful labels
        # without duplicating the mapping in JS.
        def _serialise_factors(
            factors: List[Dict[str, Any]]
        ) -> List[Dict[str, Any]]:
            out = []
            for f in factors:
                feat_name = f.get("feature_name", "")
                out.append({
                    "source":       f.get("source", ""),
                    "feature_name": feat_name,
                    "label":        human_label(feat_name),
                    "raw_value":    float(f.get("raw_value") or 0.0),
                    "shap_value":   (
                        float(f["shap_value"])
                        if f.get("shap_value") is not None else None
                    ),
                    "direction":    f.get("direction", ""),
                })
            return out

        return {
            "congestion_probability":  round(congestion_prob,   4),
            "predicted_delay_hours":   round(predicted_delay,   4),
            "risk_level":              rec_result.risk_level,
            "risk_increasing_factors": _serialise_factors(
                rec_result.risk_increasing_factors
            ),
            "protective_factors":      _serialise_factors(
                rec_result.protective_factors
            ),
            "actions":                 rec_result.actions,
            "reason":                  rec_result.reason,
            "escalation_triggers":     rec_result.escalation_triggers,
            "shap_available":          shap_ok,
        }
