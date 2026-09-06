"""
FastAPI application entry point for Safiri PortPulse.

Endpoints
---------
POST /predict   — Full prediction pipeline: congestion + delay + SHAP + recommendation.
GET  /health    — Liveness + model-readiness check.

All ML inference, SHAP, and recommendation logic lives in
backend/services/prediction_service.py.  This file contains only HTTP
concerns (routing, error handling, serialisation).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.schemas import HealthResponse, PredictionRequest, PredictionResponse
from backend.services.prediction_service import ModelRegistry, PredictionService

# ─────────────────────────────────────────────────────────────────────────────
# Application state
# ─────────────────────────────────────────────────────────────────────────────

_registry: ModelRegistry | None = None
_service:  PredictionService | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan — load models once at startup
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts at startup; release nothing on shutdown."""
    global _registry, _service
    try:
        _registry = ModelRegistry.load()
        _service  = PredictionService(_registry)
    except RuntimeError as exc:
        # Log but don't crash the process; /health will report degraded state.
        import sys
        print(f"[PortPulse] WARNING — model loading failed: {exc}", file=sys.stderr)
        _registry = None
        _service  = None
    yield
    # Nothing to clean up for in-process models.


# ─────────────────────────────────────────────────────────────────────────────
# Application
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Safiri PortPulse Prediction API",
    description=(
        "Deterministic port congestion and delay prediction with "
        "SHAP-based explanations and operational recommendations."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ─────────────────────────────────────────────────────────────────────────────
# CORS — local development only
#
# Allows the vanilla HTML/JS frontend served from any local origin to call
# this API.  Explicitly lists only localhost addresses used during local
# development; this is NOT a production CORS policy.
#
# Permitted origins:
#   http://localhost:8000   — Python http.server / similar dev server
#   http://127.0.0.1:8000  — same, by numeric address
#   http://localhost:5500   — VS Code Live Server default
#   http://127.0.0.1:5500  — same, by numeric address
#
# To add a deployed frontend origin, extend ALLOWED_ORIGINS explicitly —
# do not use wildcard ("*") in production as it disables credentials.
# ─────────────────────────────────────────────────────────────────────────────

_ALLOWED_ORIGINS = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Exception handlers
# ─────────────────────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def _global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch unexpected errors and return a clean 500."""
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {type(exc).__name__}: {exc}"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """
    Liveness and model-readiness check.

    Returns status="ok" when all model artifacts are loaded.
    Returns status="degraded" when models failed to load at startup.
    """
    if _registry is None:
        return HealthResponse(
            status        = "degraded",
            models_loaded = False,
            shap_available= False,
            message       = (
                "Model artifacts could not be loaded at startup. "
                "Run train.py and train_delay.py to generate required files."
            ),
        )

    return HealthResponse(
        status        = "ok",
        models_loaded = True,
        shap_available= _registry.shap_available,
        message       = (
            "All models loaded."
            if _registry.shap_available
            else "Models loaded; SHAP explainer unavailable (predictions still work)."
        ),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
async def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Full prediction pipeline for a single port operational snapshot.

    Accepts 32 prediction-time features and returns:
    - Congestion probability (Logistic Regression)
    - Expected delay in hours (Ridge Regression)
    - Risk level (LOW / MEDIUM / HIGH)
    - SHAP-based risk-increasing and protective factors
    - Deterministic operational recommendation
    - Human-readable reason and any escalation triggers

    Target and metadata fields (congestion_label, future_delay_hours,
    is_valid_training_row, port_id, timestamp) are explicitly excluded from
    the request schema. Submitting them returns 422 Unprocessable Entity.
    """
    if _service is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Prediction service unavailable — model artifacts not loaded. "
                "Check /health for details."
            ),
        )

    result = _service.predict(request)
    return PredictionResponse(**result)
