"""
Tests for E-010 — FastAPI Prediction API  (updated for E-010 consistency fix).

Uses FastAPI TestClient — no server process needed.
The lifespan loads the real model artifacts from artifacts/models/.

Schema change (consistency fix)
--------------------------------
Five derived features are no longer accepted from clients:
  capacity_pressure, queue_pressure, equipment_pressure,
  weather_pressure, queue_capacity_interaction.
They are computed internally by PredictionService using the exact
formulas from the training generator.  Submitting them returns 422.

traffic_pressure and traffic_weather_interaction are still client inputs
(require daily_vessel_capacity, a hidden port constant).

Covers:
- GET /health
- POST /predict: happy-path, response schema, value ranges
- Congestion probability in [0, 1]
- Predicted delay is non-negative
- Risk level is LOW / MEDIUM / HIGH
- SHAP factors structured correctly
- Determinism
- Recommendation fields
- Missing required fields → 422
- Out-of-range values → 422
- Cross-field validation (available_berths ≤ total_berths)
- Categorical validation
- Leakage prevention: target/metadata fields → 422
- Derived feature override prevention: the 5 derived fields → 422
- Derived formula correctness (unit tests, no HTTP)
- PredictionService builds correct 32-col vector
- Model predictions valid with derived features
- Encoding constants
"""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.schemas import (
    CARGO_TYPE_ENCODING,
    VESSEL_TYPE_ENCODING,
    PredictionRequest,
)
from backend.services.prediction_service import _derive_features, _build_feature_row

# ─────────────────────────────────────────────────────────────────────────────
# Shared client (models loaded once per session via lifespan)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """TestClient that triggers the FastAPI lifespan (loads real models)."""
    with TestClient(app) as c:
        yield c


# ─────────────────────────────────────────────────────────────────────────────
# Canonical valid payloads — 27 raw fields only (no derived fields)
# ─────────────────────────────────────────────────────────────────────────────

def _valid_payload() -> Dict[str, Any]:
    """Complete valid PredictionRequest payload — 27 raw client fields."""
    return {
        "total_berths":               8,
        "available_berths":           5,
        "berth_utilization":          0.375,
        "vessels_currently_in_port":  3,
        "vessels_anchored":           1,
        "vessels_approaching":        2,
        "arrivals_last_1h":           1,
        "arrivals_last_6h":           6,
        "arrivals_last_24h":          20,
        "arrival_rate":               1.0,
        "queue_length":               1.0,
        "avg_waiting_time_hours":     0.5,
        "cranes_operational":         4,
        "crane_utilization":          0.5,
        "equipment_failure_count":    0,
        "labor_availability_pct":     0.95,
        "weather_severity":           0.1,
        "storm_flag":                 0,
        "hour":                       10,
        "day_of_week":                2,
        "is_weekend":                 0,
        "historical_avg_wait_time":   1.5,
        "historical_congestion_rate": 0.10,
        "traffic_pressure":           0.5,
        # capacity_pressure, queue_pressure, equipment_pressure,
        # weather_pressure, queue_capacity_interaction  — NOT submitted
        "traffic_weather_interaction": 0.05,
        "vessel_type":                "container",
        "cargo_type":                 "containerized",
    }


def _high_congestion_payload() -> Dict[str, Any]:
    """High queue-pressure snapshot — expected to yield HIGH risk."""
    return {
        "total_berths":               8,
        "available_berths":           0,
        "berth_utilization":          1.0,
        "vessels_currently_in_port":  30,
        "vessels_anchored":           18,
        "vessels_approaching":        8,
        "arrivals_last_1h":           8,
        "arrivals_last_6h":           30,
        "arrivals_last_24h":          90,
        "arrival_rate":               4.0,
        "queue_length":               22.0,
        "avg_waiting_time_hours":     12.0,
        "cranes_operational":         5,
        "crane_utilization":          0.9,
        "equipment_failure_count":    2,
        "labor_availability_pct":     0.80,
        "weather_severity":           0.4,
        "storm_flag":                 0,
        "hour":                       14,
        "day_of_week":                1,
        "is_weekend":                 0,
        "historical_avg_wait_time":   2.0,
        "historical_congestion_rate": 0.30,
        "traffic_pressure":           1.2,
        "traffic_weather_interaction": 0.48,
        "vessel_type":                "bulk",
        "cargo_type":                 "dry_bulk",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Health endpoint
# ─────────────────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        assert client.get("/health").status_code == 200

    def test_health_status_field(self, client):
        assert client.get("/health").json()["status"] in {"ok", "degraded"}

    def test_health_models_loaded_bool(self, client):
        assert isinstance(client.get("/health").json()["models_loaded"], bool)

    def test_health_shap_available_bool(self, client):
        assert isinstance(client.get("/health").json()["shap_available"], bool)

    def test_health_message_non_empty(self, client):
        msg = client.get("/health").json()["message"]
        assert isinstance(msg, str) and len(msg) > 0

    def test_health_models_loaded_true_with_artifacts(self, client):
        assert client.get("/health").json()["models_loaded"] is True

    def test_health_status_ok_with_artifacts(self, client):
        assert client.get("/health").json()["status"] == "ok"


# ─────────────────────────────────────────────────────────────────────────────
# Happy-path prediction
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictHappyPath:
    def test_predict_returns_200(self, client):
        assert client.post("/predict", json=_valid_payload()).status_code == 200

    def test_predict_response_is_json(self, client):
        r = client.post("/predict", json=_valid_payload())
        assert r.headers["content-type"].startswith("application/json")

    def test_predict_required_keys_present(self, client):
        d = client.post("/predict", json=_valid_payload()).json()
        for key in [
            "congestion_probability", "predicted_delay_hours", "risk_level",
            "risk_increasing_factors", "protective_factors",
            "actions", "reason", "escalation_triggers", "shap_available",
        ]:
            assert key in d, f"Missing key: {key}"


# ─────────────────────────────────────────────────────────────────────────────
# Response value ranges and types
# ─────────────────────────────────────────────────────────────────────────────

class TestResponseValues:
    def test_congestion_probability_in_range(self, client):
        p = client.post("/predict", json=_valid_payload()).json()["congestion_probability"]
        assert 0.0 <= p <= 1.0

    def test_predicted_delay_non_negative(self, client):
        assert client.post("/predict", json=_valid_payload()).json()["predicted_delay_hours"] >= 0.0

    def test_risk_level_valid(self, client):
        assert client.post("/predict", json=_valid_payload()).json()["risk_level"] in {"LOW", "MEDIUM", "HIGH"}

    def test_risk_increasing_factors_is_list(self, client):
        assert isinstance(client.post("/predict", json=_valid_payload()).json()["risk_increasing_factors"], list)

    def test_protective_factors_is_list(self, client):
        assert isinstance(client.post("/predict", json=_valid_payload()).json()["protective_factors"], list)

    def test_actions_is_non_empty_list(self, client):
        acts = client.post("/predict", json=_valid_payload()).json()["actions"]
        assert isinstance(acts, list) and len(acts) > 0

    def test_actions_are_strings(self, client):
        for a in client.post("/predict", json=_valid_payload()).json()["actions"]:
            assert isinstance(a, str)

    def test_reason_is_non_empty_string(self, client):
        r = client.post("/predict", json=_valid_payload()).json()["reason"]
        assert isinstance(r, str) and len(r) > 0

    def test_escalation_triggers_is_list(self, client):
        assert isinstance(client.post("/predict", json=_valid_payload()).json()["escalation_triggers"], list)

    def test_shap_available_is_bool(self, client):
        assert isinstance(client.post("/predict", json=_valid_payload()).json()["shap_available"], bool)


# ─────────────────────────────────────────────────────────────────────────────
# SHAP factor structure
# ─────────────────────────────────────────────────────────────────────────────

class TestShapFactorStructure:
    def _all_factors(self, client) -> list:
        d = client.post("/predict", json=_valid_payload()).json()
        return d["risk_increasing_factors"] + d["protective_factors"]

    def test_factor_required_keys(self, client):
        for f in self._all_factors(client):
            for key in ["source", "feature_name", "label", "raw_value", "direction"]:
                assert key in f, f"Factor missing key '{key}': {f}"

    def test_factor_source_values(self, client):
        for f in self._all_factors(client):
            assert f["source"] in {"congestion_shap", "delay_shap", "operational"}

    def test_factor_direction_values(self, client):
        valid = {"increases_risk", "decreases_risk", "increases_delay", "decreases_delay", "within_normal"}
        for f in self._all_factors(client):
            assert f["direction"] in valid

    def test_risk_increasing_have_correct_directions(self, client):
        d = client.post("/predict", json=_valid_payload()).json()
        for f in d["risk_increasing_factors"]:
            assert f["direction"] in {"increases_risk", "increases_delay"}

    def test_protective_have_correct_directions(self, client):
        d = client.post("/predict", json=_valid_payload()).json()
        for f in d["protective_factors"]:
            assert f["direction"] in {"decreases_risk", "decreases_delay", "within_normal"}

    def test_raw_value_numeric(self, client):
        for f in self._all_factors(client):
            assert isinstance(f["raw_value"], (int, float))

    def test_shap_value_numeric_or_null(self, client):
        for f in self._all_factors(client):
            assert f.get("shap_value") is None or isinstance(f["shap_value"], (int, float))

    def test_label_non_empty_string(self, client):
        for f in self._all_factors(client):
            assert isinstance(f["label"], str) and len(f["label"]) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Risk-level scenarios
# ─────────────────────────────────────────────────────────────────────────────

class TestRiskLevelScenarios:
    def test_high_congestion_produces_high_risk(self, client):
        d = client.post("/predict", json=_high_congestion_payload()).json()
        assert d["risk_level"] == "HIGH", (
            f"Expected HIGH, got {d['risk_level']} (prob={d['congestion_probability']:.3f})"
        )

    def test_high_congestion_probability_above_50pct(self, client):
        assert client.post("/predict", json=_high_congestion_payload()).json()["congestion_probability"] > 0.50

    def test_high_risk_actions_use_review_framing(self, client):
        d = client.post("/predict", json=_high_congestion_payload()).json()
        if d["risk_level"] == "HIGH":
            combined = " ".join(d["actions"]).lower()
            assert any(w in combined for w in ["review", "evaluate", "notify"])


# ─────────────────────────────────────────────────────────────────────────────
# Determinism
# ─────────────────────────────────────────────────────────────────────────────

class TestDeterminism:
    def test_identical_requests_identical_response(self, client):
        p = _valid_payload()
        r1 = client.post("/predict", json=p).json()
        r2 = client.post("/predict", json=p).json()
        assert r1["congestion_probability"] == r2["congestion_probability"]
        assert r1["predicted_delay_hours"]  == r2["predicted_delay_hours"]
        assert r1["risk_level"]             == r2["risk_level"]
        assert r1["actions"]                == r2["actions"]
        assert r1["reason"]                 == r2["reason"]

    def test_high_congestion_deterministic(self, client):
        p = _high_congestion_payload()
        r1 = client.post("/predict", json=p).json()
        r2 = client.post("/predict", json=p).json()
        assert r1["congestion_probability"] == r2["congestion_probability"]
        assert r1["risk_level"]             == r2["risk_level"]


# ─────────────────────────────────────────────────────────────────────────────
# Recommendation engine integration
# ─────────────────────────────────────────────────────────────────────────────

class TestRecommendationIntegration:
    def test_risk_level_consistent_with_probability(self, client):
        from src.recommendations.engine import DEFAULT_CONFIG
        d = client.post("/predict", json=_valid_payload()).json()
        p  = d["congestion_probability"]
        rl = d["risk_level"]
        if not d["escalation_triggers"]:
            if p < DEFAULT_CONFIG.prob_low_max:
                assert rl == "LOW"
            elif p > DEFAULT_CONFIG.prob_high_min:
                assert rl == "HIGH"
            else:
                assert rl == "MEDIUM"

    def test_high_queue_escalates_risk(self, client):
        """Very high queue_length → high derived queue_pressure → at least MEDIUM."""
        p = _valid_payload()
        p["queue_length"]           = 18.0
        p["avg_waiting_time_hours"] = 14.0
        d = client.post("/predict", json=p).json()
        assert d["risk_level"] in {"MEDIUM", "HIGH"}


# ─────────────────────────────────────────────────────────────────────────────
# Validation — missing required fields
# ─────────────────────────────────────────────────────────────────────────────

class TestMissingFields:
    @pytest.mark.parametrize("field", [
        "total_berths", "available_berths", "berth_utilization",
        "vessels_currently_in_port", "queue_length", "avg_waiting_time_hours",
        "traffic_pressure", "traffic_weather_interaction",
        "vessel_type", "cargo_type",
    ])
    def test_missing_required_field_returns_422(self, client, field):
        p = _valid_payload()
        del p[field]
        assert client.post("/predict", json=p).status_code == 422, (
            f"Expected 422 for missing '{field}'"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Validation — out-of-range and invalid values
# ─────────────────────────────────────────────────────────────────────────────

class TestInvalidValues:
    def test_negative_total_berths_rejected(self, client):
        p = _valid_payload(); p["total_berths"] = -1
        assert client.post("/predict", json=p).status_code == 422

    def test_negative_queue_length_rejected(self, client):
        p = _valid_payload(); p["queue_length"] = -0.1
        assert client.post("/predict", json=p).status_code == 422

    def test_negative_avg_waiting_time_rejected(self, client):
        p = _valid_payload(); p["avg_waiting_time_hours"] = -1.0
        assert client.post("/predict", json=p).status_code == 422

    def test_berth_utilization_above_one_rejected(self, client):
        p = _valid_payload(); p["berth_utilization"] = 1.5
        assert client.post("/predict", json=p).status_code == 422

    def test_berth_utilization_negative_rejected(self, client):
        p = _valid_payload(); p["berth_utilization"] = -0.1
        assert client.post("/predict", json=p).status_code == 422

    def test_hour_above_23_rejected(self, client):
        p = _valid_payload(); p["hour"] = 24
        assert client.post("/predict", json=p).status_code == 422

    def test_hour_negative_rejected(self, client):
        p = _valid_payload(); p["hour"] = -1
        assert client.post("/predict", json=p).status_code == 422

    def test_day_of_week_above_6_rejected(self, client):
        p = _valid_payload(); p["day_of_week"] = 7
        assert client.post("/predict", json=p).status_code == 422

    def test_storm_flag_value_2_rejected(self, client):
        p = _valid_payload(); p["storm_flag"] = 2
        assert client.post("/predict", json=p).status_code == 422

    def test_is_weekend_value_2_rejected(self, client):
        p = _valid_payload(); p["is_weekend"] = 2
        assert client.post("/predict", json=p).status_code == 422

    def test_labor_availability_above_one_rejected(self, client):
        p = _valid_payload(); p["labor_availability_pct"] = 1.01
        assert client.post("/predict", json=p).status_code == 422

    def test_negative_arrivals_rejected(self, client):
        p = _valid_payload(); p["arrivals_last_1h"] = -1
        assert client.post("/predict", json=p).status_code == 422

    def test_available_berths_exceeds_total_berths_rejected(self, client):
        p = _valid_payload()
        p["total_berths"] = 4; p["available_berths"] = 6
        assert client.post("/predict", json=p).status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# Categorical validation
# ─────────────────────────────────────────────────────────────────────────────

class TestCategoricalValidation:
    def test_unknown_vessel_type_rejected(self, client):
        p = _valid_payload(); p["vessel_type"] = "submarine"
        assert client.post("/predict", json=p).status_code == 422

    def test_unknown_cargo_type_rejected(self, client):
        p = _valid_payload(); p["cargo_type"] = "radioactive"
        assert client.post("/predict", json=p).status_code == 422

    def test_empty_string_vessel_type_rejected(self, client):
        p = _valid_payload(); p["vessel_type"] = ""
        assert client.post("/predict", json=p).status_code == 422

    @pytest.mark.parametrize("vt", ["bulk", "container", "general_cargo", "tanker"])
    def test_all_valid_vessel_types_accepted(self, client, vt):
        p = _valid_payload(); p["vessel_type"] = vt
        assert client.post("/predict", json=p).status_code == 200

    @pytest.mark.parametrize("ct", ["containerized", "dry_bulk", "general", "liquid_bulk"])
    def test_all_valid_cargo_types_accepted(self, client, ct):
        p = _valid_payload(); p["cargo_type"] = ct
        assert client.post("/predict", json=p).status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# Leakage prevention — target / metadata fields rejected
# ─────────────────────────────────────────────────────────────────────────────

class TestLeakagePrevention:
    @pytest.mark.parametrize("forbidden_field,value", [
        ("congestion_label",      1),
        ("future_delay_hours",    5.0),
        ("is_valid_training_row", True),
        ("port_id",               "PORT_01"),
        ("timestamp",             "2026-01-10T14:00:00"),
    ])
    def test_forbidden_field_returns_422(self, client, forbidden_field, value):
        p = _valid_payload()
        p[forbidden_field] = value
        assert client.post("/predict", json=p).status_code == 422, (
            f"Expected 422 for '{forbidden_field}', got non-422"
        )

    def test_future_delay_hours_not_in_schema(self):
        assert "future_delay_hours" not in PredictionRequest.model_fields

    def test_congestion_label_not_in_schema(self):
        assert "congestion_label" not in PredictionRequest.model_fields

    def test_is_valid_training_row_not_in_schema(self):
        assert "is_valid_training_row" not in PredictionRequest.model_fields


# ─────────────────────────────────────────────────────────────────────────────
# Derived-field override prevention — the 5 removed fields must be rejected
# ─────────────────────────────────────────────────────────────────────────────

class TestDerivedFieldOverridePrevention:
    """
    Clients must NOT be able to supply the five internally-derived features.
    Because extra='forbid', any attempt to include them must return 422.
    This guarantees that capacity_pressure, queue_pressure, equipment_pressure,
    weather_pressure, and queue_capacity_interaction are always computed from
    the raw inputs — they can never be overridden or spoofed.
    """

    @pytest.mark.parametrize("derived_field,value", [
        ("capacity_pressure",          0.9),
        ("queue_pressure",             2.5),
        ("equipment_pressure",         0.8),
        ("weather_pressure",           0.7),
        ("queue_capacity_interaction", 2.0),
    ])
    def test_derived_field_in_request_returns_422(self, client, derived_field, value):
        p = _valid_payload()
        p[derived_field] = value
        r = client.post("/predict", json=p)
        assert r.status_code == 422, (
            f"Expected 422 when client supplies derived field '{derived_field}', "
            f"got {r.status_code}"
        )

    def test_capacity_pressure_not_in_schema(self):
        assert "capacity_pressure" not in PredictionRequest.model_fields

    def test_queue_pressure_not_in_schema(self):
        assert "queue_pressure" not in PredictionRequest.model_fields

    def test_equipment_pressure_not_in_schema(self):
        assert "equipment_pressure" not in PredictionRequest.model_fields

    def test_weather_pressure_not_in_schema(self):
        assert "weather_pressure" not in PredictionRequest.model_fields

    def test_queue_capacity_interaction_not_in_schema(self):
        assert "queue_capacity_interaction" not in PredictionRequest.model_fields

    def test_contradictory_berth_fields_still_accepted(self, client):
        """
        Removing derived fields prevents impossible combinations.
        available_berths=30, berth_utilization=1.0 is still a client
        inconsistency but the schema only enforces available_berths ≤ total_berths.
        capacity_pressure is now derived from berth_utilization, so the client
        cannot supply capacity_pressure=0.0 to contradict berth_utilization=1.0.
        """
        p = _valid_payload()
        # Internally capacity_pressure = clip(1.0, 0, 1) = 1.0 regardless
        p["berth_utilization"] = 1.0
        p["available_berths"]  = 0   # consistent with utilization=1.0
        # This should succeed — no schema violation
        assert client.post("/predict", json=p).status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# Derived formula correctness (unit tests — no HTTP)
# ─────────────────────────────────────────────────────────────────────────────

class TestDerivedFormulaCorrectness:
    """
    Verify _derive_features() against the exact formulas in
    src/data/generator.py::_calculate_derived_features().
    """

    def _raw(
        self,
        total_berths=8,
        berth_utilization=0.375,
        queue_length=1.0,
        cranes_operational=4,
        weather_severity=0.1,
    ) -> dict:
        """Minimal raw dict for formula testing."""
        return {
            "total_berths":      total_berths,
            "berth_utilization": berth_utilization,
            "queue_length":      queue_length,
            "cranes_operational": cranes_operational,
            "weather_severity":  weather_severity,
        }

    def test_capacity_pressure_equals_berth_utilization(self):
        raw = self._raw(berth_utilization=0.60)
        d = _derive_features(raw)
        assert d["capacity_pressure"] == pytest.approx(0.60, abs=1e-12)

    def test_capacity_pressure_clips_at_zero(self):
        # berth_utilization is validated ≥ 0 by Pydantic, so 0 is the min
        raw = self._raw(berth_utilization=0.0)
        assert _derive_features(raw)["capacity_pressure"] == pytest.approx(0.0)

    def test_capacity_pressure_clips_at_one(self):
        raw = self._raw(berth_utilization=1.0)
        assert _derive_features(raw)["capacity_pressure"] == pytest.approx(1.0)

    def test_queue_pressure_formula(self):
        # queue_pressure = clip(queue_length / total_berths, 0, 3)
        raw = self._raw(total_berths=8, queue_length=4.0)
        assert _derive_features(raw)["queue_pressure"] == pytest.approx(0.5, abs=1e-12)

    def test_queue_pressure_clips_at_3(self):
        # queue_length=30, total_berths=8 → 3.75 → clipped to 3.0
        raw = self._raw(total_berths=8, queue_length=30.0)
        assert _derive_features(raw)["queue_pressure"] == pytest.approx(3.0)

    def test_queue_pressure_zero_queue(self):
        raw = self._raw(queue_length=0.0)
        assert _derive_features(raw)["queue_pressure"] == pytest.approx(0.0)

    def test_equipment_pressure_formula(self):
        # cranes_available = max(2, ceil(8 * 1.2)) = max(2, 10) = 10
        # equipment_pressure = clip(1 - 4/10, 0, 1) = 0.6
        raw = self._raw(total_berths=8, cranes_operational=4)
        assert _derive_features(raw)["equipment_pressure"] == pytest.approx(0.6, abs=1e-12)

    def test_equipment_pressure_all_cranes_operational(self):
        # total_berths=8 → cranes_available=10; cranes_operational=10 → pressure=0
        raw = self._raw(total_berths=8, cranes_operational=10)
        assert _derive_features(raw)["equipment_pressure"] == pytest.approx(0.0)

    def test_equipment_pressure_no_cranes(self):
        # cranes_operational=0 → pressure=1.0
        raw = self._raw(total_berths=8, cranes_operational=0)
        assert _derive_features(raw)["equipment_pressure"] == pytest.approx(1.0)

    def test_cranes_available_formula_matches_generator(self):
        """cranes_available = max(2, ceil(total_berths * 1.2)) — exact match."""
        for tb in [4, 5, 6, 7, 8, 9, 10, 11, 12]:
            expected_ca = max(2, math.ceil(tb * 1.2))
            raw = self._raw(total_berths=tb, cranes_operational=min(3, expected_ca))
            d = _derive_features(raw)
            # back-calculate cranes_available from equipment_pressure
            cranes_op = min(3, expected_ca)
            ep = d["equipment_pressure"]
            # ep = 1 - cranes_op / ca  →  ca = cranes_op / (1 - ep)
            if ep < 1.0:
                ca_backfilled = cranes_op / (1.0 - ep)
                assert abs(ca_backfilled - expected_ca) < 0.01, (
                    f"total_berths={tb}: expected cranes_available={expected_ca}, "
                    f"back-calculated={ca_backfilled:.2f}"
                )

    def test_weather_pressure_equals_weather_severity(self):
        raw = self._raw(weather_severity=0.42)
        assert _derive_features(raw)["weather_pressure"] == pytest.approx(0.42, abs=1e-12)

    def test_weather_pressure_zero(self):
        assert _derive_features(self._raw(weather_severity=0.0))["weather_pressure"] == 0.0

    def test_weather_pressure_one(self):
        assert _derive_features(self._raw(weather_severity=1.0))["weather_pressure"] == pytest.approx(1.0)

    def test_queue_capacity_interaction_formula(self):
        # queue_pressure * capacity_pressure
        raw = self._raw(total_berths=8, queue_length=4.0, berth_utilization=0.5)
        d = _derive_features(raw)
        expected = d["queue_pressure"] * d["capacity_pressure"]
        assert d["queue_capacity_interaction"] == pytest.approx(expected, abs=1e-12)

    def test_queue_capacity_interaction_zero_when_queue_zero(self):
        raw = self._raw(queue_length=0.0)
        assert _derive_features(raw)["queue_capacity_interaction"] == pytest.approx(0.0)

    def test_all_five_keys_present(self):
        d = _derive_features(self._raw())
        for key in [
            "capacity_pressure", "queue_pressure", "equipment_pressure",
            "weather_pressure", "queue_capacity_interaction",
        ]:
            assert key in d, f"Missing key: {key}"

    def test_against_real_training_data(self):
        """
        Spot-check _derive_features against real rows from the training CSV.
        Reconstruction error must be ≤ 1e-10 (floating-point only).
        """
        import sys, json
        sys.path.insert(0, ".")
        from src.config import load_dataset
        df = load_dataset()
        sample = df.sample(50, random_state=42)
        for _, row in sample.iterrows():
            raw = {
                "total_berths":      row["total_berths"],
                "berth_utilization": row["berth_utilization"],
                "queue_length":      row["queue_length"],
                "cranes_operational": row["cranes_operational"],
                "weather_severity":  row["weather_severity"],
            }
            derived = _derive_features(raw)
            assert abs(derived["capacity_pressure"]          - row["capacity_pressure"])  < 1e-10
            assert abs(derived["queue_pressure"]             - row["queue_pressure"])     < 1e-10
            assert abs(derived["equipment_pressure"]         - row["equipment_pressure"]) < 1e-10
            assert abs(derived["weather_pressure"]           - row["weather_pressure"])   < 1e-10
            assert abs(derived["queue_capacity_interaction"] - row["queue_capacity_interaction"]) < 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# Feature vector assembly — _build_feature_row (unit, no HTTP)
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildFeatureRow:
    """Verify _build_feature_row produces the correct 32-column dict."""

    @pytest.fixture(scope="class")
    def feature_names(self):
        import joblib
        from src.config import ARTIFACTS_MODELS
        return list(joblib.load(ARTIFACTS_MODELS / "scaler.joblib").feature_names_in_)

    def _raw_row(self) -> dict:
        req = PredictionRequest(**_valid_payload())
        return req.to_feature_row()

    def test_has_32_keys(self, feature_names):
        full = _build_feature_row(self._raw_row(), feature_names)
        assert len(full) == 32

    def test_column_order_matches_scaler(self, feature_names):
        full = _build_feature_row(self._raw_row(), feature_names)
        assert list(full.keys()) == feature_names

    def test_derived_slots_at_correct_positions(self, feature_names):
        """Positions 24–28 must be the five derived features."""
        full = _build_feature_row(self._raw_row(), feature_names)
        keys = list(full.keys())
        assert keys[24] == "capacity_pressure"
        assert keys[25] == "queue_pressure"
        assert keys[26] == "equipment_pressure"
        assert keys[27] == "weather_pressure"
        assert keys[28] == "queue_capacity_interaction"

    def test_raw_fields_preserved(self, feature_names):
        raw = self._raw_row()
        full = _build_feature_row(raw, feature_names)
        assert full["total_berths"]        == raw["total_berths"]
        assert full["queue_length"]        == raw["queue_length"]
        assert full["berth_utilization"]   == raw["berth_utilization"]
        assert full["traffic_pressure"]    == raw["traffic_pressure"]
        assert full["vessel_type"]         == raw["vessel_type"]

    def test_derived_values_correct(self, feature_names):
        raw = self._raw_row()
        full = _build_feature_row(raw, feature_names)
        expected = _derive_features(raw)
        for k, v in expected.items():
            assert full[k] == pytest.approx(v, abs=1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# PredictionRequest unit tests (no HTTP)
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictionRequestUnit:
    def test_to_feature_row_has_27_keys(self):
        """to_feature_row returns 27 raw fields (not 32 — derived are absent)."""
        req = PredictionRequest(**_valid_payload())
        assert len(req.to_feature_row()) == 27

    def test_to_feature_row_does_not_contain_derived_fields(self):
        req = PredictionRequest(**_valid_payload())
        row = req.to_feature_row()
        for derived in [
            "capacity_pressure", "queue_pressure", "equipment_pressure",
            "weather_pressure", "queue_capacity_interaction",
        ]:
            assert derived not in row, f"Derived field '{derived}' should not be in to_feature_row()"

    def test_to_feature_row_contains_traffic_fields(self):
        """traffic_pressure and traffic_weather_interaction remain as client inputs."""
        req = PredictionRequest(**_valid_payload())
        row = req.to_feature_row()
        assert "traffic_pressure" in row
        assert "traffic_weather_interaction" in row

    def test_vessel_type_encoding(self):
        for vt, expected in VESSEL_TYPE_ENCODING.items():
            p = _valid_payload(); p["vessel_type"] = vt
            assert PredictionRequest(**p).to_feature_row()["vessel_type"] == expected

    def test_cargo_type_encoding(self):
        for ct, expected in CARGO_TYPE_ENCODING.items():
            p = _valid_payload(); p["cargo_type"] = ct
            assert PredictionRequest(**p).to_feature_row()["cargo_type"] == expected

    def test_extra_field_raises_validation_error(self):
        from pydantic import ValidationError
        p = _valid_payload(); p["congestion_label"] = 1
        with pytest.raises(ValidationError):
            PredictionRequest(**p)

    def test_derived_field_raises_validation_error(self):
        from pydantic import ValidationError
        p = _valid_payload(); p["queue_pressure"] = 2.0
        with pytest.raises(ValidationError):
            PredictionRequest(**p)

    def test_available_berths_gt_total_berths_raises(self):
        from pydantic import ValidationError
        p = _valid_payload(); p["total_berths"] = 4; p["available_berths"] = 6
        with pytest.raises(ValidationError):
            PredictionRequest(**p)

    def test_negative_queue_length_raises(self):
        from pydantic import ValidationError
        p = _valid_payload(); p["queue_length"] = -1.0
        with pytest.raises(ValidationError):
            PredictionRequest(**p)

    def test_numerics_preserved(self):
        p = _valid_payload()
        row = PredictionRequest(**p).to_feature_row()
        assert row["total_berths"]      == p["total_berths"]
        assert row["queue_length"]      == p["queue_length"]
        assert row["berth_utilization"] == p["berth_utilization"]


# ─────────────────────────────────────────────────────────────────────────────
# Model prediction parity: derived vs. previously-supplied values
# ─────────────────────────────────────────────────────────────────────────────

class TestModelPredictionParity:
    """
    The model predictions must be the same whether the 5 derived fields are
    computed internally (new schema) or were supplied directly (old schema).

    We pick a real row from the dataset and verify that the API result matches
    what direct model inference with the exact generator-derived values gives.
    """

    def test_probability_matches_direct_inference(self, client):
        """
        Build the full 32-col feature row with internally-derived values,
        run the scaler + LR directly, and compare to the API response.
        """
        import joblib
        import sys
        sys.path.insert(0, ".")
        from src.config import load_dataset, ARTIFACTS_MODELS
        from src.data.preprocessing import create_temporal_split, prepare_features, apply_scaler

        df = load_dataset()
        train_df, test_df = create_temporal_split(df)
        X_train, X_test, _, _, feat_names = prepare_features(train_df, test_df)
        scaler = joblib.load(ARTIFACTS_MODELS / "scaler.joblib")
        lr_p   = joblib.load(ARTIFACTS_MODELS / "logistic_regression.joblib")
        lr     = lr_p["model"]

        # Use the 6th test row (non-congested)
        row = X_test.iloc[5]
        raw_df = test_df.iloc[5]

        # Direct inference
        X_scaled = scaler.transform(X_test.iloc[[5]])
        direct_prob = float(lr.predict_proba(X_scaled)[0, 1])

        # Build API payload from the raw feature values
        vessel_str = {0: "bulk", 1: "container", 2: "general_cargo", 3: "tanker"}
        cargo_str  = {0: "containerized", 1: "dry_bulk", 2: "general", 3: "liquid_bulk"}

        payload = {
            "total_berths":               int(row["total_berths"]),
            "available_berths":           int(row["available_berths"]),
            "berth_utilization":          round(float(row["berth_utilization"]), 6),
            "vessels_currently_in_port":  int(row["vessels_currently_in_port"]),
            "vessels_anchored":           int(row["vessels_anchored"]),
            "vessels_approaching":        int(row["vessels_approaching"]),
            "arrivals_last_1h":           int(row["arrivals_last_1h"]),
            "arrivals_last_6h":           int(row["arrivals_last_6h"]),
            "arrivals_last_24h":          int(row["arrivals_last_24h"]),
            "arrival_rate":               round(float(row["arrival_rate"]), 6),
            "queue_length":               round(float(row["queue_length"]), 6),
            "avg_waiting_time_hours":     round(float(row["avg_waiting_time_hours"]), 6),
            "cranes_operational":         int(row["cranes_operational"]),
            "crane_utilization":          round(float(row["crane_utilization"]), 6),
            "equipment_failure_count":    int(row["equipment_failure_count"]),
            "labor_availability_pct":     round(float(row["labor_availability_pct"]), 6),
            "weather_severity":           round(float(row["weather_severity"]), 6),
            "storm_flag":                 int(row["storm_flag"]),
            "hour":                       int(row["hour"]),
            "day_of_week":                int(row["day_of_week"]),
            "is_weekend":                 int(row["is_weekend"]),
            "historical_avg_wait_time":   round(float(row["historical_avg_wait_time"]), 6),
            "historical_congestion_rate": round(float(row["historical_congestion_rate"]), 6),
            "traffic_pressure":           round(float(row["traffic_pressure"]), 6),
            "traffic_weather_interaction": round(float(row["traffic_weather_interaction"]), 6),
            "vessel_type":                vessel_str[int(row["vessel_type"])],
            "cargo_type":                 cargo_str[int(row["cargo_type"])],
        }

        api_prob = client.post("/predict", json=payload).json()["congestion_probability"]

        # Allow a tiny floating-point tolerance from rounding in the payload
        assert abs(api_prob - direct_prob) < 0.001, (
            f"API probability {api_prob:.6f} differs from direct inference "
            f"{direct_prob:.6f} by {abs(api_prob - direct_prob):.6f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Encoding constants
# ─────────────────────────────────────────────────────────────────────────────

class TestEncodingConstants:
    def test_vessel_type_has_four_entries(self):
        assert len(VESSEL_TYPE_ENCODING) == 4

    def test_cargo_type_has_four_entries(self):
        assert len(CARGO_TYPE_ENCODING) == 4

    def test_vessel_type_values_0_to_3(self):
        assert sorted(VESSEL_TYPE_ENCODING.values()) == [0, 1, 2, 3]

    def test_cargo_type_values_0_to_3(self):
        assert sorted(CARGO_TYPE_ENCODING.values()) == [0, 1, 2, 3]

    def test_vessel_type_alphabetical(self):
        for i, k in enumerate(sorted(VESSEL_TYPE_ENCODING)):
            assert VESSEL_TYPE_ENCODING[k] == i

    def test_cargo_type_alphabetical(self):
        for i, k in enumerate(sorted(CARGO_TYPE_ENCODING)):
            assert CARGO_TYPE_ENCODING[k] == i
