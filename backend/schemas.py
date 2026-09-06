"""
Pydantic request / response schemas for the PortPulse prediction API.

PredictionRequest  — the 27 raw prediction-time fields accepted from clients.
PredictionResponse — structured JSON output of the full prediction pipeline.
FactorItem         — a single SHAP-based or operational contributing factor.
HealthResponse     — /health endpoint payload.

Leakage prevention
------------------
PredictionRequest deliberately excludes every target and metadata column:
  - congestion_label       (training target)
  - future_delay_hours     (training target)
  - is_valid_training_row  (metadata)
  - port_id                (identifier)
  - timestamp              (identifier)

extra = "forbid" ensures that any attempt to submit these fields returns
a 422 Unprocessable Entity rather than silently ignoring them.

Derived feature policy (E-010 consistency fix)
-----------------------------------------------
Five model features are computed internally by PredictionService and must
NOT be supplied by API clients.  Accepting them as inputs would allow
contradictory combinations (e.g. available_berths=30 + berth_utilization=1.0).

Internally derived (formulas from generator._calculate_derived_features):
  capacity_pressure        = clip(berth_utilization, 0, 1)
  queue_pressure           = clip(queue_length / total_berths, 0, 3)
  equipment_pressure       = clip(1 - cranes_operational /
                                  max(2, ceil(total_berths * 1.2)), 0, 1)
  weather_pressure         = weather_severity
  queue_capacity_interaction = queue_pressure * capacity_pressure

Kept as client inputs (require daily_vessel_capacity, a hidden port constant):
  traffic_pressure         — client must supply
  traffic_weather_interaction — client must supply

Categorical encoding contract
------------------------------
vessel_type and cargo_type are accepted as human-readable strings and
converted to the same ordinal integers used during training:
  vessel_type : bulk=0  container=1  general_cargo=2  tanker=3
  cargo_type  : containerized=0  dry_bulk=1  general=2  liquid_bulk=3
The encoding matches _encode_categoricals() in src/data/preprocessing.py,
which sorts unique training values alphabetically and assigns 0-based ints.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# Categorical value sets  (must match training encoding)
# ─────────────────────────────────────────────────────────────────────────────

VesselType = Literal["bulk", "container", "general_cargo", "tanker"]
CargoType  = Literal["containerized", "dry_bulk", "general", "liquid_bulk"]

# Ordinal mappings — alphabetical sort of training-set unique values
VESSEL_TYPE_ENCODING: Dict[str, int] = {
    "bulk": 0, "container": 1, "general_cargo": 2, "tanker": 3
}
CARGO_TYPE_ENCODING: Dict[str, int] = {
    "containerized": 0, "dry_bulk": 1, "general": 2, "liquid_bulk": 3
}


# ─────────────────────────────────────────────────────────────────────────────
# Request schema
# ─────────────────────────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    """
    The 27 raw prediction-time fields accepted from API clients.

    Five model features are intentionally absent from this schema because
    they are derived internally by PredictionService using the exact formulas
    from the training generator.  Clients cannot supply — and cannot
    contradict — these values:

        capacity_pressure        = clip(berth_utilization, 0, 1)
        queue_pressure           = clip(queue_length / total_berths, 0, 3)
        equipment_pressure       = clip(1 - cranes_operational /
                                        max(2, ceil(total_berths * 1.2)), 0, 1)
        weather_pressure         = weather_severity
        queue_capacity_interaction = queue_pressure * capacity_pressure

    traffic_pressure and traffic_weather_interaction are still accepted as
    client inputs because they require daily_vessel_capacity, a hidden
    port-specific constant not available in the feature schema.

    Field bounds are conservative operational limits drawn from the training
    dataset.  Values outside them are rejected with 422.

    The schema does NOT include:
      congestion_label, future_delay_hours, is_valid_training_row,
      port_id, timestamp — submitting any of these returns 422.
    """

    model_config = ConfigDict(extra="forbid")

    # ── Capacity features ─────────────────────────────────────────────────────
    total_berths: int = Field(
        ..., ge=1, le=50,
        description="Total number of berths at the port."
    )
    available_berths: int = Field(
        ..., ge=0, le=50,
        description="Berths currently available (not occupied)."
    )
    berth_utilization: float = Field(
        ..., ge=0.0, le=1.0,
        description="Fraction of berths in use [0, 1]."
    )

    # ── Traffic / vessel counts ───────────────────────────────────────────────
    vessels_currently_in_port: int = Field(
        ..., ge=0, le=200,
        description="Vessels currently berthed or in port."
    )
    vessels_anchored: int = Field(
        ..., ge=0, le=200,
        description="Vessels waiting at anchor."
    )
    vessels_approaching: int = Field(
        ..., ge=0, le=100,
        description="Vessels en route and within reporting range."
    )

    # ── Arrival rates ─────────────────────────────────────────────────────────
    arrivals_last_1h: int = Field(
        ..., ge=0, le=50,
        description="Vessel arrivals in the last 1 hour."
    )
    arrivals_last_6h: int = Field(
        ..., ge=0, le=200,
        description="Vessel arrivals in the last 6 hours."
    )
    arrivals_last_24h: int = Field(
        ..., ge=0, le=500,
        description="Vessel arrivals in the last 24 hours."
    )
    arrival_rate: float = Field(
        ..., ge=0.0, le=50.0,
        description="Mean vessel arrival rate (vessels per hour)."
    )

    # ── Queue / waiting ───────────────────────────────────────────────────────
    queue_length: float = Field(
        ..., ge=0.0, le=200.0,
        description="Current queue length (vessels waiting for a berth)."
    )
    avg_waiting_time_hours: float = Field(
        ..., ge=0.0, le=100.0,
        description="Average vessel waiting time in hours."
    )

    # ── Equipment ─────────────────────────────────────────────────────────────
    cranes_operational: int = Field(
        ..., ge=0, le=50,
        description="Number of cranes currently operational."
    )
    crane_utilization: float = Field(
        ..., ge=0.0, le=1.0,
        description="Fraction of cranes in active use [0, 1]."
    )
    equipment_failure_count: int = Field(
        ..., ge=0, le=20,
        description="Number of equipment failures in the current shift."
    )

    # ── Labour ────────────────────────────────────────────────────────────────
    labor_availability_pct: float = Field(
        ..., ge=0.0, le=1.0,
        description="Fraction of required labour on shift [0, 1]."
    )

    # ── Weather ───────────────────────────────────────────────────────────────
    weather_severity: float = Field(
        ..., ge=0.0, le=1.0,
        description="Normalised weather severity index [0, 1]."
    )
    storm_flag: int = Field(
        ..., ge=0, le=1,
        description="1 if a storm warning is active, 0 otherwise."
    )

    # ── Temporal ─────────────────────────────────────────────────────────────
    hour: int = Field(
        ..., ge=0, le=23,
        description="Hour of day [0, 23]."
    )
    day_of_week: int = Field(
        ..., ge=0, le=6,
        description="Day of week [0=Monday, 6=Sunday]."
    )
    is_weekend: int = Field(
        ..., ge=0, le=1,
        description="1 if Saturday or Sunday, 0 otherwise."
    )

    # ── Historical ────────────────────────────────────────────────────────────
    historical_avg_wait_time: float = Field(
        ..., ge=0.0, le=100.0,
        description="Port historical average waiting time (hours)."
    )
    historical_congestion_rate: float = Field(
        ..., ge=0.0, le=1.0,
        description="Port historical congestion rate [0, 1]."
    )

    # ── Derived pressure / interaction features kept as client inputs ─────────
    # (require daily_vessel_capacity, a hidden port-specific constant)
    traffic_pressure: float = Field(
        ..., ge=0.0, le=10.0,
        description=(
            "Composite traffic pressure index "
            "(arrival_rate / daily_vessel_capacity * 24, clipped to [0, 2]). "
            "Must be supplied by the client; daily_vessel_capacity is a "
            "port-specific constant not available in the feature schema."
        )
    )
    traffic_weather_interaction: float = Field(
        ..., ge=0.0, le=10.0,
        description="Interaction term: traffic_pressure × weather_severity."
    )

    # ── Categoricals (string input, encoded internally) ───────────────────────
    vessel_type: VesselType = Field(
        ...,
        description="Dominant vessel type. One of: bulk, container, general_cargo, tanker."
    )
    cargo_type: CargoType = Field(
        ...,
        description="Dominant cargo type. One of: containerized, dry_bulk, general, liquid_bulk."
    )

    # ── Cross-field validation ────────────────────────────────────────────────

    @model_validator(mode="after")
    def _check_available_le_total(self) -> "PredictionRequest":
        if self.available_berths > self.total_berths:
            raise ValueError(
                f"available_berths ({self.available_berths}) cannot exceed "
                f"total_berths ({self.total_berths})"
            )
        return self

    def to_feature_row(self) -> Dict[str, Any]:
        """
        Return a dict of the 27 raw model features with categoricals encoded
        as integers.

        The five internally-derived features (capacity_pressure,
        queue_pressure, equipment_pressure, weather_pressure,
        queue_capacity_interaction) are NOT included here.
        PredictionService.predict() computes and inserts them at the correct
        scaler positions before building the 32-column feature DataFrame.

        traffic_pressure and traffic_weather_interaction remain here because
        they require the hidden daily_vessel_capacity constant and must be
        supplied by the client.
        """
        return {
            "total_berths":               self.total_berths,
            "available_berths":           self.available_berths,
            "berth_utilization":          self.berth_utilization,
            "vessels_currently_in_port":  self.vessels_currently_in_port,
            "vessels_anchored":           self.vessels_anchored,
            "vessels_approaching":        self.vessels_approaching,
            "arrivals_last_1h":           self.arrivals_last_1h,
            "arrivals_last_6h":           self.arrivals_last_6h,
            "arrivals_last_24h":          self.arrivals_last_24h,
            "arrival_rate":               self.arrival_rate,
            "queue_length":               self.queue_length,
            "avg_waiting_time_hours":     self.avg_waiting_time_hours,
            "cranes_operational":         self.cranes_operational,
            "crane_utilization":          self.crane_utilization,
            "equipment_failure_count":    self.equipment_failure_count,
            "labor_availability_pct":     self.labor_availability_pct,
            "weather_severity":           self.weather_severity,
            "storm_flag":                 self.storm_flag,
            "hour":                       self.hour,
            "day_of_week":                self.day_of_week,
            "is_weekend":                 self.is_weekend,
            "historical_avg_wait_time":   self.historical_avg_wait_time,
            "historical_congestion_rate": self.historical_congestion_rate,
            "traffic_pressure":           self.traffic_pressure,
            # slots 24-28 (capacity_pressure … queue_capacity_interaction)
            # are filled by PredictionService — not present in client input
            "traffic_weather_interaction": self.traffic_weather_interaction,
            # Categoricals encoded to integers
            "vessel_type":                VESSEL_TYPE_ENCODING[self.vessel_type],
            "cargo_type":                 CARGO_TYPE_ENCODING[self.cargo_type],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Response schemas
# ─────────────────────────────────────────────────────────────────────────────

class FactorItem(BaseModel):
    """A single contributing factor from SHAP or operational fallback."""
    source:       str            # "congestion_shap" | "delay_shap" | "operational"
    feature_name: str
    label:        str            # human-readable label
    raw_value:    float
    shap_value:   Optional[float] = None   # None for operational fallback
    direction:    str            # "increases_risk" | "decreases_risk" | etc.


class PredictionResponse(BaseModel):
    """Full structured output of the prediction pipeline."""
    congestion_probability:  float
    predicted_delay_hours:   float
    risk_level:              str    # LOW | MEDIUM | HIGH
    risk_increasing_factors: List[FactorItem]
    protective_factors:      List[FactorItem]
    actions:                 List[str]
    reason:                  str
    escalation_triggers:     List[str]
    shap_available:          bool   # True if SHAP ran successfully


class HealthResponse(BaseModel):
    """Response payload for GET /health."""
    status:          str   # "ok" | "degraded"
    models_loaded:   bool
    shap_available:  bool
    message:         str
