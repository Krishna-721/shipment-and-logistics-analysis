"""
Format SHAP explanations for API and frontend consumption.

Converts raw FeatureContribution lists into structured dicts, concise
human-readable summaries, and JSON-serialisable payloads compatible with
the PortPulse API response schema.
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.explainability.shap_explainer import FeatureContribution


# ─────────────────────────────────────────────────────────────────────────────
# Human-readable labels
# ─────────────────────────────────────────────────────────────────────────────

_FEATURE_LABELS: Dict[str, str] = {
    "queue_length":               "Queue length (vessels)",
    "queue_pressure":             "Queue pressure (queue/berths)",
    "avg_waiting_time_hours":     "Average waiting time",
    "berth_utilization":          "Berth utilisation",
    "vessels_anchored":           "Vessels anchored",
    "vessels_currently_in_port":  "Vessels in port",
    "vessels_approaching":        "Vessels approaching",
    "arrivals_last_1h":           "Arrivals (last 1h)",
    "arrivals_last_6h":           "Arrivals (last 6h)",
    "arrivals_last_24h":          "Arrivals (last 24h)",
    "arrival_rate":               "Arrival rate (vessels/h)",
    "total_berths":               "Total berths",
    "available_berths":           "Available berths",
    "cranes_operational":         "Operational cranes",
    "crane_utilization":          "Crane utilisation",
    "equipment_failure_count":    "Equipment failures",
    "labor_availability_pct":     "Labour availability",
    "weather_severity":           "Weather severity",
    "storm_flag":                 "Storm flag",
    "hour":                       "Hour of day",
    "day_of_week":                "Day of week",
    "is_weekend":                 "Weekend flag",
    "historical_avg_wait_time":   "Historical avg wait time",
    "historical_congestion_rate": "Historical congestion rate",
    "traffic_pressure":           "Traffic pressure",
    "capacity_pressure":          "Capacity pressure",
    "equipment_pressure":         "Equipment pressure",
    "weather_pressure":           "Weather pressure",
    "queue_capacity_interaction": "Queue × capacity interaction",
    "traffic_weather_interaction": "Traffic × weather interaction",
    "vessel_type":                "Vessel type",
    "cargo_type":                 "Cargo type",
}


def human_label(feature_name: str) -> str:
    """Return a human-readable label for a feature name."""
    return _FEATURE_LABELS.get(feature_name, feature_name.replace("_", " ").title())


# ─────────────────────────────────────────────────────────────────────────────
# Formatters
# ─────────────────────────────────────────────────────────────────────────────

def format_contributions_for_api(
    contributions: List[FeatureContribution],
) -> List[Dict[str, Any]]:
    """
    Convert a FeatureContribution list to a JSON-serialisable list of dicts
    suitable for an API response.

    Each dict contains:
        feature_name    : internal name
        label           : human-readable label
        raw_value       : original feature value (unscaled)
        shap_value      : SHAP contribution
        direction       : "increases_risk" / "decreases_risk" etc.
        abs_shap        : |shap_value| — useful for sorting on the frontend
    """
    return [
        {
            "feature_name": c.feature_name,
            "label":        human_label(c.feature_name),
            "raw_value":    round(float(c.raw_value), 4),
            "shap_value":   round(float(c.shap_value), 6),
            "direction":    c.direction,
            "abs_shap":     round(abs(float(c.shap_value)), 6),
        }
        for c in contributions
    ]


def format_contributions_text(
    contributions: List[FeatureContribution],
    task: str = "classification",
) -> str:
    """
    Return a multi-line human-readable summary of top contributing features.

    Example output (classification)::

        Top contributing features:
          ↑ Queue pressure (queue/berths)   = 1.2300   (+0.8412, increases risk)
          ↑ Average waiting time            = 4.5000   (+0.5201, increases risk)
          ↓ Available berths                = 3.0000   (−0.3100, decreases risk)

    Parameters
    ----------
    contributions : sorted list of FeatureContribution (top-N already applied)
    task          : "classification" or "regression" — affects direction labels
    """
    lines = ["Top contributing features:"]
    for c in contributions:
        arrow = "↑" if c.shap_value > 0 else "↓"
        sign  = "+" if c.shap_value >= 0 else "−"
        label = human_label(c.feature_name)
        lines.append(
            f"  {arrow} {label:<40} = {c.raw_value:>10.4f}   "
            f"({sign}{abs(c.shap_value):.4f}, {c.direction})"
        )
    return "\n".join(lines)


def format_global_importance(
    importance_dict: Dict[str, float],
    top_n: int = 15,
) -> List[Dict[str, Any]]:
    """
    Convert a {feature: mean_abs_shap} dict to a ranked list of dicts
    for API / documentation use.

    Returns
    -------
    list of {"rank", "feature_name", "label", "mean_abs_shap"} dicts
    """
    ranked = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [
        {
            "rank":          i + 1,
            "feature_name":  feat,
            "label":         human_label(feat),
            "mean_abs_shap": round(val, 6),
        }
        for i, (feat, val) in enumerate(ranked)
    ]
