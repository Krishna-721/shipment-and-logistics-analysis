"""
Deterministic recommendation engine for Safiri PortPulse (E-009).

Consumes outputs from:
  - Logistic Regression congestion model  (congestion_probability float [0,1])
  - Ridge Regression delay model          (predicted_delay_hours  float >=0)
  - SHAP explainer (optional)             (List[FeatureContribution])

Produces a structured RecommendationResult containing:
  - risk_level             : LOW / MEDIUM / HIGH
  - congestion_probability : passed through
  - predicted_delay_hours  : passed through
  - risk_increasing_factors: SHAP contributions with direction "increases_risk"
                             or "increases_delay", or derived from raw features
  - protective_factors     : SHAP contributions with direction "decreases_risk"
                             or "decreases_delay"
  - actions                : list of recommended operational review actions
  - reason                 : plain-English explanation
  - escalation_triggers    : which secondary conditions caused escalation

Design principles
-----------------
* Entirely deterministic — same inputs always produce the same output.
* No ML, no LLM, no external calls.
* Human-in-the-loop: all HIGH-risk actions are framed as review or evaluate
  instructions, never as direct operational commands. A human operator makes
  the final decision on any diversion or contingency activation.
* SHAP direction-awareness: positive SHAP values (increases_risk / increases_delay)
  surface as risk_increasing_factors; negative values (decreases_risk /
  decreases_delay) surface as protective_factors. The two lists are never merged.
* If SHAP contributions are absent, key factors are derived from raw operational
  thresholds (queue_pressure, berth_utilization, delay).
* All thresholds are in RecommendationConfig — no magic numbers in logic.
* The engine never loads or calls any model or scaler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Configuration — all thresholds in one place
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RecommendationConfig:
    """
    All decision thresholds for the recommendation engine.

    Frozen so instances are safe to share across threads / test cases.
    Pass a custom instance to RecommendationEngine to override thresholds
    in tests or configuration files without modifying logic code.

    Probability thresholds (primary signal)
    ----------------------------------------
    prob_low_max  : congestion_probability  < this  → LOW risk
    prob_high_min : congestion_probability  > this  → HIGH risk
    (MEDIUM is the range in between: prob_low_max <= p <= prob_high_min)

    Escalation overrides (secondary signals — can only raise risk level, never lower)
    ----------------------------------------------------------------------------------
    delay_medium_min  : predicted_delay_hours >= this  → at least MEDIUM
    delay_high_min    : predicted_delay_hours >= this  → HIGH regardless of probability
    queue_medium_min  : queue_pressure        >= this  → at least MEDIUM
    berth_medium_min  : berth_utilization     >= this  → at least MEDIUM

    SHAP display limits
    --------------------
    shap_top_n : maximum risk-increasing factors to surface from SHAP
    shap_protective_n : maximum protective factors to surface from SHAP
    """
    # Primary probability thresholds
    prob_low_max:    float = 0.35
    prob_high_min:   float = 0.70

    # Secondary escalation thresholds
    delay_medium_min:  float = 6.0     # hours
    delay_high_min:    float = 12.0    # hours
    queue_medium_min:  float = 1.5     # queue_length / total_berths (can exceed 1.0)
    berth_medium_min:  float = 0.90    # fraction [0, 1]

    # SHAP display
    shap_top_n:        int   = 3
    shap_protective_n: int   = 2

    def __post_init__(self) -> None:
        if not (0.0 <= self.prob_low_max <= 1.0):
            raise ValueError(f"prob_low_max must be in [0,1], got {self.prob_low_max}")
        if not (0.0 <= self.prob_high_min <= 1.0):
            raise ValueError(f"prob_high_min must be in [0,1], got {self.prob_high_min}")
        if self.prob_low_max >= self.prob_high_min:
            raise ValueError(
                f"prob_low_max ({self.prob_low_max}) must be < prob_high_min ({self.prob_high_min})"
            )
        if self.delay_medium_min < 0 or self.delay_high_min < 0:
            raise ValueError("delay thresholds must be >= 0")
        if self.delay_medium_min >= self.delay_high_min:
            raise ValueError(
                f"delay_medium_min ({self.delay_medium_min}) must be < "
                f"delay_high_min ({self.delay_high_min})"
            )


# Singleton default — used when no config is supplied
DEFAULT_CONFIG = RecommendationConfig()

# Risk level constants
RISK_LOW    = "LOW"
RISK_MEDIUM = "MEDIUM"
RISK_HIGH   = "HIGH"

_RISK_ORDER: Dict[str, int] = {RISK_LOW: 0, RISK_MEDIUM: 1, RISK_HIGH: 2}


def _max_risk(a: str, b: str) -> str:
    """Return the higher of two risk level strings."""
    return a if _RISK_ORDER[a] >= _RISK_ORDER[b] else b


# ─────────────────────────────────────────────────────────────────────────────
# Input and output data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RecommendationInput:
    """
    All information needed to produce a recommendation.

    Required
    --------
    congestion_probability : float [0, 1]
        Positive-class probability from the Logistic Regression model.
    predicted_delay_hours : float >= 0
        Expected delay in hours from the Ridge Regression model.

    Optional operational context (used for escalation and fallback factors)
    -----------------------------------------------------------------------
    queue_pressure         : queue_length / total_berths (can exceed 1.0)
    berth_utilization      : fraction [0, 1]
    available_berths       : integer count
    avg_waiting_time_hours : float >= 0

    Optional SHAP contributions
    ---------------------------
    congestion_contributions : List of FeatureContribution objects or dicts.
        Sorted by |shap_value| descending (as returned by explain_instance).
        Entries with direction=="increases_risk" → risk_increasing_factors.
        Entries with direction=="decreases_risk" → protective_factors.
    delay_contributions      : Same structure for the delay model.
        direction=="increases_delay" → risk_increasing_factors.
        direction=="decreases_delay" → protective_factors.
    """
    # Core model outputs — required
    congestion_probability: float
    predicted_delay_hours:  float

    # Operational context — optional
    queue_pressure:           Optional[float] = None
    berth_utilization:        Optional[float] = None
    available_berths:         Optional[int]   = None
    avg_waiting_time_hours:   Optional[float] = None

    # SHAP contributions — optional
    congestion_contributions: Optional[List[Any]] = None
    delay_contributions:      Optional[List[Any]] = None

    def __post_init__(self) -> None:
        p = self.congestion_probability
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"congestion_probability must be in [0,1], got {p}")
        if self.predicted_delay_hours < 0:
            raise ValueError(
                f"predicted_delay_hours must be >= 0, got {self.predicted_delay_hours}"
            )


@dataclass
class RecommendationResult:
    """
    Structured output from the recommendation engine.

    Attributes
    ----------
    risk_level               : "LOW" | "MEDIUM" | "HIGH"
    congestion_probability   : float — from input
    predicted_delay_hours    : float — from input
    risk_increasing_factors  : features that push toward congestion/delay
    protective_factors       : features that reduce congestion/delay risk
    actions                  : recommended operational review steps (human-in-the-loop)
    reason                   : plain-English explanation string
    escalation_triggers      : secondary conditions that raised the risk level
                               (empty list when only probability drove the level)
    """
    risk_level:              str
    congestion_probability:  float
    predicted_delay_hours:   float
    risk_increasing_factors: List[Dict[str, Any]]
    protective_factors:      List[Dict[str, Any]]
    actions:                 List[str]
    reason:                  str
    escalation_triggers:     List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return {
            "risk_level":              self.risk_level,
            "congestion_probability":  round(self.congestion_probability, 4),
            "predicted_delay_hours":   round(self.predicted_delay_hours,  4),
            "risk_increasing_factors": self.risk_increasing_factors,
            "protective_factors":      self.protective_factors,
            "actions":                 self.actions,
            "reason":                  self.reason,
            "escalation_triggers":     self.escalation_triggers,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Action templates  (human-in-the-loop framing)
# ─────────────────────────────────────────────────────────────────────────────

_ACTIONS: Dict[str, List[str]] = {
    RISK_LOW: [
        "Continue monitoring normal operations.",
        "No immediate action required.",
    ],
    RISK_MEDIUM: [
        "Pre-position additional berth crew and equipment.",
        "Alert duty officer to elevated queue pressure.",
        "Monitor queue length and berth availability closely.",
        "Prepare contingency berth allocation plan for review.",
    ],
    RISK_HIGH: [
        "Notify duty officer and terminal manager immediately.",
        "Review current berth allocation and vessel sequencing.",
        "Evaluate alternate-port or anchorage options as part of operational review.",
        "Prepare and review contingency resource activation plan.",
        "Notify port authority and vessel traffic services for situational awareness.",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Helper: attribute access for dicts or dataclass-like objects
# ─────────────────────────────────────────────────────────────────────────────

def _get_attr(obj: Any, key: str, default: Any = None) -> Any:
    """Read an attribute from either a dict or an object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _contribution_to_factor(obj: Any, source: str) -> Dict[str, Any]:
    """Convert a FeatureContribution or dict to a factor dict."""
    feature_name = _get_attr(obj, "feature_name", "unknown")
    label        = _get_attr(obj, "label", None) or feature_name
    return {
        "source":       source,
        "feature_name": feature_name,
        "label":        label,
        "raw_value":    round(float(_get_attr(obj, "raw_value", 0.0) or 0.0), 4),
        "shap_value":   round(float(_get_attr(obj, "shap_value", 0.0) or 0.0), 4),
        "direction":    _get_attr(obj, "direction", ""),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Key-factor extraction  (direction-aware)
# ─────────────────────────────────────────────────────────────────────────────

_RISK_INCREASING_DIRS = {"increases_risk", "increases_delay"}
_PROTECTIVE_DIRS      = {"decreases_risk", "decreases_delay"}


def _extract_shap_factors(
    inp: RecommendationInput,
    config: RecommendationConfig,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extract direction-aware factors from SHAP contributions.

    Returns
    -------
    (risk_increasing, protective) — both may be empty lists.
    """
    risk_inc: List[Dict[str, Any]] = []
    protective: List[Dict[str, Any]] = []

    def _process(contributions: List[Any], source: str) -> None:
        for c in contributions:
            direction = _get_attr(c, "direction", "")
            if direction in _RISK_INCREASING_DIRS and len(risk_inc) < config.shap_top_n:
                risk_inc.append(_contribution_to_factor(c, source))
            elif direction in _PROTECTIVE_DIRS and len(protective) < config.shap_protective_n:
                protective.append(_contribution_to_factor(c, source))

    if inp.congestion_contributions:
        _process(inp.congestion_contributions, "congestion_shap")
    if inp.delay_contributions:
        _process(inp.delay_contributions, "delay_shap")

    return risk_inc, protective


def _derive_operational_factors(
    inp: RecommendationInput,
    config: RecommendationConfig,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Fallback when no SHAP contributions are available.

    Classifies raw operational values against their thresholds:
    - Values above the escalation threshold  → risk_increasing
    - Values in safe territory               → protective
    """
    risk_inc: List[Dict[str, Any]] = []
    protective: List[Dict[str, Any]] = []

    def _add(feature_name: str, label: str, raw_value: float,
             is_risk_increasing: bool) -> None:
        entry = {
            "source":       "operational",
            "feature_name": feature_name,
            "label":        label,
            "raw_value":    round(raw_value, 4),
            "shap_value":   None,
            "direction":    "increases_risk" if is_risk_increasing else "decreases_risk",
        }
        if is_risk_increasing:
            risk_inc.append(entry)
        else:
            protective.append(entry)

    if inp.queue_pressure is not None:
        _add("queue_pressure", "Queue pressure (queue/berths)",
             inp.queue_pressure,
             inp.queue_pressure >= config.queue_medium_min)

    if inp.berth_utilization is not None:
        _add("berth_utilization", "Berth utilisation",
             inp.berth_utilization,
             inp.berth_utilization >= config.berth_medium_min)

    if inp.available_berths is not None:
        # Low available berths is risk-increasing; plenty of berths is protective
        _add("available_berths", "Available berths",
             float(inp.available_berths),
             inp.available_berths == 0)

    if inp.avg_waiting_time_hours is not None:
        _add("avg_waiting_time_hours", "Average waiting time",
             inp.avg_waiting_time_hours,
             inp.avg_waiting_time_hours >= config.delay_medium_min)

    return risk_inc, protective


# ─────────────────────────────────────────────────────────────────────────────
# Reason builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_reason(
    risk_level: str,
    inp: RecommendationInput,
    escalation_triggers: List[str],
    config: RecommendationConfig,
) -> str:
    """
    Construct a plain-English reason string.
    Deterministic — identical inputs produce identical output.
    """
    p_pct = round(inp.congestion_probability * 100, 1)
    d     = round(inp.predicted_delay_hours, 1)
    parts: List[str] = []

    if risk_level == RISK_LOW:
        parts.append(
            f"Congestion probability is low ({p_pct:.1f} %) "
            f"with expected delay of {d:.1f} h."
        )
        parts.append("Normal port operations anticipated.")

    elif risk_level == RISK_MEDIUM:
        parts.append(
            f"Congestion probability is moderate ({p_pct:.1f} %) "
            f"with expected delay of {d:.1f} h."
        )
        if escalation_triggers:
            parts.append(
                "Escalated to MEDIUM due to: " + "; ".join(escalation_triggers) + "."
            )
        else:
            parts.append("Elevated congestion risk warrants preparation.")

    else:  # HIGH
        parts.append(
            f"Congestion probability is high ({p_pct:.1f} %) "
            f"with expected delay of {d:.1f} h."
        )
        if escalation_triggers:
            parts.append(
                "Escalated to HIGH due to: " + "; ".join(escalation_triggers) + "."
            )
        else:
            parts.append("Immediate operational review is required.")

    if inp.queue_pressure is not None:
        threshold_word = (
            "above escalation threshold"
            if inp.queue_pressure >= config.queue_medium_min
            else "within normal range"
        )
        parts.append(f"Queue pressure {inp.queue_pressure:.2f} ({threshold_word}).")

    if inp.available_berths is not None:
        parts.append(f"Available berths: {inp.available_berths}.")

    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Core engine
# ─────────────────────────────────────────────────────────────────────────────

class RecommendationEngine:
    """
    Deterministic, rule-based recommendation engine.

    The engine is stateless — it holds only configuration. Call
    ``recommend(inp)`` to get a ``RecommendationResult`` for a single
    prediction event.

    Parameters
    ----------
    config : RecommendationConfig, optional
        Threshold configuration. Defaults to DEFAULT_CONFIG.
    """

    def __init__(self, config: Optional[RecommendationConfig] = None) -> None:
        self.config: RecommendationConfig = config or DEFAULT_CONFIG

    # ── Primary entry point ───────────────────────────────────────────────────

    def recommend(self, inp: RecommendationInput) -> RecommendationResult:
        """
        Produce a recommendation for a single prediction event.

        Parameters
        ----------
        inp : RecommendationInput
            Pre-computed model outputs and optional operational context.

        Returns
        -------
        RecommendationResult
        """
        risk_level, escalation_triggers = self._classify_risk(inp)
        risk_inc, protective            = self._build_factors(inp)
        actions                         = list(_ACTIONS[risk_level])
        reason                          = _build_reason(
                                            risk_level, inp,
                                            escalation_triggers, self.config
                                          )

        return RecommendationResult(
            risk_level              = risk_level,
            congestion_probability  = inp.congestion_probability,
            predicted_delay_hours   = inp.predicted_delay_hours,
            risk_increasing_factors = risk_inc,
            protective_factors      = protective,
            actions                 = actions,
            reason                  = reason,
            escalation_triggers     = escalation_triggers,
        )

    # ── Risk classification ───────────────────────────────────────────────────

    def _classify_risk(
        self, inp: RecommendationInput
    ) -> tuple[str, List[str]]:
        """
        Determine risk level and collect escalation triggers.

        Steps:
        1. Assign base level from congestion probability.
        2. Apply secondary escalation rules — can only raise, never lower.
        3. Return (final_risk_level, list_of_trigger_descriptions).
        """
        cfg = self.config
        p   = inp.congestion_probability
        d   = inp.predicted_delay_hours

        # Step 1 — probability-based base level
        if p < cfg.prob_low_max:
            base = RISK_LOW
        elif p > cfg.prob_high_min:
            base = RISK_HIGH
        else:
            base = RISK_MEDIUM

        risk = base
        triggers: List[str] = []

        # Step 2 — escalation rules
        # Delay escalation to HIGH (checked before MEDIUM so it can skip straight to HIGH)
        if d >= cfg.delay_high_min:
            if _RISK_ORDER[risk] < _RISK_ORDER[RISK_HIGH]:
                risk = RISK_HIGH
                triggers.append(
                    f"expected delay {d:.1f} h >= {cfg.delay_high_min:.1f} h threshold"
                )

        # Delay escalation to MEDIUM
        elif d >= cfg.delay_medium_min:
            if _RISK_ORDER[risk] < _RISK_ORDER[RISK_MEDIUM]:
                risk = RISK_MEDIUM
                triggers.append(
                    f"expected delay {d:.1f} h >= {cfg.delay_medium_min:.1f} h threshold"
                )

        # Queue pressure escalation to MEDIUM
        if (
            inp.queue_pressure is not None
            and inp.queue_pressure >= cfg.queue_medium_min
            and _RISK_ORDER[risk] < _RISK_ORDER[RISK_MEDIUM]
        ):
            risk = RISK_MEDIUM
            triggers.append(
                f"queue pressure {inp.queue_pressure:.2f} >= {cfg.queue_medium_min:.2f} threshold"
            )

        # Berth utilisation escalation to MEDIUM
        if (
            inp.berth_utilization is not None
            and inp.berth_utilization >= cfg.berth_medium_min
            and _RISK_ORDER[risk] < _RISK_ORDER[RISK_MEDIUM]
        ):
            risk = RISK_MEDIUM
            triggers.append(
                f"berth utilisation {inp.berth_utilization:.2f} >= {cfg.berth_medium_min:.2f} threshold"
            )

        return risk, triggers

    # ── Factor extraction ─────────────────────────────────────────────────────

    def _build_factors(
        self, inp: RecommendationInput
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Return (risk_increasing_factors, protective_factors).

        Prefers SHAP-based factors when available; falls back to operational
        threshold-derived factors when SHAP is absent.
        """
        has_shap = bool(inp.congestion_contributions or inp.delay_contributions)

        if has_shap:
            return _extract_shap_factors(inp, self.config)
        return _derive_operational_factors(inp, self.config)

    # ── Convenience ───────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"RecommendationEngine("
            f"prob_thresholds=[<{cfg.prob_low_max}, >{cfg.prob_high_min}], "
            f"delay_thresholds=[>={cfg.delay_medium_min}h, >={cfg.delay_high_min}h])"
        )
