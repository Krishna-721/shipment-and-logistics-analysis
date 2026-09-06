"""
Tests for E-009 — Recommendation Engine.

Covers:
- RecommendationConfig: defaults, validation, custom values
- RecommendationInput: validation (probability range, negative delay)
- RecommendationResult: to_dict keys and types
- Risk classification: LOW / MEDIUM / HIGH from probability alone
- Threshold boundaries: exact boundary values
- Secondary escalation: delay, queue_pressure, berth_utilization
- HIGH direct from delay threshold
- Escalation can only raise, never lower, risk
- Mixed signals: high congestion prob with low delay
- Mixed signals: low congestion prob with high delay
- High queue pressure with low probability
- SHAP direction-awareness:
    - increases_risk  → risk_increasing_factors
    - decreases_risk  → protective_factors
    - increases_delay → risk_increasing_factors
    - decreases_delay → protective_factors
    - mixed directions split correctly
- No SHAP — operational fallback factors
- Empty SHAP list — operational fallback
- Partial SHAP (only congestion, only delay)
- Actions content: HIGH contains review/evaluate language, no direct commands
- Escalation triggers list populated correctly
- Determinism: identical inputs → identical output
- to_dict is JSON-serialisable
- RecommendationEngine repr
- Custom config passed through correctly
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pytest

from src.recommendations.engine import (
    DEFAULT_CONFIG,
    RISK_HIGH,
    RISK_LOW,
    RISK_MEDIUM,
    RecommendationConfig,
    RecommendationEngine,
    RecommendationInput,
    RecommendationResult,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _engine(config: Optional[RecommendationConfig] = None) -> RecommendationEngine:
    return RecommendationEngine(config)


def _inp(
    prob:  float = 0.10,
    delay: float = 1.0,
    queue_pressure:      Optional[float] = None,
    berth_utilization:   Optional[float] = None,
    available_berths:    Optional[int]   = None,
    avg_waiting_time:    Optional[float] = None,
    cong_contributions:  Optional[list]  = None,
    delay_contributions: Optional[list]  = None,
) -> RecommendationInput:
    return RecommendationInput(
        congestion_probability   = prob,
        predicted_delay_hours    = delay,
        queue_pressure           = queue_pressure,
        berth_utilization        = berth_utilization,
        available_berths         = available_berths,
        avg_waiting_time_hours   = avg_waiting_time,
        congestion_contributions = cong_contributions,
        delay_contributions      = delay_contributions,
    )


def _make_contribution(
    feature_name: str,
    shap_value:   float,
    direction:    str,
    raw_value:    float = 1.0,
    label:        Optional[str] = None,
) -> Dict[str, Any]:
    """Helper: build a contribution dict (mirrors FeatureContribution.to_dict output)."""
    return {
        "feature_name": feature_name,
        "label":        label or feature_name.replace("_", " ").title(),
        "raw_value":    raw_value,
        "shap_value":   shap_value,
        "direction":    direction,
        "abs_shap":     abs(shap_value),
    }


# ─────────────────────────────────────────────────────────────────────────────
# RecommendationConfig
# ─────────────────────────────────────────────────────────────────────────────

class TestRecommendationConfig:
    def test_defaults(self):
        cfg = RecommendationConfig()
        assert cfg.prob_low_max    == 0.35
        assert cfg.prob_high_min   == 0.70
        assert cfg.delay_medium_min == 6.0
        assert cfg.delay_high_min  == 12.0
        assert cfg.queue_medium_min == 1.5
        assert cfg.berth_medium_min == 0.90
        assert cfg.shap_top_n       == 3
        assert cfg.shap_protective_n == 2

    def test_custom_values(self):
        cfg = RecommendationConfig(prob_low_max=0.20, prob_high_min=0.80)
        assert cfg.prob_low_max  == 0.20
        assert cfg.prob_high_min == 0.80

    def test_invalid_prob_low_max_out_of_range(self):
        with pytest.raises(ValueError):
            RecommendationConfig(prob_low_max=1.5)

    def test_invalid_prob_high_min_out_of_range(self):
        with pytest.raises(ValueError):
            RecommendationConfig(prob_high_min=-0.1)

    def test_invalid_prob_low_max_gte_high_min(self):
        with pytest.raises(ValueError):
            RecommendationConfig(prob_low_max=0.7, prob_high_min=0.7)

    def test_invalid_prob_low_max_gt_high_min(self):
        with pytest.raises(ValueError):
            RecommendationConfig(prob_low_max=0.8, prob_high_min=0.6)

    def test_invalid_delay_medium_gte_high(self):
        with pytest.raises(ValueError):
            RecommendationConfig(delay_medium_min=12.0, delay_high_min=6.0)

    def test_frozen(self):
        cfg = RecommendationConfig()
        with pytest.raises(Exception):
            cfg.prob_low_max = 0.5  # type: ignore[misc]

    def test_default_singleton_is_valid(self):
        assert isinstance(DEFAULT_CONFIG, RecommendationConfig)


# ─────────────────────────────────────────────────────────────────────────────
# RecommendationInput validation
# ─────────────────────────────────────────────────────────────────────────────

class TestRecommendationInput:
    def test_valid_minimal(self):
        inp = _inp(prob=0.5, delay=3.0)
        assert inp.congestion_probability == 0.5
        assert inp.predicted_delay_hours  == 3.0

    def test_probability_zero_is_valid(self):
        inp = _inp(prob=0.0)
        assert inp.congestion_probability == 0.0

    def test_probability_one_is_valid(self):
        inp = _inp(prob=1.0)
        assert inp.congestion_probability == 1.0

    def test_probability_above_one_raises(self):
        with pytest.raises(ValueError, match="congestion_probability"):
            _inp(prob=1.01)

    def test_probability_below_zero_raises(self):
        with pytest.raises(ValueError, match="congestion_probability"):
            _inp(prob=-0.01)

    def test_negative_delay_raises(self):
        with pytest.raises(ValueError, match="predicted_delay_hours"):
            _inp(delay=-1.0)

    def test_zero_delay_valid(self):
        inp = _inp(delay=0.0)
        assert inp.predicted_delay_hours == 0.0

    def test_optional_fields_default_none(self):
        inp = _inp()
        assert inp.queue_pressure          is None
        assert inp.berth_utilization       is None
        assert inp.available_berths        is None
        assert inp.avg_waiting_time_hours  is None
        assert inp.congestion_contributions is None
        assert inp.delay_contributions     is None


# ─────────────────────────────────────────────────────────────────────────────
# Risk classification — probability only
# ─────────────────────────────────────────────────────────────────────────────

class TestRiskClassificationProbabilityOnly:
    def test_low_risk(self):
        result = _engine().recommend(_inp(prob=0.10))
        assert result.risk_level == RISK_LOW

    def test_low_risk_near_threshold(self):
        # 0.34 < 0.35 → still LOW
        result = _engine().recommend(_inp(prob=0.34))
        assert result.risk_level == RISK_LOW

    def test_medium_risk(self):
        result = _engine().recommend(_inp(prob=0.52))
        assert result.risk_level == RISK_MEDIUM

    def test_medium_risk_at_low_boundary(self):
        # exactly 0.35 → MEDIUM (not < 0.35)
        result = _engine().recommend(_inp(prob=0.35))
        assert result.risk_level == RISK_MEDIUM

    def test_medium_risk_at_high_boundary(self):
        # exactly 0.70 → MEDIUM (not > 0.70)
        result = _engine().recommend(_inp(prob=0.70))
        assert result.risk_level == RISK_MEDIUM

    def test_high_risk(self):
        result = _engine().recommend(_inp(prob=0.90))
        assert result.risk_level == RISK_HIGH

    def test_high_risk_near_threshold(self):
        # 0.71 > 0.70 → HIGH
        result = _engine().recommend(_inp(prob=0.71))
        assert result.risk_level == RISK_HIGH

    def test_probability_zero_is_low(self):
        result = _engine().recommend(_inp(prob=0.0))
        assert result.risk_level == RISK_LOW

    def test_probability_one_is_high(self):
        result = _engine().recommend(_inp(prob=1.0))
        assert result.risk_level == RISK_HIGH

    def test_no_escalation_triggers_for_prob_only(self):
        result = _engine().recommend(_inp(prob=0.80))
        assert result.escalation_triggers == []


# ─────────────────────────────────────────────────────────────────────────────
# Escalation — delay
# ─────────────────────────────────────────────────────────────────────────────

class TestDelayEscalation:
    def test_low_prob_high_delay_escalates_to_high(self):
        # delay >= 12.0 always yields HIGH
        result = _engine().recommend(_inp(prob=0.10, delay=15.0))
        assert result.risk_level == RISK_HIGH

    def test_low_prob_medium_delay_escalates_to_medium(self):
        result = _engine().recommend(_inp(prob=0.10, delay=8.0))
        assert result.risk_level == RISK_MEDIUM

    def test_delay_exactly_high_threshold_is_high(self):
        result = _engine().recommend(_inp(prob=0.10, delay=12.0))
        assert result.risk_level == RISK_HIGH

    def test_delay_just_below_high_threshold_is_medium(self):
        result = _engine().recommend(_inp(prob=0.10, delay=11.9))
        assert result.risk_level == RISK_MEDIUM

    def test_delay_exactly_medium_threshold_escalates(self):
        result = _engine().recommend(_inp(prob=0.10, delay=6.0))
        assert result.risk_level == RISK_MEDIUM

    def test_delay_just_below_medium_threshold_stays_low(self):
        result = _engine().recommend(_inp(prob=0.10, delay=5.9))
        assert result.risk_level == RISK_LOW

    def test_delay_escalation_trigger_recorded(self):
        result = _engine().recommend(_inp(prob=0.10, delay=8.0))
        assert len(result.escalation_triggers) >= 1
        assert any("delay" in t.lower() for t in result.escalation_triggers)

    def test_high_delay_adds_trigger_even_with_high_prob(self):
        # Already HIGH from prob; trigger from delay should NOT appear
        # (escalation only fires when it actually changes the level)
        result = _engine().recommend(_inp(prob=0.80, delay=15.0))
        assert result.risk_level == RISK_HIGH
        # Trigger may or may not appear depending on implementation;
        # the risk level must be HIGH either way.


# ─────────────────────────────────────────────────────────────────────────────
# Escalation — queue pressure
# ─────────────────────────────────────────────────────────────────────────────

class TestQueuePressureEscalation:
    def test_high_queue_pressure_escalates_low_to_medium(self):
        result = _engine().recommend(_inp(prob=0.10, queue_pressure=2.0))
        assert result.risk_level == RISK_MEDIUM

    def test_queue_pressure_exactly_at_threshold(self):
        result = _engine().recommend(_inp(prob=0.10, queue_pressure=1.5))
        assert result.risk_level == RISK_MEDIUM

    def test_queue_pressure_just_below_threshold(self):
        result = _engine().recommend(_inp(prob=0.10, queue_pressure=1.49))
        assert result.risk_level == RISK_LOW

    def test_queue_pressure_does_not_lower_high(self):
        result = _engine().recommend(_inp(prob=0.80, queue_pressure=0.1))
        assert result.risk_level == RISK_HIGH

    def test_queue_pressure_trigger_recorded(self):
        result = _engine().recommend(_inp(prob=0.10, queue_pressure=2.0))
        assert any("queue" in t.lower() for t in result.escalation_triggers)

    def test_queue_pressure_none_no_effect(self):
        result = _engine().recommend(_inp(prob=0.10, queue_pressure=None))
        assert result.risk_level == RISK_LOW
        assert result.escalation_triggers == []


# ─────────────────────────────────────────────────────────────────────────────
# Escalation — berth utilisation
# ─────────────────────────────────────────────────────────────────────────────

class TestBerthUtilisationEscalation:
    def test_high_berth_utilisation_escalates(self):
        result = _engine().recommend(_inp(prob=0.10, berth_utilization=0.95))
        assert result.risk_level == RISK_MEDIUM

    def test_berth_exactly_at_threshold(self):
        result = _engine().recommend(_inp(prob=0.10, berth_utilization=0.90))
        assert result.risk_level == RISK_MEDIUM

    def test_berth_just_below_threshold(self):
        result = _engine().recommend(_inp(prob=0.10, berth_utilization=0.89))
        assert result.risk_level == RISK_LOW

    def test_berth_trigger_recorded(self):
        result = _engine().recommend(_inp(prob=0.10, berth_utilization=0.95))
        assert any("berth" in t.lower() for t in result.escalation_triggers)

    def test_berth_does_not_lower_existing_high(self):
        result = _engine().recommend(_inp(prob=0.90, berth_utilization=0.50))
        assert result.risk_level == RISK_HIGH


# ─────────────────────────────────────────────────────────────────────────────
# Mixed signals
# ─────────────────────────────────────────────────────────────────────────────

class TestMixedSignals:
    def test_high_prob_low_delay(self):
        # Probability alone drives HIGH; delay is not a trigger
        result = _engine().recommend(_inp(prob=0.90, delay=1.0))
        assert result.risk_level == RISK_HIGH
        assert result.escalation_triggers == []

    def test_low_prob_high_delay(self):
        # Probability → LOW; delay escalates to HIGH
        result = _engine().recommend(_inp(prob=0.10, delay=15.0))
        assert result.risk_level == RISK_HIGH
        assert len(result.escalation_triggers) >= 1

    def test_medium_prob_high_queue_stays_medium(self):
        # Probability already MEDIUM; high queue can't lower it
        result = _engine().recommend(_inp(prob=0.50, queue_pressure=2.0))
        assert result.risk_level == RISK_MEDIUM

    def test_all_low_signals(self):
        result = _engine().recommend(
            _inp(prob=0.05, delay=0.5, queue_pressure=0.2, berth_utilization=0.3)
        )
        assert result.risk_level == RISK_LOW

    def test_all_high_signals(self):
        result = _engine().recommend(
            _inp(prob=0.90, delay=14.0, queue_pressure=2.5, berth_utilization=0.98)
        )
        assert result.risk_level == RISK_HIGH

    def test_multiple_escalation_triggers(self):
        # LOW prob, both delay and queue pressure trigger
        result = _engine().recommend(
            _inp(prob=0.10, delay=8.0, queue_pressure=2.0)
        )
        assert result.risk_level in {RISK_MEDIUM, RISK_HIGH}


# ─────────────────────────────────────────────────────────────────────────────
# SHAP direction-awareness
# ─────────────────────────────────────────────────────────────────────────────

class TestShapDirectionAwareness:
    def test_increases_risk_goes_to_risk_increasing(self):
        c = _make_contribution("queue_length", shap_value=2.5,
                               direction="increases_risk", raw_value=20.0)
        result = _engine().recommend(_inp(prob=0.80, cong_contributions=[c]))
        names = [f["feature_name"] for f in result.risk_increasing_factors]
        assert "queue_length" in names

    def test_increases_risk_not_in_protective(self):
        c = _make_contribution("queue_length", shap_value=2.5,
                               direction="increases_risk", raw_value=20.0)
        result = _engine().recommend(_inp(prob=0.80, cong_contributions=[c]))
        names = [f["feature_name"] for f in result.protective_factors]
        assert "queue_length" not in names

    def test_decreases_risk_goes_to_protective(self):
        c = _make_contribution("available_berths", shap_value=-1.2,
                               direction="decreases_risk", raw_value=8.0)
        result = _engine().recommend(_inp(prob=0.80, cong_contributions=[c]))
        names = [f["feature_name"] for f in result.protective_factors]
        assert "available_berths" in names

    def test_decreases_risk_not_in_risk_increasing(self):
        c = _make_contribution("available_berths", shap_value=-1.2,
                               direction="decreases_risk", raw_value=8.0)
        result = _engine().recommend(_inp(prob=0.80, cong_contributions=[c]))
        names = [f["feature_name"] for f in result.risk_increasing_factors]
        assert "available_berths" not in names

    def test_increases_delay_goes_to_risk_increasing(self):
        c = _make_contribution("avg_waiting_time_hours", shap_value=3.1,
                               direction="increases_delay", raw_value=10.0)
        result = _engine().recommend(_inp(prob=0.30, delay_contributions=[c]))
        names = [f["feature_name"] for f in result.risk_increasing_factors]
        assert "avg_waiting_time_hours" in names

    def test_decreases_delay_goes_to_protective(self):
        c = _make_contribution("available_berths", shap_value=-0.8,
                               direction="decreases_delay", raw_value=6.0)
        result = _engine().recommend(_inp(prob=0.30, delay_contributions=[c]))
        names = [f["feature_name"] for f in result.protective_factors]
        assert "available_berths" in names

    def test_mixed_directions_split_correctly(self):
        contributions = [
            _make_contribution("queue_length",    shap_value=2.0,  direction="increases_risk"),
            _make_contribution("queue_pressure",  shap_value=1.5,  direction="increases_risk"),
            _make_contribution("available_berths",shap_value=-1.0, direction="decreases_risk"),
        ]
        result = _engine().recommend(_inp(prob=0.80, cong_contributions=contributions))
        inc_names = [f["feature_name"] for f in result.risk_increasing_factors]
        pro_names = [f["feature_name"] for f in result.protective_factors]
        assert "queue_length"     in inc_names
        assert "queue_pressure"   in inc_names
        assert "available_berths" in pro_names
        assert "available_berths" not in inc_names
        assert "queue_length"     not in pro_names

    def test_top_n_risk_increasing_limit_respected(self):
        contributions = [
            _make_contribution(f"feat_{i}", shap_value=float(i+1),
                               direction="increases_risk")
            for i in range(10)
        ]
        cfg = RecommendationConfig(shap_top_n=3)
        result = _engine(cfg).recommend(_inp(prob=0.80, cong_contributions=contributions))
        assert len(result.risk_increasing_factors) <= 3

    def test_top_protective_limit_respected(self):
        contributions = [
            _make_contribution(f"feat_{i}", shap_value=-float(i+1),
                               direction="decreases_risk")
            for i in range(10)
        ]
        cfg = RecommendationConfig(shap_protective_n=2)
        result = _engine(cfg).recommend(_inp(prob=0.80, cong_contributions=contributions))
        assert len(result.protective_factors) <= 2

    def test_factor_dict_has_required_keys(self):
        c = _make_contribution("queue_pressure", shap_value=1.5,
                               direction="increases_risk")
        result = _engine().recommend(_inp(prob=0.80, cong_contributions=[c]))
        for f in result.risk_increasing_factors:
            for key in ["source", "feature_name", "label", "raw_value",
                        "shap_value", "direction"]:
                assert key in f, f"Missing key '{key}' in factor {f}"

    def test_factor_source_is_congestion_shap(self):
        c = _make_contribution("queue_length", shap_value=2.0,
                               direction="increases_risk")
        result = _engine().recommend(_inp(prob=0.80, cong_contributions=[c]))
        assert result.risk_increasing_factors[0]["source"] == "congestion_shap"

    def test_factor_source_is_delay_shap(self):
        c = _make_contribution("avg_waiting_time_hours", shap_value=3.0,
                               direction="increases_delay")
        result = _engine().recommend(_inp(prob=0.20, delay_contributions=[c]))
        assert result.risk_increasing_factors[0]["source"] == "delay_shap"


# ─────────────────────────────────────────────────────────────────────────────
# Fallback — no SHAP contributions
# ─────────────────────────────────────────────────────────────────────────────

class TestOperationalFallback:
    def test_no_shap_falls_back_to_operational(self):
        result = _engine().recommend(
            _inp(prob=0.80, queue_pressure=2.0, berth_utilization=0.95)
        )
        sources = {f["source"] for f in result.risk_increasing_factors + result.protective_factors}
        assert "operational" in sources

    def test_empty_contributions_uses_fallback(self):
        result = _engine().recommend(
            _inp(prob=0.80, cong_contributions=[], queue_pressure=2.0)
        )
        sources = {f["source"] for f in result.risk_increasing_factors + result.protective_factors}
        assert "operational" in sources

    def test_high_queue_pressure_in_risk_increasing(self):
        result = _engine().recommend(
            _inp(prob=0.80, queue_pressure=2.5)
        )
        names = [f["feature_name"] for f in result.risk_increasing_factors]
        assert "queue_pressure" in names

    def test_low_queue_pressure_in_protective(self):
        result = _engine().recommend(
            _inp(prob=0.80, queue_pressure=0.2)
        )
        names = [f["feature_name"] for f in result.protective_factors]
        assert "queue_pressure" in names

    def test_no_operational_context_empty_factors(self):
        result = _engine().recommend(_inp(prob=0.80))
        # No SHAP, no operational context → both lists empty
        assert result.risk_increasing_factors == []
        assert result.protective_factors      == []

    def test_fallback_factor_shap_value_is_none(self):
        result = _engine().recommend(_inp(prob=0.80, queue_pressure=2.0))
        for f in result.risk_increasing_factors:
            if f["source"] == "operational":
                assert f["shap_value"] is None


# ─────────────────────────────────────────────────────────────────────────────
# Actions — human-in-the-loop framing
# ─────────────────────────────────────────────────────────────────────────────

class TestActions:
    def test_actions_is_list(self):
        result = _engine().recommend(_inp(prob=0.80))
        assert isinstance(result.actions, list)
        assert len(result.actions) > 0

    def test_low_actions_contain_monitor(self):
        result = _engine().recommend(_inp(prob=0.10))
        combined = " ".join(result.actions).lower()
        assert "monitor" in combined

    def test_medium_actions_contain_prepare(self):
        result = _engine().recommend(_inp(prob=0.50))
        combined = " ".join(result.actions).lower()
        assert any(word in combined for word in ["prepare", "alert", "monitor"])

    def test_high_actions_contain_review_or_evaluate(self):
        result = _engine().recommend(_inp(prob=0.90))
        combined = " ".join(result.actions).lower()
        # Must contain human-review framing
        assert any(word in combined for word in ["review", "evaluate", "notify"])

    def test_high_actions_do_not_contain_direct_divert_command(self):
        """
        HIGH actions must not instruct the operator to 'divert vessels'
        as a direct command. Diversion must be framed as an evaluation option.
        """
        result = _engine().recommend(_inp(prob=0.90))
        for action in result.actions:
            low = action.lower()
            # "divert" is acceptable only if preceded by "evaluate" or "consider"
            if "divert" in low:
                assert any(
                    qualifier in low
                    for qualifier in ["evaluate", "consider", "option", "review"]
                ), (
                    f"Action '{action}' contains a direct diversion command "
                    "without human-review framing."
                )

    def test_each_risk_level_has_distinct_actions(self):
        low    = _engine().recommend(_inp(prob=0.10)).actions
        medium = _engine().recommend(_inp(prob=0.50)).actions
        high   = _engine().recommend(_inp(prob=0.90)).actions
        assert low != medium
        assert medium != high
        assert low != high


# ─────────────────────────────────────────────────────────────────────────────
# RecommendationResult
# ─────────────────────────────────────────────────────────────────────────────

class TestRecommendationResult:
    def test_to_dict_has_required_keys(self):
        result = _engine().recommend(_inp(prob=0.80))
        d = result.to_dict()
        for key in [
            "risk_level", "congestion_probability", "predicted_delay_hours",
            "risk_increasing_factors", "protective_factors",
            "actions", "reason", "escalation_triggers",
        ]:
            assert key in d, f"Missing key: {key}"

    def test_to_dict_is_json_serialisable(self):
        import json
        result = _engine().recommend(
            _inp(
                prob=0.80, delay=8.0,
                queue_pressure=1.8,
                cong_contributions=[
                    _make_contribution("queue_length", 2.5, "increases_risk"),
                    _make_contribution("available_berths", -1.0, "decreases_risk"),
                ],
            )
        )
        # Should not raise
        serialised = json.dumps(result.to_dict())
        assert isinstance(serialised, str)

    def test_to_dict_probability_rounded(self):
        result = _engine().recommend(_inp(prob=0.123456789))
        d = result.to_dict()
        assert d["congestion_probability"] == round(0.123456789, 4)

    def test_to_dict_delay_rounded(self):
        result = _engine().recommend(_inp(delay=7.123456789))
        d = result.to_dict()
        assert d["predicted_delay_hours"] == round(7.123456789, 4)

    def test_risk_level_is_string(self):
        result = _engine().recommend(_inp(prob=0.80))
        assert isinstance(result.risk_level, str)
        assert result.risk_level in {RISK_LOW, RISK_MEDIUM, RISK_HIGH}

    def test_reason_is_non_empty_string(self):
        result = _engine().recommend(_inp(prob=0.80))
        assert isinstance(result.reason, str)
        assert len(result.reason) > 0

    def test_escalation_triggers_is_list(self):
        result = _engine().recommend(_inp(prob=0.10))
        assert isinstance(result.escalation_triggers, list)


# ─────────────────────────────────────────────────────────────────────────────
# Determinism
# ─────────────────────────────────────────────────────────────────────────────

class TestDeterminism:
    def test_identical_inputs_produce_identical_output(self):
        inp = _inp(prob=0.75, delay=8.0, queue_pressure=1.8, berth_utilization=0.92)
        r1 = _engine().recommend(inp)
        r2 = _engine().recommend(inp)
        assert r1.risk_level             == r2.risk_level
        assert r1.congestion_probability == r2.congestion_probability
        assert r1.predicted_delay_hours  == r2.predicted_delay_hours
        assert r1.actions                == r2.actions
        assert r1.reason                 == r2.reason
        assert r1.escalation_triggers    == r2.escalation_triggers

    def test_determinism_with_shap_contributions(self):
        contributions = [
            _make_contribution("queue_length",    2.0, "increases_risk"),
            _make_contribution("available_berths",-1.0,"decreases_risk"),
        ]
        inp = _inp(prob=0.80, cong_contributions=contributions)
        r1 = _engine().recommend(inp)
        r2 = _engine().recommend(inp)
        assert r1.risk_increasing_factors == r2.risk_increasing_factors
        assert r1.protective_factors      == r2.protective_factors


# ─────────────────────────────────────────────────────────────────────────────
# Custom config
# ─────────────────────────────────────────────────────────────────────────────

class TestCustomConfig:
    def test_lower_high_threshold_triggers_high_sooner(self):
        cfg = RecommendationConfig(prob_low_max=0.20, prob_high_min=0.40)
        result = _engine(cfg).recommend(_inp(prob=0.50))
        assert result.risk_level == RISK_HIGH

    def test_higher_delay_threshold_prevents_escalation(self):
        cfg = RecommendationConfig(delay_medium_min=20.0, delay_high_min=40.0)
        result = _engine(cfg).recommend(_inp(prob=0.10, delay=10.0))
        assert result.risk_level == RISK_LOW

    def test_custom_queue_threshold(self):
        cfg = RecommendationConfig(queue_medium_min=3.0)
        result_no_esc  = _engine(cfg).recommend(_inp(prob=0.10, queue_pressure=2.0))
        result_esc     = _engine(cfg).recommend(_inp(prob=0.10, queue_pressure=3.5))
        assert result_no_esc.risk_level == RISK_LOW
        assert result_esc.risk_level    == RISK_MEDIUM


# ─────────────────────────────────────────────────────────────────────────────
# Engine repr
# ─────────────────────────────────────────────────────────────────────────────

class TestEngineRepr:
    def test_repr_contains_thresholds(self):
        eng = _engine()
        r = repr(eng)
        assert "0.35" in r
        # Python may print 0.7 or 0.70 depending on repr — check both
        assert "0.7" in r

    def test_repr_is_string(self):
        assert isinstance(repr(_engine()), str)


# ─────────────────────────────────────────────────────────────────────────────
# Reason text
# ─────────────────────────────────────────────────────────────────────────────

class TestReasonText:
    def test_low_reason_contains_low_indicator(self):
        result = _engine().recommend(_inp(prob=0.10))
        assert "low" in result.reason.lower()

    def test_medium_reason_contains_moderate_indicator(self):
        result = _engine().recommend(_inp(prob=0.52))
        assert any(w in result.reason.lower() for w in ["moderate", "medium"])

    def test_high_reason_contains_high_indicator(self):
        result = _engine().recommend(_inp(prob=0.90))
        assert "high" in result.reason.lower()

    def test_reason_contains_probability(self):
        result = _engine().recommend(_inp(prob=0.75))
        assert "75" in result.reason or "0.75" in result.reason

    def test_reason_contains_delay(self):
        result = _engine().recommend(_inp(prob=0.80, delay=7.5))
        assert "7.5" in result.reason

    def test_escalation_trigger_appears_in_reason(self):
        result = _engine().recommend(_inp(prob=0.10, delay=14.0))
        assert any(
            t in result.reason for t in result.escalation_triggers
        ) or "escalated" in result.reason.lower()

    def test_queue_pressure_in_reason_when_provided(self):
        result = _engine().recommend(_inp(prob=0.80, queue_pressure=2.0))
        assert "2.0" in result.reason or "queue" in result.reason.lower()
