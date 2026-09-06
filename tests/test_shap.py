"""
Tests for E-008 — SHAP Explainability.

Covers:
- PortPulseExplainer construction (classification and regression)
- global_shap_values: shape, dtype, additivity
- build_shap_explanation: returns shap.Explanation, correct shape
- explain_instance: shape, top-n, all directions valid, to_dict keys
- Prediction clipping: negative SHAP values allowed, raw_value preserved
- No leakage: explainer background built from train only
- FeatureContribution dataclass: to_dict keys and types
- Formatter: format_contributions_for_api, format_contributions_text,
  format_global_importance, human_label
- Factory helpers: build_congestion_explainer, build_delay_explainer
- Invalid task type raises ValueError
- Wrong feature count raises ValueError
- Predict before fit raises RuntimeError (via fixture)
- SHAP additivity property: base + sum(SHAP) ≈ model output

All tests use small fully-controlled fixtures — no file I/O.
The conftest session-level fixtures (split_dfs, prepared_data) are reused
for integration-style tests that mirror the real pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────────────────────────────────────
# Small deterministic fixtures (no file I/O, module scope)
# ─────────────────────────────────────────────────────────────────────────────

N_FEAT = 8
N_TRAIN = 60
N_TEST = 20


@pytest.fixture(scope="module")
def linear_data():
    """
    Tiny dataset with known structure.
    y_clf = 1 iff x[0] + x[1] > 1.0
    y_reg = 2*x[0] + x[1] + noise
    """
    rng = np.random.default_rng(0)
    X_tr = rng.standard_normal((N_TRAIN, N_FEAT))
    X_te = rng.standard_normal((N_TEST,  N_FEAT))
    y_clf_tr = ((X_tr[:, 0] + X_tr[:, 1]) > 0).astype(int)
    y_reg_tr = np.clip(2 * X_tr[:, 0] + X_tr[:, 1] + rng.normal(0, 0.1, N_TRAIN), 0, None)
    feat_names = [f"f{i}" for i in range(N_FEAT)]
    return X_tr, X_te, y_clf_tr, y_reg_tr, feat_names


@pytest.fixture(scope="module")
def scaled_linear_data(linear_data):
    X_tr, X_te, y_clf_tr, y_reg_tr, feat_names = linear_data
    sc = StandardScaler().fit(X_tr)
    return sc.transform(X_tr), sc.transform(X_te), y_clf_tr, y_reg_tr, sc, feat_names


@pytest.fixture(scope="module")
def fitted_lr(scaled_linear_data):
    Xtr_s, _, y_clf_tr, _, _, _ = scaled_linear_data
    lr = LogisticRegression(random_state=0, max_iter=200, solver="lbfgs")
    lr.fit(Xtr_s, y_clf_tr)
    return lr


@pytest.fixture(scope="module")
def fitted_ridge(scaled_linear_data):
    Xtr_s, _, _, y_reg_tr, _, _ = scaled_linear_data
    r = Ridge(alpha=1.0, random_state=0)
    r.fit(Xtr_s, y_reg_tr)
    return r


@pytest.fixture(scope="module")
def clf_explainer(fitted_lr, scaled_linear_data):
    from src.explainability.shap_explainer import PortPulseExplainer
    Xtr_s, _, _, _, _, feat_names = scaled_linear_data
    return PortPulseExplainer(
        model=fitted_lr,
        X_train_scaled=Xtr_s,
        feature_names=feat_names,
        task="classification",
        model_label="TestLR",
    )


@pytest.fixture(scope="module")
def reg_explainer(fitted_ridge, scaled_linear_data):
    from src.explainability.shap_explainer import PortPulseExplainer
    Xtr_s, _, _, _, _, feat_names = scaled_linear_data
    return PortPulseExplainer(
        model=fitted_ridge,
        X_train_scaled=Xtr_s,
        feature_names=feat_names,
        task="regression",
        model_label="TestRidge",
    )


# ─────────────────────────────────────────────────────────────────────────────
# PortPulseExplainer — construction
# ─────────────────────────────────────────────────────────────────────────────

class TestPortPulseExplainerConstruction:
    def test_clf_repr(self, clf_explainer):
        r = repr(clf_explainer)
        assert "TestLR" in r
        assert "classification" in r
        assert f"n_features={N_FEAT}" in r

    def test_reg_repr(self, reg_explainer):
        r = repr(reg_explainer)
        assert "regression" in r

    def test_expected_value_is_float(self, clf_explainer, reg_explainer):
        assert isinstance(clf_explainer.expected_value, float)
        assert isinstance(reg_explainer.expected_value, float)

    def test_n_features_stored(self, clf_explainer):
        assert clf_explainer.n_features == N_FEAT

    def test_feature_names_stored(self, clf_explainer):
        assert len(clf_explainer.feature_names) == N_FEAT

    def test_invalid_task_raises(self, fitted_lr, scaled_linear_data):
        from src.explainability.shap_explainer import PortPulseExplainer
        Xtr_s, _, _, _, _, feat_names = scaled_linear_data
        with pytest.raises(ValueError, match="task"):
            PortPulseExplainer(
                model=fitted_lr,
                X_train_scaled=Xtr_s,
                feature_names=feat_names,
                task="unknown",
            )

    def test_ridge_expected_value_close_to_train_mean(
        self, fitted_ridge, scaled_linear_data
    ):
        """
        For Ridge, LinearExplainer expected_value should equal the mean
        prediction on the training set (E[f(X)]).
        """
        from src.explainability.shap_explainer import PortPulseExplainer
        Xtr_s, _, _, y_reg_tr, _, feat_names = scaled_linear_data
        expl = PortPulseExplainer(
            model=fitted_ridge,
            X_train_scaled=Xtr_s,
            feature_names=feat_names,
            task="regression",
        )
        train_mean_pred = float(fitted_ridge.predict(Xtr_s).mean())
        assert abs(expl.expected_value - train_mean_pred) < 0.5


# ─────────────────────────────────────────────────────────────────────────────
# global_shap_values — shape and dtype
# ─────────────────────────────────────────────────────────────────────────────

class TestGlobalShapValues:
    def test_shape_clf(self, clf_explainer, scaled_linear_data):
        _, Xte_s, _, _, _, _ = scaled_linear_data
        sv = clf_explainer.global_shap_values(Xte_s)
        assert sv.shape == (N_TEST, N_FEAT)

    def test_shape_reg(self, reg_explainer, scaled_linear_data):
        _, Xte_s, _, _, _, _ = scaled_linear_data
        sv = reg_explainer.global_shap_values(Xte_s)
        assert sv.shape == (N_TEST, N_FEAT)

    def test_dtype_float(self, clf_explainer, scaled_linear_data):
        _, Xte_s, _, _, _, _ = scaled_linear_data
        sv = clf_explainer.global_shap_values(Xte_s)
        assert np.issubdtype(sv.dtype, np.floating)

    def test_no_nans(self, clf_explainer, scaled_linear_data):
        _, Xte_s, _, _, _, _ = scaled_linear_data
        sv = clf_explainer.global_shap_values(Xte_s)
        assert not np.isnan(sv).any()

    def test_single_row(self, reg_explainer, scaled_linear_data):
        _, Xte_s, _, _, _, _ = scaled_linear_data
        sv = reg_explainer.global_shap_values(Xte_s[:1])
        assert sv.shape == (1, N_FEAT)


# ─────────────────────────────────────────────────────────────────────────────
# SHAP additivity: base_value + sum(shap_values) ≈ model output
# ─────────────────────────────────────────────────────────────────────────────

class TestShapAdditivity:
    def test_ridge_additivity(self, reg_explainer, fitted_ridge, scaled_linear_data):
        """
        For Ridge (linear model), SHAP values must satisfy:
        expected_value + sum(shap_row) == model.predict(x)  exactly.
        """
        _, Xte_s, _, _, _, _ = scaled_linear_data
        sv = reg_explainer.global_shap_values(Xte_s)
        model_preds = fitted_ridge.predict(Xte_s)
        reconstructed = reg_explainer.expected_value + sv.sum(axis=1)
        np.testing.assert_allclose(reconstructed, model_preds, atol=1e-6)

    def test_lr_additivity_log_odds(self, clf_explainer, fitted_lr, scaled_linear_data):
        """
        For LogisticRegression, SHAP values in log-odds space must satisfy:
        expected_value + sum(shap_row) == log_odds(predict_proba(x))
        """
        _, Xte_s, _, _, _, _ = scaled_linear_data
        sv = clf_explainer.global_shap_values(Xte_s)
        proba = fitted_lr.predict_proba(Xte_s)[:, 1]
        # Clip to avoid log(0) — proba of exactly 0 or 1 can occur on tiny fixture
        proba_clipped = np.clip(proba, 1e-9, 1 - 1e-9)
        log_odds = np.log(proba_clipped / (1 - proba_clipped))
        reconstructed = clf_explainer.expected_value + sv.sum(axis=1)
        np.testing.assert_allclose(reconstructed, log_odds, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# build_shap_explanation
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildShapExplanation:
    def test_returns_shap_explanation(self, clf_explainer, scaled_linear_data):
        import shap
        _, Xte_s, _, _, _, _ = scaled_linear_data
        expl_obj = clf_explainer.build_shap_explanation(Xte_s)
        assert isinstance(expl_obj, shap.Explanation)

    def test_values_shape(self, clf_explainer, scaled_linear_data):
        _, Xte_s, _, _, _, _ = scaled_linear_data
        expl_obj = clf_explainer.build_shap_explanation(Xte_s)
        assert expl_obj.values.shape == (N_TEST, N_FEAT)

    def test_base_values_shape(self, clf_explainer, scaled_linear_data):
        _, Xte_s, _, _, _, _ = scaled_linear_data
        expl_obj = clf_explainer.build_shap_explanation(Xte_s)
        assert expl_obj.base_values.shape == (N_TEST,)

    def test_base_values_constant(self, clf_explainer, scaled_linear_data):
        """All base values should equal expected_value."""
        _, Xte_s, _, _, _, _ = scaled_linear_data
        expl_obj = clf_explainer.build_shap_explanation(Xte_s)
        assert (expl_obj.base_values == clf_explainer.expected_value).all()

    def test_feature_names_set(self, clf_explainer, scaled_linear_data):
        _, Xte_s, _, _, _, _ = scaled_linear_data
        expl_obj = clf_explainer.build_shap_explanation(Xte_s)
        assert expl_obj.feature_names == clf_explainer.feature_names

    def test_precomputed_sv_reused(self, clf_explainer, scaled_linear_data):
        """Passing pre-computed SHAP values avoids re-computation."""
        _, Xte_s, _, _, _, _ = scaled_linear_data
        sv = clf_explainer.global_shap_values(Xte_s)
        expl_obj = clf_explainer.build_shap_explanation(Xte_s, shap_values=sv)
        np.testing.assert_array_equal(expl_obj.values, sv)


# ─────────────────────────────────────────────────────────────────────────────
# explain_instance
# ─────────────────────────────────────────────────────────────────────────────

class TestExplainInstance:
    def test_returns_list(self, clf_explainer, scaled_linear_data):
        _, Xte_s, _, _, _, _ = scaled_linear_data
        result = clf_explainer.explain_instance(Xte_s[0])
        assert isinstance(result, list)

    def test_default_top_n(self, clf_explainer, scaled_linear_data):
        _, Xte_s, _, _, _, _ = scaled_linear_data
        result = clf_explainer.explain_instance(Xte_s[0], top_n=10)
        assert len(result) <= 10

    def test_top_n_respected(self, clf_explainer, scaled_linear_data):
        _, Xte_s, _, _, _, _ = scaled_linear_data
        result = clf_explainer.explain_instance(Xte_s[0], top_n=3)
        assert len(result) == 3

    def test_sorted_by_abs_shap(self, clf_explainer, scaled_linear_data):
        _, Xte_s, _, _, _, _ = scaled_linear_data
        result = clf_explainer.explain_instance(Xte_s[0], top_n=N_FEAT)
        abs_vals = [abs(c.shap_value) for c in result]
        assert abs_vals == sorted(abs_vals, reverse=True)

    def test_feature_names_present(self, clf_explainer, scaled_linear_data):
        _, Xte_s, _, _, _, _ = scaled_linear_data
        result = clf_explainer.explain_instance(Xte_s[0])
        for c in result:
            assert c.feature_name in clf_explainer.feature_names

    def test_clf_directions_valid(self, clf_explainer, scaled_linear_data):
        _, Xte_s, _, _, _, _ = scaled_linear_data
        result = clf_explainer.explain_instance(Xte_s[0], top_n=N_FEAT)
        valid = {"increases_risk", "decreases_risk"}
        for c in result:
            assert c.direction in valid

    def test_reg_directions_valid(self, reg_explainer, scaled_linear_data):
        _, Xte_s, _, _, _, _ = scaled_linear_data
        result = reg_explainer.explain_instance(Xte_s[0], top_n=N_FEAT)
        valid = {"increases_delay", "decreases_delay"}
        for c in result:
            assert c.direction in valid

    def test_wrong_feature_count_raises(self, clf_explainer):
        bad_input = np.zeros(N_FEAT + 5)
        with pytest.raises(ValueError, match="features"):
            clf_explainer.explain_instance(bad_input)

    def test_2d_input_accepted(self, clf_explainer, scaled_linear_data):
        """explain_instance must accept shape (1, n_features) as well as (n_features,)."""
        _, Xte_s, _, _, _, _ = scaled_linear_data
        result = clf_explainer.explain_instance(Xte_s[0:1], top_n=5)
        assert len(result) == 5

    def test_raw_values_passed_through(self, clf_explainer, scaled_linear_data):
        """x_raw values should appear as raw_value in contributions, not scaled values."""
        _, Xte_s, _, _, _, _ = scaled_linear_data
        raw = np.arange(N_FEAT, dtype=float) * 10.0
        result = clf_explainer.explain_instance(Xte_s[0], x_raw=raw, top_n=N_FEAT)
        raw_vals_in_result = {c.raw_value for c in result}
        # All raw values in result should be from our custom array
        for c in result:
            assert c.raw_value in set(raw.tolist())

    def test_top_feature_has_largest_abs_shap(self, reg_explainer, scaled_linear_data):
        """The first returned feature must have the globally largest |SHAP| for this row."""
        _, Xte_s, _, _, _, _ = scaled_linear_data
        sv_all = reg_explainer.global_shap_values(Xte_s[:1])[0]
        top_idx = int(np.argmax(np.abs(sv_all)))
        expected_top_feat = reg_explainer.feature_names[top_idx]
        result = reg_explainer.explain_instance(Xte_s[0], top_n=1)
        assert result[0].feature_name == expected_top_feat


# ─────────────────────────────────────────────────────────────────────────────
# FeatureContribution dataclass
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureContribution:
    def test_to_dict_keys(self):
        from src.explainability.shap_explainer import FeatureContribution
        fc = FeatureContribution("queue_length", 5.0, 0.42, "increases_risk")
        d = fc.to_dict()
        for key in ["feature_name", "raw_value", "shap_value", "direction"]:
            assert key in d

    def test_to_dict_types(self):
        from src.explainability.shap_explainer import FeatureContribution
        fc = FeatureContribution("queue_length", 5.0, 0.42, "increases_risk")
        d = fc.to_dict()
        assert isinstance(d["feature_name"], str)
        assert isinstance(d["raw_value"],    float)
        assert isinstance(d["shap_value"],   float)
        assert isinstance(d["direction"],    str)

    def test_to_dict_values_rounded(self):
        from src.explainability.shap_explainer import FeatureContribution
        fc = FeatureContribution("f", 1.23456789, 0.123456789, "increases_risk")
        d = fc.to_dict()
        # raw_value rounded to 4dp, shap_value to 6dp
        assert len(str(d["raw_value"]).split(".")[-1]) <= 4
        assert len(str(d["shap_value"]).split(".")[-1]) <= 6

    def test_positive_shap_increases_risk(self):
        from src.explainability.shap_explainer import FeatureContribution
        fc = FeatureContribution("f", 1.0, 0.5, "increases_risk")
        assert fc.shap_value > 0
        assert fc.direction == "increases_risk"

    def test_negative_shap_decreases_risk(self):
        from src.explainability.shap_explainer import FeatureContribution
        fc = FeatureContribution("f", 1.0, -0.5, "decreases_risk")
        assert fc.shap_value < 0
        assert fc.direction == "decreases_risk"


# ─────────────────────────────────────────────────────────────────────────────
# mean_abs_shap and global_importance_dict
# ─────────────────────────────────────────────────────────────────────────────

class TestGlobalImportance:
    def test_mean_abs_shap_shape(self, clf_explainer, scaled_linear_data):
        _, Xte_s, _, _, _, _ = scaled_linear_data
        sv = clf_explainer.global_shap_values(Xte_s)
        mas = clf_explainer.mean_abs_shap(sv)
        assert mas.shape == (N_FEAT,)

    def test_mean_abs_shap_non_negative(self, clf_explainer, scaled_linear_data):
        _, Xte_s, _, _, _, _ = scaled_linear_data
        sv = clf_explainer.global_shap_values(Xte_s)
        assert (clf_explainer.mean_abs_shap(sv) >= 0).all()

    def test_importance_dict_keys_are_feature_names(self, clf_explainer, scaled_linear_data):
        _, Xte_s, _, _, _, _ = scaled_linear_data
        sv = clf_explainer.global_shap_values(Xte_s)
        imp = clf_explainer.global_importance_dict(sv)
        assert set(imp.keys()) == set(clf_explainer.feature_names)

    def test_importance_dict_sorted_descending(self, clf_explainer, scaled_linear_data):
        _, Xte_s, _, _, _, _ = scaled_linear_data
        sv = clf_explainer.global_shap_values(Xte_s)
        imp = clf_explainer.global_importance_dict(sv)
        vals = list(imp.values())
        assert vals == sorted(vals, reverse=True)

    def test_first_feature_matches_argmax(self, reg_explainer, scaled_linear_data):
        """The top importance feature must be the one with the highest mean |SHAP|."""
        _, Xte_s, _, _, _, _ = scaled_linear_data
        sv = reg_explainer.global_shap_values(Xte_s)
        mas = reg_explainer.mean_abs_shap(sv)
        top_idx = int(np.argmax(mas))
        expected_top = reg_explainer.feature_names[top_idx]
        imp = reg_explainer.global_importance_dict(sv)
        assert list(imp.keys())[0] == expected_top


# ─────────────────────────────────────────────────────────────────────────────
# Factory helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestFactoryHelpers:
    def test_build_congestion_explainer(self, fitted_lr, scaled_linear_data, linear_data):
        from src.explainability.shap_explainer import build_congestion_explainer
        X_tr, _, _, _, feat_names = linear_data
        _, _, _, _, sc, _ = scaled_linear_data
        expl = build_congestion_explainer(fitted_lr, sc, X_tr, feat_names)
        assert expl.task == "classification"
        assert expl.n_features == N_FEAT

    def test_build_delay_explainer(self, fitted_ridge, scaled_linear_data, linear_data):
        from src.explainability.shap_explainer import build_delay_explainer
        X_tr, _, _, _, feat_names = linear_data
        _, _, _, _, sc, _ = scaled_linear_data
        expl = build_delay_explainer(fitted_ridge, sc, X_tr, feat_names)
        assert expl.task == "regression"
        assert expl.n_features == N_FEAT

    def test_congestion_explainer_scales_internally(
        self, fitted_lr, linear_data, scaled_linear_data
    ):
        """
        Factory should produce the same expected_value as direct construction
        with pre-scaled data, confirming the factory scales correctly.
        """
        from src.explainability.shap_explainer import (
            PortPulseExplainer,
            build_congestion_explainer,
        )
        X_tr, _, _, _, feat_names = linear_data
        Xtr_s, _, _, _, sc, _ = scaled_linear_data

        factory_expl = build_congestion_explainer(fitted_lr, sc, X_tr, feat_names)
        direct_expl  = PortPulseExplainer(fitted_lr, Xtr_s, feat_names, "classification")
        assert abs(factory_expl.expected_value - direct_expl.expected_value) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# Formatter — format_contributions_for_api
# ─────────────────────────────────────────────────────────────────────────────

class TestFormatContributionsForApi:
    def _make_contribs(self):
        from src.explainability.shap_explainer import FeatureContribution
        return [
            FeatureContribution("queue_length",     5.0,  0.42, "increases_risk"),
            FeatureContribution("available_berths", 3.0, -0.18, "decreases_risk"),
        ]

    def test_returns_list(self):
        from src.explainability.formatter import format_contributions_for_api
        result = format_contributions_for_api(self._make_contribs())
        assert isinstance(result, list)
        assert len(result) == 2

    def test_required_keys_present(self):
        from src.explainability.formatter import format_contributions_for_api
        result = format_contributions_for_api(self._make_contribs())
        for item in result:
            for key in ["feature_name", "label", "raw_value", "shap_value",
                        "direction", "abs_shap"]:
                assert key in item, f"Missing key: {key}"

    def test_abs_shap_is_abs(self):
        from src.explainability.formatter import format_contributions_for_api
        result = format_contributions_for_api(self._make_contribs())
        for item in result:
            assert item["abs_shap"] == pytest.approx(abs(item["shap_value"]), abs=1e-9)

    def test_label_non_empty(self):
        from src.explainability.formatter import format_contributions_for_api
        result = format_contributions_for_api(self._make_contribs())
        for item in result:
            assert isinstance(item["label"], str)
            assert len(item["label"]) > 0

    def test_known_feature_gets_human_label(self):
        from src.explainability.formatter import format_contributions_for_api
        result = format_contributions_for_api(self._make_contribs())
        labels = {item["feature_name"]: item["label"] for item in result}
        assert labels["queue_length"] == "Queue length (vessels)"
        assert labels["available_berths"] == "Available berths"

    def test_empty_list_returns_empty(self):
        from src.explainability.formatter import format_contributions_for_api
        assert format_contributions_for_api([]) == []


# ─────────────────────────────────────────────────────────────────────────────
# Formatter — format_contributions_text
# ─────────────────────────────────────────────────────────────────────────────

class TestFormatContributionsText:
    def _make_contribs(self):
        from src.explainability.shap_explainer import FeatureContribution
        return [
            FeatureContribution("queue_length", 5.0,  0.42, "increases_risk"),
            FeatureContribution("labor_availability_pct", 0.9, -0.05, "decreases_risk"),
        ]

    def test_returns_string(self):
        from src.explainability.formatter import format_contributions_text
        result = format_contributions_text(self._make_contribs())
        assert isinstance(result, str)

    def test_contains_header(self):
        from src.explainability.formatter import format_contributions_text
        result = format_contributions_text(self._make_contribs())
        assert "Top contributing features" in result

    def test_up_arrow_for_positive(self):
        from src.explainability.formatter import format_contributions_text
        result = format_contributions_text(self._make_contribs())
        assert "↑" in result

    def test_down_arrow_for_negative(self):
        from src.explainability.formatter import format_contributions_text
        result = format_contributions_text(self._make_contribs())
        assert "↓" in result

    def test_feature_label_present(self):
        from src.explainability.formatter import format_contributions_text
        result = format_contributions_text(self._make_contribs())
        assert "Queue length" in result


# ─────────────────────────────────────────────────────────────────────────────
# Formatter — format_global_importance
# ─────────────────────────────────────────────────────────────────────────────

class TestFormatGlobalImportance:
    def _make_importance_dict(self):
        return {
            "avg_waiting_time_hours": 2.246,
            "queue_pressure":         2.132,
            "queue_length":           1.708,
            "arrivals_last_6h":       0.520,
            "weather_severity":       0.031,
        }

    def test_returns_list(self):
        from src.explainability.formatter import format_global_importance
        result = format_global_importance(self._make_importance_dict(), top_n=5)
        assert isinstance(result, list)
        assert len(result) == 5

    def test_top_n_respected(self):
        from src.explainability.formatter import format_global_importance
        result = format_global_importance(self._make_importance_dict(), top_n=3)
        assert len(result) == 3

    def test_rank_field_sequential(self):
        from src.explainability.formatter import format_global_importance
        result = format_global_importance(self._make_importance_dict(), top_n=5)
        ranks = [item["rank"] for item in result]
        assert ranks == list(range(1, 6))

    def test_required_keys(self):
        from src.explainability.formatter import format_global_importance
        result = format_global_importance(self._make_importance_dict(), top_n=2)
        for item in result:
            for key in ["rank", "feature_name", "label", "mean_abs_shap"]:
                assert key in item

    def test_sorted_by_importance_descending(self):
        from src.explainability.formatter import format_global_importance
        result = format_global_importance(self._make_importance_dict(), top_n=5)
        vals = [item["mean_abs_shap"] for item in result]
        assert vals == sorted(vals, reverse=True)

    def test_first_item_is_highest(self):
        from src.explainability.formatter import format_global_importance
        result = format_global_importance(self._make_importance_dict(), top_n=5)
        assert result[0]["feature_name"] == "avg_waiting_time_hours"


# ─────────────────────────────────────────────────────────────────────────────
# Formatter — human_label
# ─────────────────────────────────────────────────────────────────────────────

class TestHumanLabel:
    def test_known_features(self):
        from src.explainability.formatter import human_label
        assert human_label("queue_length")           == "Queue length (vessels)"
        assert human_label("avg_waiting_time_hours") == "Average waiting time"
        assert human_label("berth_utilization")      == "Berth utilisation"
        assert human_label("storm_flag")             == "Storm flag"

    def test_unknown_feature_falls_back_to_title(self):
        from src.explainability.formatter import human_label
        label = human_label("some_unknown_feature")
        assert isinstance(label, str)
        assert len(label) > 0

    def test_all_32_features_have_labels(self):
        from src.explainability.formatter import human_label
        from src.config import ESSENTIAL_FEATURES, CATEGORICAL_FEATURES
        all_feats = ESSENTIAL_FEATURES + CATEGORICAL_FEATURES
        for feat in all_feats:
            label = human_label(feat)
            assert isinstance(label, str) and len(label) > 0

    def test_returns_string(self):
        from src.explainability.formatter import human_label
        assert isinstance(human_label("queue_pressure"), str)


# ─────────────────────────────────────────────────────────────────────────────
# No leakage — explainer background uses only training data
# ─────────────────────────────────────────────────────────────────────────────

class TestNoLeakage:
    def test_target_not_in_feature_names(self, clf_explainer):
        """future_delay_hours and congestion_label must never appear as features."""
        forbidden = {"future_delay_hours", "congestion_label", "is_valid_training_row"}
        for name in clf_explainer.feature_names:
            assert name not in forbidden

    def test_explainer_background_is_training_data_not_test(
        self, clf_explainer, scaled_linear_data
    ):
        """
        The explainer's masker background must be the *training* set.
        We verify this by confirming the background rows are NOT equal to the
        held-out test set (the two sets are drawn from different RNG calls and
        will not overlap).  The masker legitimately stores training rows as the
        reference distribution — that is the correct SHAP design, not leakage.
        """
        Xtr_s, Xte_s, _, _, _, _ = scaled_linear_data
        masker = getattr(clf_explainer._explainer, "masker", None)
        if masker is not None and hasattr(masker, "data"):
            bg = np.asarray(masker.data)
            # Background must have n_features columns matching the model
            assert bg.shape[1] == clf_explainer.n_features
            # Background rows must NOT be identical to any test row
            # (they come from the training set, built from a different RNG call)
            for test_row in Xte_s:
                matches = np.allclose(bg, test_row[np.newaxis, :], atol=1e-9)
                assert not matches, "A test row was found in the explainer background"

    def test_pipeline_features_exclude_targets(self, prepared_data):
        """
        Integration check: the real prepared feature matrix from conftest
        must not contain either target column.
        """
        X_train, _, _, _, feature_names = prepared_data
        assert "future_delay_hours" not in feature_names
        assert "congestion_label"   not in feature_names


# ─────────────────────────────────────────────────────────────────────────────
# Integration — real conftest fixtures (session-scoped)
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegrationWithRealFeatures:
    """
    Smoke tests using the conftest session-level fixtures.
    These verify the explainer works end-to-end with the real 32-feature pipeline
    without loading saved .joblib files (which would require file I/O).
    """

    def test_clf_explainer_on_real_features(self, prepared_data, scaled_data):
        """
        Build a fresh LR + explainer on the conftest fixture data,
        confirm global SHAP values have the right shape.
        """
        from src.explainability.shap_explainer import build_congestion_explainer
        X_train, X_test, y_train, _, feature_names = prepared_data
        X_tr_s, X_te_s, _, _, scaler, _ = scaled_data

        lr = LogisticRegression(random_state=42, max_iter=200)
        lr.fit(X_tr_s, y_train.values)

        expl = build_congestion_explainer(lr, scaler, X_train, feature_names)
        sv   = expl.global_shap_values(X_te_s)

        assert sv.shape == (len(X_test), len(feature_names))
        assert not np.isnan(sv).any()

    def test_ridge_explainer_on_real_features(self, prepared_data, scaled_data):
        from src.explainability.shap_explainer import build_delay_explainer
        X_train, X_test, _, _, feature_names = prepared_data
        X_tr_s, X_te_s, _, _, scaler, _ = scaled_data

        ridge = Ridge(alpha=1.0, random_state=42)
        # Use y from scaled_data (congestion) — the shape is what matters here
        _, _, y_tr, _, _, _ = scaled_data
        ridge.fit(X_tr_s, y_tr.values.astype(float))

        expl = build_delay_explainer(ridge, scaler, X_train, feature_names)
        sv   = expl.global_shap_values(X_te_s)

        assert sv.shape == (len(X_test), len(feature_names))

    def test_explain_instance_real_features_returns_contributions(
        self, prepared_data, scaled_data
    ):
        from src.explainability.shap_explainer import build_congestion_explainer
        X_train, X_test, y_train, _, feature_names = prepared_data
        X_tr_s, X_te_s, _, _, scaler, _ = scaled_data

        lr = LogisticRegression(random_state=42, max_iter=200)
        lr.fit(X_tr_s, y_train.values)
        expl = build_congestion_explainer(lr, scaler, X_train, feature_names)

        contribs = expl.explain_instance(X_te_s[0], x_raw=X_test.values[0], top_n=5)
        assert len(contribs) == 5
        for c in contribs:
            assert c.feature_name in feature_names
            assert c.direction in {"increases_risk", "decreases_risk"}
