# Safiri PortPulse — Congestion Classification Experiment Report

## 1. What Was Implemented

A reproducible end-to-end ML classification pipeline comparing four models on a
temporal train/test split of the synthetic port-operations dataset.

**Pipeline stages:**

1. Temporal train/test split (chronological, no shuffling)
2. Feature selection and categorical encoding
3. StandardScaler fitted on training data only
4. Queue-only operational baseline (single-rule)
5. Logistic Regression
6. Random Forest (100 trees)
7. XGBoost (100 estimators)
8. Metric evaluation: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Brier Score
9. Feature importance extraction for all models
10. Artefact persistence (models, metrics JSON/CSV, plots)
11. 72 automated tests covering the full pipeline

---

## 2. Dataset and Split

**Source:** `data/raw/port_operations.csv`

| Property | Value |
|---|---|
| Total rows | 5,000 |
| Valid training rows | 4,880 |
| Ports | 20 synthetic |
| Features used | 32 |

**Temporal split (strict chronological — no future leakage):**

| Set | Period | Rows | Congestion rate |
|---|---|---|---|
| Train | 2026-01-01 → 2026-01-08 | 3,840 | 20.6 % |
| Test | 2026-01-09 → 2026-01-11 | 1,040 | 45.2 % |

**Distributional shift:** Congestion rises from 20.6 % in training to 45.2 % in test,
reflecting genuine queue build-up over the simulation window. Documented and accepted
rather than masked by random shuffling.

**Features (32):** 30 numeric operational features (capacity, traffic, queue/waiting,
equipment, weather, temporal, historical, derived pressure/interaction terms) plus
`vessel_type` and `cargo_type` as ordinal-encoded categoricals.

**Leakage check:** Confirmed — `congestion_label`, `future_delay_hours`,
`is_valid_training_row`, `port_id`, and `timestamp` excluded from all feature matrices.

---

## 3. Queue Rule Baseline Results

**Rule:** Predict congestion when `queue_length > 0.50 × total_berths`
(threshold fixed at audit value — not tuned on the test set)

| Metric | Value |
|---|---|
| Accuracy | 0.886 |
| Precision | 0.800 |
| Recall | **0.996** |
| F1 | 0.887 |
| ROC-AUC | 0.895 |
| PR-AUC | 0.799 |
| Brier Score | 0.112 |

**Confusion matrix (n = 1,040):**

| | Pred: No | Pred: Yes |
|---|---|---|
| Actual: No | 453 | **117** |
| Actual: Yes | **2** | 468 |

Near-perfect recall (0.996) — catches almost every true congestion event.
Poor precision (0.800) — 117 false alarms (20 % of non-congested hours flagged).
High Brier score (0.112) — hard 0.01/0.99 probabilities are not calibrated.
This is the benchmark every ML model must beat.

---

## 4. Logistic Regression Results

| Metric | Value | Δ vs Queue Rule |
|---|---|---|
| Accuracy | **0.968** | +0.083 |
| Precision | **0.974** | +0.174 |
| Recall | 0.955 | −0.040 |
| F1 | **0.965** | +0.077 |
| ROC-AUC | **0.996** | +0.101 |
| PR-AUC | **0.996** | +0.197 |
| Brier Score | **0.025** | −0.087 |
| Train time | 0.04 s | — |

**Confusion matrix (n = 1,040):**

| | Pred: No | Pred: Yes |
|---|---|---|
| Actual: No | 558 | 12 |
| Actual: Yes | 21 | 449 |

Top 3 features by absolute coefficient magnitude (after scaling):
`arrivals_last_6h` (3.23), `queue_length` (2.24), `vessels_anchored` (2.07).

---

## 5. Random Forest Results

| Metric | Value | Δ vs Queue Rule | Δ vs LR |
|---|---|---|---|
| Accuracy | 0.949 | +0.063 | −0.019 |
| Precision | 0.964 | +0.164 | −0.010 |
| Recall | 0.921 | −0.074 | −0.034 |
| F1 | 0.942 | +0.055 | −0.023 |
| ROC-AUC | 0.993 | +0.098 | −0.003 |
| PR-AUC | 0.993 | +0.194 | −0.003 |
| Brier Score | 0.031 | −0.081 | +0.006 |
| Train time | 0.34 s | — | — |

**Confusion matrix (n = 1,040):**

| | Pred: No | Pred: Yes |
|---|---|---|
| Actual: No | 554 | 16 |
| Actual: Yes | 37 | 433 |

Top 3 features by impurity: `queue_pressure` (0.245),
`queue_capacity_interaction` (0.192), `queue_length` (0.143).

---

## 6. XGBoost Results

| Metric | Value | Δ vs Queue Rule | Δ vs LR |
|---|---|---|---|
| Accuracy | 0.965 | +0.080 | −0.003 |
| Precision | 0.968 | +0.168 | −0.006 |
| Recall | 0.955 | −0.040 | 0.000 |
| F1 | 0.961 | +0.074 | −0.004 |
| ROC-AUC | 0.996 | +0.101 | 0.000 |
| PR-AUC | 0.995 | +0.196 | −0.001 |
| Brier Score | 0.027 | −0.085 | +0.002 |
| Train time | 0.17 s | — | — |

**Confusion matrix (n = 1,040):**

| | Pred: No | Pred: Yes |
|---|---|---|
| Actual: No | 555 | 15 |
| Actual: Yes | 21 | 449 |

Top 3 features by gain: `queue_pressure` (0.331),
`queue_length` (0.266), `queue_capacity_interaction` (0.113).

---

## 7. Best-Performing Model

**Selected: Logistic Regression**

| Criterion | Assessment |
|---|---|
| F1 | Best (0.965) |
| ROC-AUC | Best / tied with XGBoost (0.996) |
| PR-AUC | Best (0.996) |
| Accuracy | Best (0.968) |
| Precision | Best (0.974) |
| Brier Score | Best (0.025) — most calibrated |
| Recall | Second (0.955); Queue Rule highest at 0.996 |
| Train time | Fastest (0.04 s) |
| Interpretability | Directly readable coefficients |
| Complexity | Simplest ML model |

XGBoost is within 0.4 pp on F1 and identical on ROC-AUC. It is a valid
alternative if SHAP interaction values are prioritised in the next phase.
Random Forest is the weakest of the three: 37 false negatives vs 21 for the
other two, and no compensating advantage on any other metric.

---

## 8. Does ML Beat the Queue Baseline?

**Yes — all three ML models beat the queue-only rule on precision and F1.**

| Model | Beats baseline? | Precision gain | F1 gain |
|---|---|---|---|
| Logistic Regression | **Yes — meaningful** | +0.174 | +0.077 |
| Random Forest | **Yes — meaningful** | +0.164 | +0.055 |
| XGBoost | **Yes — meaningful** | +0.168 | +0.074 |

**The trade-off is explicit:** ML models reduce false alarms from 117 to 12–16
(−89 %), accepting 19–35 extra missed congestion events (+FN) in exchange.

In operational terms: the queue rule is the right choice if the cost of missing
a congestion event is very high (e.g. safety-critical). Logistic Regression is
the right choice if the cost of false alarms (unnecessary mobilisation, crew
standby, vessel rerouting) is also significant — which is the more common
port-operations scenario.

**Important caveat:** the dataset primarily represents congestion persistence
(nowcasting), not early-warning forecasting. The improvement from ML is real but
incremental. The system should not be described as detecting congestion risk
before any operational stress is visible.

---

## 9. Most Important Features

Consistent across all models:

| Rank | Feature | Why it matters |
|---|---|---|
| 1 | `queue_pressure` | Direct ratio of queue to berth capacity |
| 2 | `queue_length` | Raw vessel count in queue |
| 3 | `queue_capacity_interaction` | Interaction: high queue × high utilisation |
| 4 | `avg_waiting_time_hours` | Accumulated operational stress |
| 5 | `vessels_anchored` | Proxy for visible queue size |
| 6 | `arrivals_last_6h` (strong in LR) | Incoming pressure trend |
| 7 | `vessels_currently_in_port` | Total load including berths |
| 8 | `available_berths` | Remaining capacity headroom |

Weather, equipment, and labor features appear in the lower half of all importance
rankings — they contribute but do not dominate.

---

## 10. Unexpected Findings

**1. Logistic Regression outperforms tree models.**
On a synthetic tabular dataset with strong nonlinear structure expected from a
simulation, the simplest model won. After standardisation, the relationships
between features and the congestion label appear approximately linear. The
nonlinear capacity of Random Forest and XGBoost adds no measurable signal.

**2. The temporal distribution shift does not collapse performance.**
Despite a ~25 pp increase in congestion rate between train (20.6 %) and test
(45.2 %), all ML models maintain ROC-AUC ≥ 0.993.

**3. The queue rule achieves near-perfect recall despite simplicity.**
Recall 0.996 with a single threshold is remarkable. It directly confirms the
audit finding that the dataset is primarily nowcasting.

**4. The Brier score gap is large.**
Queue rule: 0.112. Logistic Regression: 0.025. This 4.5× difference in
probability calibration is the strongest argument for preferring ML in any
system that uses the predicted probability as a risk score.

---

## 11. Files Changed

| File | Change |
|---|---|
| `src/config.py` | New — project paths, split dates, feature lists, `load_dataset()` |
| `src/data/preprocessing.py` | New — `create_temporal_split`, `prepare_features`, `fit_scaler`, `apply_scaler` |
| `src/models/baseline.py` | New — `QueueOnlyBaseline` class |
| `src/models/congestion.py` | New — `CongestionClassifier` wrapper (LR / RF / XGB) |
| `src/models/evaluation.py` | New — `evaluate_classification_model`, `compare_models`, `analyze_baseline_value`, `format_results_table` |
| `scripts/train.py` | New — full experiment runner |
| `docs/EXPERIMENTS.md` | Rewritten with actual results (E-000 through E-009) |
| `docs/DESIGN_DECISIONS.md` | Extended with DD-15 through DD-18 |
| `requirements.txt` | Added pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, joblib, pytest |
| `pyproject.toml` | Added `[tool.pytest.ini_options]` |

---

## 12. Files Created

| File | Purpose |
|---|---|
| `tests/conftest.py` | Session-scoped shared fixtures (synthetic data, trained models) |
| `tests/test_generator.py` | 22 tests — reproducibility, shape, missing values, constraints, validity |
| `tests/test_features.py` | 14 tests — temporal split, feature preparation, scaling, encoding |
| `tests/test_models.py` | 36 tests — baseline logic, classifier interface, reproducibility, importance, metrics, persistence |
| `artifacts/models/logistic_regression.joblib` | Trained Logistic Regression + scaler |
| `artifacts/models/random_forest.joblib` | Trained Random Forest |
| `artifacts/models/xgboost.joblib` | Trained XGBoost |
| `artifacts/models/scaler.joblib` | Fitted StandardScaler |
| `artifacts/metrics/classification_results.json` | Full metrics including confusion matrices |
| `artifacts/metrics/classification_results.csv` | Summary comparison table |
| `artifacts/metrics/feature_names.json` | Ordered list of 32 model features |
| `artifacts/metrics/experiment_metadata.json` | Split details, dataset sizes, random state |
| `artifacts/plots/confusion_matrices.png` | 4-panel confusion matrix grid |
| `artifacts/plots/model_comparison.png` | Grouped bar chart of all metrics |
| `artifacts/plots/feature_importance_logistic_regression.png` | Top-15 LR coefficients |
| `artifacts/plots/feature_importance_random_forest.png` | Top-15 RF importances |
| `artifacts/plots/feature_importance_xgboost.png` | Top-15 XGB importances |
| `docs/EXPERIMENT_REPORT.md` | This document |

---

## 13. Tests Executed and Results (Classification Experiment)

```
pytest tests/test_generator.py tests/test_features.py tests/test_models.py -v

72 passed in 2.22s
```

| Module | Tests | Classes covered |
|---|---|---|
| `test_generator.py` | 22 | Reproducibility, Shape, MissingValues, PhysicalConstraints, ValidTrainingRows, TemporalOrdering, Targets |
| `test_features.py` | 14 | TemporalSplit, FeaturePreparation, Scaling, CategoricalEncoding |
| `test_models.py` | 36 | QueueOnlyBaseline, CongestionClassifierInterface, Reproducibility, FeatureImportance, EvaluationMetrics, ModelPersistence |

No warnings. No skipped tests. No flaky tests identified.

---

## 14. E-006 — Probability Calibration Results

### Objective

Determine whether Logistic Regression probability outputs are suitable as an operational congestion risk score.

### Method

- Loaded saved `logistic_regression.joblib` and `scaler.joblib` — no refitting.
- Reconstructed identical test set (n = 1,040) from raw data using the same temporal split.
- Computed Brier Score, Log-Loss, ECE (10 equal-width bins), and per-bin detail.
- Applied a material-miscalibration decision rule based on scalar thresholds and bin-count-weighted gap analysis.

### Results

| Metric | Value | Assessment |
|---|---|---|
| Brier Score | 0.0248 | Excellent (< 0.05 threshold) |
| Log-Loss | 0.0812 | Excellent (< 0.15 threshold) |
| ECE | 0.0207 | Good |

**Probability distribution:** 50.1 % of predictions ≤ 0.05 (confident No), 39.4 % ≥ 0.95 (confident Yes). Only 109/1,040 predictions (10.5 %) fall in the uncertain range (0.05, 0.95).

**Calibration gaps in middle bins (0.2–0.8) are noise artefacts.** Each bin contains only 7–17 samples. No bin with ≥ 20 samples has a calibration gap > 0.10.

**Anchor bins are well-calibrated:**
- Low bin [0.0, 0.1): n = 534, gap = 0.009
- High bin [0.9, 1.0): n = 421, gap = 0.002

### Decision

**No recalibration applied. Original model retained.**

Platt scaling and isotonic regression were not applied — fitting a calibration layer on 109 sparse middle-range samples would learn noise, not signal, and could degrade the excellent low/high-bin calibration.

### Operational Readiness

The model is suitable for use as a risk score in the PortPulse interface:
- "Congestion Risk: 2 %" and "Congestion Risk: 99 %" are reliable statements.
- The intermediate range is rare (~10 % of predictions) and appropriate for a confidence caveat in the UI.
- `logistic_regression.joblib` proceeds unchanged to all downstream stages.

### New Artefacts

| File | Description |
|---|---|
| `src/models/calibration.py` | `compute_calibration_metrics`, `CalibrationResult`, `fit_calibrated_wrapper` |
| `scripts/calibrate.py` | Full E-006 experiment runner |
| `tests/test_calibration.py` | 31 calibration tests (103 total in suite) |
| `artifacts/plots/logistic_regression_calibration.png` | Three-panel calibration figure |
| `artifacts/metrics/calibration_results.json` | Full calibration metrics |
| `artifacts/metrics/calibration_results.csv` | Summary row |

### Tests

```
pytest tests/ -v
103 passed, 2 warnings in 3.92s
```

31 new tests across 7 classes: `TestCalibrationResult`, `TestComputeCalibrationMetrics`,
`TestProbabilityBounds`, `TestCalibrationOnFixtureModel`, `TestCalibrationLeakage`,
`TestCompareCalibrationResults`, `TestBimodalCalibration`.

The 2 warnings are expected: sklearn's internal 5-fold CV warns about small class sizes
when the tiny fixture split (6 rows) is used for the leakage tests. They do not affect
production results.

---

## 15. E-007 — Delay Prediction Results

### Objective

Train and evaluate regression models for `future_delay_hours` using the same temporal split and 32-feature set as the congestion experiment.

### Dataset and Target

| Property | Value |
|---|---|
| Target | `future_delay_hours` |
| Train rows | 3,840 (valid-horizon only) |
| Test rows | 1,040 |
| Train mean delay | 4.20 h |
| Test mean delay | 7.04 h |
| Train max delay | 25.47 h |
| Distribution | Right-skewed (skew ≈ 1.92) |

**Temporal shift:** Test-set mean delay (7.04 h) is 68 % higher than training mean (4.20 h) — same queue build-up as congestion experiment.

**Leakage confirmed absent:** `future_delay_hours`, `congestion_label`, `is_valid_training_row` all excluded from feature matrix.

### Correlation Diagnostic

Top Pearson correlations with delay target (train set):

| Feature | r |
|---|---|
| `avg_waiting_time_hours` | 0.966 |
| `queue_length` | 0.963 |
| `vessels_anchored` | 0.962 |
| `queue_pressure` | 0.906 |

Single-feature linear regression on `queue_length` alone achieves R² = 0.90 on the test set. This confirms the same nowcasting limitation as the congestion task.

### Results

| Model | MAE | RMSE | R² | Train time |
|---|---|---|---|---|
| Baseline (train mean) | 5.2694 | 6.9675 | −0.200 | — |
| Baseline (train median) | 5.5481 | 7.7825 | −0.497 | — |
| **Ridge Regression** | **0.4834** | **0.6267** | **0.9903** | 0.01 s |
| Random Forest | 0.5860 | 0.7913 | 0.9845 | 0.94 s |
| XGBoost | 0.5159 | 0.6984 | 0.9879 | 0.15 s |

Negative baseline R² is correct and expected: the training-set mean badly undershoots
the higher-delay test period due to the temporal distribution shift.

### Selected Model: Ridge Regression

- Lowest MAE (0.4834 h) and RMSE (0.6267 h) among all models.
- Trains in 0.01 s. Coefficients are directly interpretable.
- Consistent with congestion finding: delay–feature relationships are approximately
  linear after standardisation; nonlinear tree models add no measurable benefit.
- All ML models beat both baselines on MAE by > 10× (5.27 → 0.48 h).

### Synthetic-Data Dependence Assessment

High R² (0.98–0.99) is expected and should not be over-interpreted:
- The delay target was generated as a near-linear transformation of current queue state.
- Models are primarily fitting queue-state persistence — not learning complex 6-hour dynamics.
- The value of ML over a naive rule is in reducing systematic error (MAE 5.27 → 0.48 h),
  not in discovering deep predictive patterns.

### New Artefacts

| File | Description |
|---|---|
| `src/models/delay.py` | `DelayBaseline`, `DelayRegressor`, `evaluate_delay_model` |
| `scripts/train_delay.py` | Full E-007 experiment runner |
| `tests/test_delay.py` | 51 delay tests (154 total in suite) |
| `artifacts/models/delay_ridge.joblib` | Selected Ridge Regression model |
| `artifacts/models/delay_random_forest.joblib` | Random Forest |
| `artifacts/models/delay_xgboost.joblib` | XGBoost |
| `artifacts/models/delay_scaler.joblib` | Scaler (fitted on train only) |
| `artifacts/metrics/delay_results.json` | Full regression metrics |
| `artifacts/metrics/delay_results.csv` | Summary table |
| `artifacts/metrics/delay_experiment_metadata.json` | Target stats, selected model, leakage flag |
| `artifacts/plots/delay_target_distribution.png` | Train vs test delay histogram |
| `artifacts/plots/delay_model_comparison.png` | MAE / RMSE / R² comparison bar chart |
| `artifacts/plots/delay_predicted_vs_actual.png` | Ridge: predicted vs actual scatter |
| `artifacts/plots/delay_residuals.png` | Ridge: residual diagnostics |
| `artifacts/plots/delay_feature_importance.png` | Ridge \|coeff\| and RF importance |

### Tests

```
pytest tests/ -q
154 passed, 2 warnings in 3.18s
```

51 new tests across 9 classes: `TestDelayBaseline`, `TestDelayRegressorInterface`,
`TestPredictionClipping`, `TestDelayFeatureImportance`, `TestEvaluateDelayModel`,
`TestFormatDelayResultsTable`, `TestLeakagePrevention`, `TestReproducibility`,
`TestDelayModelPersistence`, `TestDelayTargetExtraction`.

---

## 16. E-008 — SHAP Explainability Results

### Objective

Produce global and individual-prediction SHAP explanations for both selected models —
Logistic Regression (congestion) and Ridge Regression (delay) — without retraining
or altering either model.

### Explainer Design

| Property | Value |
|---|---|
| Library | SHAP 0.45.1 |
| Explainer type | `shap.LinearExplainer` |
| Masker | `shap.maskers.Independent` (training data only) |
| LR expected value | −6.330 (log-odds baseline) |
| Ridge expected value | 4.195 h (training-set mean delay) |
| n_explained | 1,040 (full test set, both models) |
| SHAP additivity | Verified: base + Σ(SHAP) = model output (atol 1e-5 / 1e-6) |
| Leakage | Confirmed absent — `future_delay_hours` and `congestion_label` not in feature names; no test rows in explainer background |

### Global Feature Importance

Both models agree: the same six queue-state features dominate, confirming the
nowcasting interpretation established in the data audit and E-007 diagnostic.

| Rank | Feature | Congestion mean \|SHAP\| | Delay mean \|SHAP\| (h) |
|---|---|---|---|
| 1 | `avg_waiting_time_hours` | 2.071 | 1.093 |
| 2 | `queue_pressure` | 1.926 | 0.763 |
| 3 | `vessels_anchored` | 1.853 | 0.724 |
| 4 | `queue_length` | 1.784 | 0.781 |
| 5 | `queue_capacity_interaction` | 1.596 | 0.407 |
| 6 | `vessels_currently_in_port` | 1.389 | 0.669 |
| 7 | `arrivals_last_6h` | 0.621 | 0.282 |
| 8 | `available_berths` | 0.589 | 0.244 |

Features outside the top 6 (arrival rates, equipment, weather, temporal) contribute
but are secondary in both tasks. This is fully consistent with the E-004 coefficient
analysis and the audit correlation findings.

### Cross-Model Consistency

| Feature | Congestion rank | Delay rank |
|---|---|---|
| `avg_waiting_time_hours` | 1 | 1 |
| `queue_pressure` | 2 | 3 |
| `vessels_anchored` | 3 | 4 |
| `queue_length` | 4 | 2 |
| `queue_capacity_interaction` | 5 | 6 |
| `vessels_currently_in_port` | 6 | 5 |

Queue variables rank 1–6 in both models. This cross-model agreement confirms that
both tasks are primarily driven by the same current operational state.

### Individual Prediction Examples

**Congestion — high-risk case** (P = 1.000, actual = congested)

| Feature | Value | SHAP |
|---|---|---|
| `vessels_anchored` | 22 | +8.94 |
| `queue_length` | 30.8 | +8.22 |
| `queue_capacity_interaction` | 2.80 | +7.86 |
| `vessels_currently_in_port` | 34 | +7.07 |
| `avg_waiting_time_hours` | 12.7 h | +6.03 |

Every top feature pushes strongly toward congestion. The expected value
(−6.33 log-odds ≈ 0.2 % baseline probability) is overcome entirely by
queue-state evidence.

**Congestion — low-risk case** (P = 0.000, actual = clear)

| Feature | Value | SHAP |
|---|---|---|
| `vessels_currently_in_port` | 2 | −1.88 |
| `vessels_anchored` | 0 | −1.86 |
| `queue_length` | 0.0 | −1.83 |
| `available_berths` | 9 | −1.83 |
| `queue_pressure` | 0.0 | −1.33 |

An empty queue with spare berths drives all major features strongly negative.

**Delay — high-delay case** (predicted 25.78 h, actual 25.35 h)

| Feature | Value | SHAP (h) |
|---|---|---|
| `avg_waiting_time_hours` | 16.66 h | +7.83 |
| `queue_length` | 26.3 | +3.14 |
| `vessels_anchored` | 18 | +2.71 |
| `vessels_currently_in_port` | 28 | +2.48 |
| `queue_pressure` | 2.19 | +1.27 |

**Delay — low-delay case** (predicted 0.00 h, actual 0.54 h)

| Feature | Value | SHAP (h) |
|---|---|---|
| `queue_length` | 0.12 | −0.74 |
| `vessels_anchored` | 0 | −0.74 |
| `vessels_currently_in_port` | 2 | −0.67 |
| `queue_pressure` | 0.01 | −0.55 |
| `available_berths` | 9 | −0.34 |

### SHAP Additivity Verification

For linear models, SHAP values satisfy exactly:

```
expected_value + Σ(shap_values_i) = model_output
```

Verified in the test suite (`TestShapAdditivity`) for both Ridge (atol=1e-6) and
LogisticRegression in log-odds space (atol=1e-5). No approximation is involved —
`LinearExplainer` computes exact analytical SHAP values.

### Operational Interpretation

Each prediction can now be explained as:

- **Global:** "Queue pressure, average waiting time, and vessel counts account for ~80 % of the model's explanatory power across the test set."
- **Individual:** "This port is predicted as high-risk (P=1.00) primarily because vessels_anchored=22 and queue_length=30.8 are far above the training baseline."

The `PortPulseExplainer.explain_instance()` method returns a ranked
`FeatureContribution` list ready for the API response layer. The
`format_contributions_for_api()` formatter converts it to a JSON-serialisable dict
with `feature_name`, `label`, `raw_value`, `shap_value`, `direction`, and
`abs_shap` fields.

### New Artefacts

| File | Description |
|---|---|
| `src/explainability/shap_explainer.py` | `PortPulseExplainer`, `FeatureContribution`, factory helpers |
| `src/explainability/formatter.py` | `format_contributions_for_api`, `format_contributions_text`, `format_global_importance`, `human_label` |
| `src/explainability/__init__.py` | Public exports |
| `scripts/explain.py` | Full E-008 experiment runner |
| `tests/test_shap.py` | 71 SHAP tests (225 total in suite) |
| `artifacts/plots/shap_congestion_beeswarm.png` | Global beeswarm — LR congestion (n=1,040) |
| `artifacts/plots/shap_congestion_bar.png` | Mean \|SHAP\| bar chart — congestion |
| `artifacts/plots/shap_congestion_waterfall_high_risk.png` | Waterfall — high-risk instance |
| `artifacts/plots/shap_congestion_waterfall_low_risk.png` | Waterfall — low-risk instance |
| `artifacts/plots/shap_delay_beeswarm.png` | Global beeswarm — Ridge delay (n=1,040) |
| `artifacts/plots/shap_delay_bar.png` | Mean \|SHAP\| bar chart — delay |
| `artifacts/plots/shap_delay_waterfall_high_delay.png` | Waterfall — high-delay instance |
| `artifacts/plots/shap_delay_waterfall_low_delay.png` | Waterfall — low-delay instance |
| `artifacts/metrics/shap_congestion_importance.json/csv` | Ranked global importance — congestion |
| `artifacts/metrics/shap_delay_importance.json/csv` | Ranked global importance — delay |
| `artifacts/metrics/shap_congestion_instance_high_risk.json` | Top-10 contributions — high-risk |
| `artifacts/metrics/shap_congestion_instance_low_risk.json` | Top-10 contributions — low-risk |
| `artifacts/metrics/shap_delay_instance_high_delay.json` | Top-10 contributions — high-delay |
| `artifacts/metrics/shap_delay_instance_low_delay.json` | Top-10 contributions — low-delay |
| `artifacts/metrics/shap_experiment_summary.json` | Experiment summary — both models |

### Tests

```
pytest tests/ -q
225 passed, 5 warnings in 6.73s
```

71 new tests across 12 classes: `TestPortPulseExplainerConstruction`,
`TestGlobalShapValues`, `TestShapAdditivity`, `TestBuildShapExplanation`,
`TestExplainInstance`, `TestFeatureContribution`, `TestGlobalImportance`,
`TestFactoryHelpers`, `TestFormatContributionsForApi`, `TestFormatContributionsText`,
`TestFormatGlobalImportance`, `TestHumanLabel`, `TestNoLeakage`,
`TestIntegrationWithRealFeatures`.

The 5 warnings are: 2 pre-existing calibration fixture warnings (sklearn small-class CV),
3 sklearn feature-names warnings from the integration tests (expected — conftest uses
array inputs with a DataFrame-fitted scaler).

---

## 17. Recommendation for Next Steps

**Both predictive models and their explanations are complete. The pipeline is ready
for the application layer.**

**Immediate next step: Recommendation Engine**

Wire `congestion_prob` + `predicted_delay_hours` + top SHAP `FeatureContribution`
list into `src/recommendations/engine.py`.

Suggested output schema for a prediction request:

```json
{
  "port_id": "PORT_01",
  "timestamp": "2026-01-10T14:00:00",
  "congestion_probability": 0.94,
  "risk_level": "HIGH",
  "predicted_delay_hours": 8.3,
  "confidence": "high",
  "key_factors": [
    {"label": "Queue pressure", "direction": "increases_risk", "shap_value": 1.93},
    {"label": "Average waiting time", "direction": "increases_risk", "shap_value": 1.72}
  ],
  "recommendation": "Consider diverting vessels. Queue pressure and wait times elevated."
}
```

**Following steps in order:**

1. **FastAPI endpoint** — `POST /predict` wrapping the full pipeline:
   load features → scale → predict congestion (LR) + delay (Ridge) → explain (SHAP) → recommend.
2. **Frontend** — Connect `frontend/` to the API; display risk badge, delay estimate,
   key-factors list, recommendation text.

**Do not retrain any models.** All four trained models (`logistic_regression.joblib`,
`delay_ridge.joblib`, `scaler.joblib`, `delay_scaler.joblib`) are production-ready.
