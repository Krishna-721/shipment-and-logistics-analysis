# Experiments

## 1. Purpose

The experiments evaluate the prediction pipeline and determine which model provides the best balance of performance, calibration, and interpretability.

All experiments should use the same dataset split and evaluation procedure where applicable.

---

## E-001 — Baseline: Logistic Regression

**Goal:** Establish a simple baseline for congestion classification.

**Model:** Logistic Regression

**Metrics:**

* Precision
* Recall
* F1-score
* ROC-AUC
* PR-AUC
* Brier Score

**Expected outcome:** Provides a simple reference against which more complex models can be compared.

---

## E-002 — Random Forest

**Goal:** Determine whether a nonlinear tree-based model improves congestion prediction.

**Model:** Random Forest

**Compare against:** Logistic Regression

**Metrics:** Same classification metrics as E-001.

**Focus:** Improvement in predictive performance and ability to capture nonlinear operational relationships.

---

## E-003 — XGBoost

**Goal:** Evaluate another strong tree-based model for tabular data.

**Model:** XGBoost

**Compare against:**

* Logistic Regression
* Random Forest

**Metrics:** Same classification metrics.

**Focus:** Predictive improvement versus additional model complexity.

---

## E-004 — Feature Importance

**Goal:** Determine which operational factors are most useful for predicting congestion.

Analyze the importance of features such as:

* Vessel arrival pressure
* Queue pressure
* Berth utilization
* Weather
* Equipment availability
* Historical congestion
* Temporal patterns

Results will be compared with the expected operational relationships defined in `DATA_DESIGN.md`.

---

## E-005 — Feature Ablation

**Goal:** Determine whether major feature groups meaningfully contribute to prediction.

Compare the full model against versions with selected groups removed, such as:

* Without weather
* Without historical features
* Without operational-resource features
* Without temporal features

**Purpose:** Verify that the model is learning from meaningful signals rather than relying on a single feature group.

---

## E-006 — Probability Calibration

**Goal:** Determine whether predicted congestion probabilities correspond well to observed outcomes.

Evaluate:

* Brier Score
* Calibration curve

If necessary, apply probability calibration and compare the calibrated and uncalibrated models.

---

## E-007 — Delay Prediction

**Goal:** Estimate expected delay in hours.

**Metrics:**

* MAE
* RMSE
* R²

The selected regression model will be based on validation performance and simplicity.

---

## E-008 — Explainability

**Goal:** Verify that individual predictions can be explained in operational terms.

Use SHAP to identify the most influential features for sample predictions.

Example explanation:

> High vessel arrival pressure and limited berth availability are increasing congestion risk.

The explanations should be consistent with the underlying operational relationships.

---

## E-009 — Temporal Evaluation

**Goal:** Test whether the model can generalize to later observations.

Where appropriate, training data will contain earlier observations and evaluation data will contain later observations.

**Purpose:** Better approximate the real-world scenario of predicting future port conditions.

---

## 2. Model Selection

The final classification model will not be selected based on a single metric.

The decision will consider:

1. Predictive performance
2. Probability calibration
3. Explainability
4. Feature relevance
5. Model complexity
6. Inference cost

The final choice and reasoning will be documented after experiments are completed.

---

## 3. Experiment Tracking

Results will be stored under:

```text
artifacts/
├── metrics/
└── plots/
```

The final results will also be summarized in the project report.

No experiment result will be claimed before it has been executed.
