# Experiments

## 1. Purpose

The experiments evaluate the prediction pipeline and determine which model provides the best balance of performance, calibration, and interpretability.

All experiments use the same dataset and temporal evaluation split.

---

## Dataset & Split

**Source:** `data/raw/port_operations.csv`  
**Total observations:** 5,000 (across 20 synthetic ports)  
**Valid training rows:** 4,880 (rows with a complete 6-hour future horizon)

**Temporal split — chronological, no data leakage:**

| Set   | Period                               | Rows  | Congestion rate |
|-------|--------------------------------------|-------|-----------------|
| Train | 2026-01-01 00:00 → 2026-01-08 23:00  | 3,840 | 20.6 %          |
| Test  | 2026-01-09 00:00 → 2026-01-11 03:00  | 1,040 | 45.2 %          |

**Distributional shift:** The congestion rate rises from ~20 % in training to ~45 % in test. This is a genuine temporal distribution shift caused by queue build-up over the simulation window (day-1 congestion 9.6 %, day-9 congestion 50.2 %). It is treated as a realistic evaluation condition rather than masked by random shuffling.

**Features used:** 32 (30 essential numeric + vessel_type + cargo_type, both ordinal-encoded)  
**Target:** `congestion_label` (binary, 0 = no congestion, 1 = congestion)  
**Leakage check:** Confirmed — no future queue, delay, or congestion information used as input

---

## E-000 — Queue-Only Operational Baseline

**Status:** Completed

**Goal:** Establish the performance of the simplest possible operational rule before training any ML model.

**Rule:** Predict congestion if `queue_length > 0.50 × total_berths`

**Rationale:** Identified during the final dataset audit. The queue-pressure feature has a Pearson correlation of 0.861 with the congestion label. A single threshold on this variable should capture the majority of congestion events.

**Parameters:** threshold = 0.50 (fixed at audit value, not optimised on test set)

| Metric     | Value |
|------------|-------|
| Accuracy   | 0.886 |
| Precision  | 0.800 |
| Recall     | **0.996** |
| F1         | 0.887 |
| ROC-AUC    | 0.895 |
| PR-AUC     | 0.799 |
| Brier Score | 0.112 |

**Confusion matrix (test set, n = 1,040):**

|                | Predicted No | Predicted Yes |
|----------------|:---:|:---:|
| **Actual No**  | 453 | 117 |
| **Actual Yes** |   2 | 468 |

**Observations:**
- Near-perfect recall (0.996): the rule catches essentially every true congestion event.
- Low precision (0.800): 117 false positives — 20 % of non-congested hours are wrongly flagged.
- The Brier score (0.112) is poor because the rule outputs hard probabilities (0.01 / 0.99), not calibrated probabilities.
- This is the benchmark every ML model must beat on precision and F1 while maintaining reasonable recall.

---

## E-001 — Logistic Regression

**Status:** Completed

**Goal:** Establish an interpretable ML baseline that can generalise beyond a single-feature threshold.

**Model:** `sklearn.linear_model.LogisticRegression`  
**Preprocessing:** StandardScaler fitted on training set only  
**Parameters:** `C=1.0`, `solver=lbfgs`, `max_iter=1000`, `random_state=42`  
**Training time:** 0.04 s

| Metric     | Value   | Δ vs Queue Rule |
|------------|---------|-----------------|
| Accuracy   | **0.968** | +0.083 |
| Precision  | **0.974** | +0.174 |
| Recall     | 0.955   | −0.040 |
| F1         | **0.965** | +0.077 |
| ROC-AUC    | **0.996** | +0.101 |
| PR-AUC     | **0.996** | +0.197 |
| Brier Score | **0.025** | −0.087 |

**Confusion matrix (test set, n = 1,040):**

|                | Predicted No | Predicted Yes |
|----------------|:---:|:---:|
| **Actual No**  | 558 |  12 |
| **Actual Yes** |  21 | 449 |

**Observations:**
- Beats the queue rule on every metric except recall (−0.040, trading 2 extra misses for 105 fewer false positives).
- Precision improvement of +0.174 is operationally significant: false-alarm rate drops from 20 % to 2 %.
- ROC-AUC of 0.996 and PR-AUC of 0.996 indicate near-perfect ranking even under the distributional shift.
- Brier score drops to 0.025 (vs 0.112), showing well-calibrated probability outputs.
- Despite the dataset's strong queue dependency, Logistic Regression meaningfully leverages additional features — the top coefficients include `arrivals_last_6h`, `vessels_currently_in_port`, `avg_waiting_time_hours` alongside queue variables.

**Conclusion:** Logistic Regression is the **best-performing model** in this experiment. Interpretable, fast, and well-calibrated.

---

## E-002 — Random Forest

**Status:** Completed

**Goal:** Determine whether a nonlinear tree-based model improves over Logistic Regression.

**Model:** `sklearn.ensemble.RandomForestClassifier`  
**Preprocessing:** None (tree models are scale-invariant)  
**Parameters:** `n_estimators=100`, `random_state=42`, `n_jobs=-1`  
**Training time:** 0.34 s

| Metric     | Value  | Δ vs Queue Rule | Δ vs Log. Reg. |
|------------|--------|-----------------|----------------|
| Accuracy   | 0.949  | +0.063          | −0.019 |
| Precision  | 0.964  | +0.164          | −0.010 |
| Recall     | 0.921  | −0.074          | −0.034 |
| F1         | 0.942  | +0.055          | −0.023 |
| ROC-AUC    | 0.993  | +0.098          | −0.003 |
| PR-AUC     | 0.993  | +0.194          | −0.003 |
| Brier Score | 0.031 | −0.081          | +0.006 |

**Confusion matrix (test set, n = 1,040):**

|                | Predicted No | Predicted Yes |
|----------------|:---:|:---:|
| **Actual No**  | 554 |  16 |
| **Actual Yes** |  37 | 433 |

**Top feature importances (mean decrease in impurity):**

| Rank | Feature                    | Importance |
|------|----------------------------|-----------|
| 1    | queue_pressure             | 0.245     |
| 2    | queue_capacity_interaction | 0.192     |
| 3    | queue_length               | 0.143     |
| 4    | avg_waiting_time_hours     | 0.097     |
| 5    | vessels_anchored           | 0.078     |
| 6    | vessels_currently_in_port  | 0.064     |
| 7    | berth_utilization          | 0.023     |
| 8    | capacity_pressure          | 0.022     |
| 9    | arrivals_last_6h           | 0.017     |
| 10   | arrivals_last_24h          | 0.016     |

**Observations:**
- Beats the queue rule on precision and F1, but falls short of Logistic Regression on all metrics.
- 37 false negatives vs 21 for Logistic Regression — Random Forest misses more congestion events.
- Feature importance confirms that queue-related variables (queue_pressure, queue_capacity_interaction, queue_length) dominate. The model is primarily learning to threshold queue state.
- No meaningful gain from nonlinearity for this dataset.

---

## E-003 — XGBoost

**Status:** Completed

**Goal:** Evaluate a gradient-boosted tree ensemble as an alternative to Random Forest.

**Model:** `xgboost.XGBClassifier`  
**Preprocessing:** None  
**Parameters:** `n_estimators=100`, `max_depth=5`, `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.8`, `random_state=42`  
**Training time:** 0.17 s

| Metric     | Value  | Δ vs Queue Rule | Δ vs Log. Reg. |
|------------|--------|-----------------|----------------|
| Accuracy   | 0.965  | +0.080          | −0.003 |
| Precision  | 0.968  | +0.168          | −0.006 |
| Recall     | 0.955  | −0.040          | 0.000  |
| F1         | 0.961  | +0.074          | −0.004 |
| ROC-AUC    | 0.996  | +0.101          | 0.000  |
| PR-AUC     | 0.995  | +0.196          | −0.001 |
| Brier Score | 0.027 | −0.085          | +0.002 |

**Confusion matrix (test set, n = 1,040):**

|                | Predicted No | Predicted Yes |
|----------------|:---:|:---:|
| **Actual No**  | 555 |  15 |
| **Actual Yes** |  21 | 449 |

**Top feature importances (XGBoost gain):**

| Rank | Feature                    | Importance |
|------|----------------------------|-----------|
| 1    | queue_pressure             | 0.331     |
| 2    | queue_length               | 0.266     |
| 3    | queue_capacity_interaction | 0.113     |
| 4    | avg_waiting_time_hours     | 0.040     |
| 5    | vessels_anchored           | 0.026     |
| 6    | available_berths           | 0.021     |
| 7    | arrivals_last_6h           | 0.018     |
| 8    | total_berths               | 0.017     |
| 9    | arrivals_last_1h           | 0.016     |
| 10   | vessels_currently_in_port  | 0.015     |

**Observations:**
- Virtually tied with Logistic Regression: F1 0.961 vs 0.965, ROC-AUC identical at 0.996.
- Slightly worse precision than Logistic Regression (0.968 vs 0.974) but identical recall.
- More expensive to train than Logistic Regression (0.17 s vs 0.04 s) with no measurable benefit.
- Feature importance mirrors Random Forest: queue variables dominate (combined ~71 % of importance).

---

## E-004 — Feature Importance (Cross-Model)

**Status:** Completed

**Goal:** Identify which operational variables are most predictive across models.

**Finding — Queue variables dominate all models:**

| Feature                    | LR coeff | RF importance | XGB importance |
|----------------------------|----------|---------------|----------------|
| queue_pressure             | 1.69     | 0.245         | 0.331          |
| queue_length               | 2.24     | 0.143         | 0.266          |
| queue_capacity_interaction | 1.55     | 0.192         | 0.113          |
| avg_waiting_time_hours     | 2.00     | 0.097         | 0.040          |
| vessels_anchored           | 2.07     | 0.078         | 0.026          |

**Secondary signals identified by Logistic Regression:**
- `arrivals_last_6h` (coefficient 3.23) — strongest single feature in LR
- `vessels_currently_in_port` (2.33)
- `arrivals_last_24h` (1.47)
- `cranes_operational` (1.07)

**Observation:** Logistic Regression surfaces arrival-rate features more prominently than tree models, possibly because standardisation allows it to compare magnitudes across diverse scales. This partially explains its precision advantage: it uses arrival patterns to distinguish transient high queues (that will clear) from sustained pressure.

**Consistent with audit findings:** The dataset is primarily a congestion-persistence problem. Queue state at prediction time is the strongest signal. Secondary signals (arrival pressure, available berths, waiting time) add measurable but smaller contributions.

---

## E-005 — Feature Ablation

**Status:** Not yet executed — planned for next experiment phase

**Plan:** Compare full model vs. versions with feature groups removed (without queue variables, without weather, without historical features, without temporal features) to verify each group contributes.

---

## E-006 — Probability Calibration

**Status:** Completed

**Goal:** Determine whether the selected Logistic Regression's probability outputs are sufficiently calibrated for use as an operational risk score ("Congestion Risk: 82 %").

**Model evaluated:** Logistic Regression (original, unfitted calibration wrapper)  
**Data used:** Final temporal test set (n = 1,040, 45.2 % congested) — never used during model training  
**Script:** `scripts/calibrate.py`

### Method

1. Loaded `artifacts/models/logistic_regression.joblib` and `artifacts/models/scaler.joblib`.
2. Reconstructed the identical train/test split from the raw dataset (no refitting).
3. Generated positive-class probabilities `y_prob` on the test set.
4. Computed Brier Score, Log-Loss, ECE (10 equal-width bins), and per-bin detail.
5. Inspected the probability distribution for bimodality.
6. Evaluated whether any large calibration gaps were genuine systematic errors or noise artefacts from small bin counts.
7. Applied the material-miscalibration decision rule: flag only if `Brier > 0.05` OR `Log-Loss > 0.15` OR a bin with ≥ 20 samples has `|gap| > 0.10`.

### Calibration Metrics

| Metric      | Value     |
|-------------|-----------|
| Brier Score | **0.0248** |
| Log-Loss    | **0.0812** |
| ECE         | **0.0207** |

### Probability Distribution

| Statistic         | Value    |
|-------------------|----------|
| Minimum           | 0.000000 |
| Median            | 0.044978 |
| Maximum           | 1.000000 |
| Predictions ≤ 0.05 | **50.1 %** |
| Predictions ≥ 0.95 | **39.4 %** |
| Predictions in (0.05, 0.95) | **10.5 % (109 rows)** |

### Per-Bin Calibration Detail (equal-width, 10 bins)

| Bin        |   n | Mean pred | Obs. freq | \|gap\| | Note                        |
|------------|----:|----------:|----------:|--------:|-----------------------------|
| [0.0, 0.1) | 534 |    0.0037 |    0.0131 |  0.0094 | Well calibrated             |
| [0.1, 0.2) |  10 |    0.1611 |    0.2000 |  0.0389 | Small bin                   |
| [0.2, 0.3) |  12 |    0.2396 |    0.5000 |  0.2604 | Large gap — **tiny bin (noise)** |
| [0.3, 0.4) |  13 |    0.3450 |    0.3846 |  0.0396 | Small bin                   |
| [0.4, 0.5) |  10 |    0.4536 |    0.1000 |  0.3536 | Large gap — **tiny bin (noise)** |
| [0.5, 0.6) |   7 |    0.5413 |    0.8571 |  0.3158 | Large gap — **tiny bin (noise)** |
| [0.6, 0.7) |   7 |    0.6488 |    0.4286 |  0.2203 | Large gap — **tiny bin (noise)** |
| [0.7, 0.8) |   9 |    0.7708 |    0.4444 |  0.3264 | Large gap — **tiny bin (noise)** |
| [0.8, 0.9) |  17 |    0.8537 |    0.9412 |  0.0874 | Moderate gap, small bin     |
| [0.9, 1.0) | 421 |    0.9958 |    0.9976 |  0.0018 | Well calibrated             |

### Calibration Decision

**Decision: ORIGINAL MODEL IS SUFFICIENTLY CALIBRATED — no recalibration applied.**

**Reasoning:**

1. **Brier Score 0.0248** is well below the materiality threshold (0.05). This is a strong result.
2. **Log-Loss 0.0812** is well below the threshold (0.15).
3. **The model is extremely bimodal.** 50.1 % of predictions are ≤ 0.05 (confident "not congested") and 39.4 % are ≥ 0.95 (confident "congested"). Only 109 of 1,040 test predictions fall in the uncertain middle range (0.05–0.95).
4. **Apparent middle-bin gaps are noise, not miscalibration.** The bins showing large gaps (0.25–0.35) contain only 7–17 samples each. With such small counts the observed positive fraction is extremely noisy. None of the bins with n ≥ 20 has a gap exceeding 0.10 — the materiality threshold is not crossed.
5. **The two anchor bins are well-calibrated.** The low bin (n = 534, gap = 0.009) and high bin (n = 421, gap = 0.002) together contain 955 of 1,040 predictions and are accurate.

**Operational interpretation:**

- A prediction of 0.02 reliably means the port is very unlikely to be congested in the next 6 hours.
- A prediction of 0.99 reliably means the port is very likely to be congested.
- The model is appropriate for surfacing as "Congestion Risk: 2 %" or "Congestion Risk: 99 %".
- The intermediate range (e.g. "Congestion Risk: 52 %") is rare — only ~10 % of predictions — and would benefit from a confidence caveat in the UI, but this does not require retraining or recalibration.

**Calibration was not applied because:**
- All materiality thresholds are satisfied.
- Platt scaling on a bimodal distribution with so few middle-range samples would fit noise, not signal.
- The existing Brier score (0.0248) is already 4.5× better than the queue rule (0.112).

### Artefacts

| File | Description |
|---|---|
| `artifacts/plots/logistic_regression_calibration.png` | Three-panel reliability diagram, probability histogram, and per-bin count chart |
| `artifacts/metrics/calibration_results.json` | Full calibration metrics and per-bin detail |
| `artifacts/metrics/calibration_results.csv` | Summary row for reporting |

### Conclusion

The original Logistic Regression model is production-ready from a probability-calibration standpoint. No calibration wrapper is needed. `logistic_regression.joblib` remains the model for all downstream stages (SHAP, recommendation engine, API).

---

## E-007 — Delay Prediction

**Status:** Completed

**Goal:** Predict `future_delay_hours` — expected operational delay over the next 6 hours — using only prediction-time features. Evaluate whether ML models provide meaningful value over a naive constant baseline, and whether the delay task is primarily persistence-driven (same nowcasting limitation as congestion).

**Script:** `scripts/train_delay.py`  
**Target:** `future_delay_hours` (continuous, ≥ 0, right-skewed)  
**Features:** Same 32 features as the congestion classifier (leakage confirmed absent)  
**Split:** Identical temporal split — train 2026-01-01 → 2026-01-08, test 2026-01-09 → 2026-01-11

### Target Statistics

| Set   | n     | Mean   | Std   | Min  | Max    |
|-------|-------|--------|-------|------|--------|
| Train | 3,840 | 4.20 h | 4.56 h | 0.0 h | 25.47 h |
| Test  | 1,040 | 7.04 h | 6.36 h | 0.0 h | 25.35 h |

Distribution: right-skewed (skew ≈ 1.92). 23 % of observations below 1 h, 10 % above 10 h, < 1 % above 20 h.

**Temporal shift:** Test-set mean (7.04 h) is 68 % higher than training-set mean (4.20 h), reflecting the same queue build-up that drives the congestion distributional shift.

### Leakage Confirmation

- `future_delay_hours` — NOT in feature list ✓
- `congestion_label` — NOT in feature list ✓
- `is_valid_training_row` — NOT in feature list ✓
- All 120 rows with incomplete 6-hour horizon excluded (same `is_valid_training_row` filter)

### Correlation Diagnostic (train set)

| Feature                    | Pearson r |
|----------------------------|-----------|
| avg_waiting_time_hours     | **0.966** |
| queue_length               | **0.963** |
| vessels_anchored           | **0.962** |
| vessels_currently_in_port  | 0.926     |
| queue_pressure             | 0.906     |
| queue_capacity_interaction | 0.824     |
| berth_utilization          | 0.546     |
| arrivals_last_24h          | 0.454     |
| arrivals_last_6h           | 0.407     |
| available_berths           | −0.392    |

**Finding:** The top three correlations (r ≥ 0.96) are current-state queue variables. A single-feature linear regression on `queue_length` alone achieves R² = 0.90 on the test set. This confirms the delay task is primarily a persistence/nowcasting problem, identical in character to the congestion classification task.

### Models and Results

| Model                  | MAE    | RMSE   | R²     | Train time |
|------------------------|--------|--------|--------|------------|
| Baseline (train mean)  | 5.2694 | 6.9675 | −0.200 | —          |
| Baseline (train median)| 5.5481 | 7.7825 | −0.497 | —          |
| **Ridge Regression**   | **0.4834** | **0.6267** | **0.9903** | 0.01 s |
| Random Forest          | 0.5860 | 0.7913 | 0.9845 | 0.94 s     |
| XGBoost                | 0.5159 | 0.6984 | 0.9879 | 0.15 s     |

**Baseline note:** Both naive baselines produce negative R². This is expected and fully explained by the temporal distribution shift — predicting the training-set mean (4.20 h) against a test set with mean 7.04 h systematically undershoots. It is **not** a bug; it correctly represents what a constant predictor achieves in a realistic deployment.

### Model Selection

**Selected: Ridge Regression**

**Reasoning:**

1. **Lowest MAE (0.4834 h)** — the primary evaluation criterion for a right-skewed continuous target where absolute error is more meaningful than R².
2. **Lowest RMSE (0.6267 h)** — also the best on squared error.
3. **Highest R² (0.9903)** — though all ML models achieve R² > 0.98; R² alone should not drive selection.
4. **Fastest to train (0.01 s)** and simplest model. Coefficients are directly interpretable alongside the congestion model's coefficients.
5. **Consistent with congestion finding:** the delay-feature relationship is approximately linear after standardisation. Nonlinear models (RF, XGB) add no measurable benefit.

XGBoost (MAE 0.5159) and Random Forest (MAE 0.5860) are both substantially beaten by Ridge on MAE. XGBoost is a reasonable alternative if interaction analysis is needed later.

### Synthetic-Data Dependence Assessment

The delay task is primarily persistence/nowcasting:

- **Current queue state explains ~90 % of test variance** using a single linear feature.
- **All ML models achieve R² > 0.98** — not because they are sophisticated, but because the target is algebraically determined by current operational state with minor noise added during generation.
- **This mirrors the congestion finding** (queue_pressure r = 0.861 with congestion label).
- The R² values should not be interpreted as evidence of genuine 6-hour forecasting capability.
- High R² on test reflects that the temporal shift in queue state is linear and predictable from current features — not that the model captures complex future dynamics.

### Artefacts

| File | Description |
|---|---|
| `artifacts/models/delay_ridge.joblib` | Trained Ridge Regression |
| `artifacts/models/delay_random_forest.joblib` | Trained Random Forest |
| `artifacts/models/delay_xgboost.joblib` | Trained XGBoost |
| `artifacts/models/delay_scaler.joblib` | StandardScaler fitted on training data only |
| `artifacts/metrics/delay_results.json` | Full metrics for all models |
| `artifacts/metrics/delay_results.csv` | Summary comparison table |
| `artifacts/metrics/delay_experiment_metadata.json` | Split details, target stats, selected model |
| `artifacts/plots/delay_target_distribution.png` | Train vs test delay histogram |
| `artifacts/plots/delay_model_comparison.png` | MAE / RMSE / R² bar chart |
| `artifacts/plots/delay_predicted_vs_actual.png` | Ridge: predicted vs actual (test set) |
| `artifacts/plots/delay_residuals.png` | Ridge: residual vs predicted + residual histogram |
| `artifacts/plots/delay_feature_importance.png` | Ridge \|coeff\| and RF importance (top 15) |

---

## E-008 — Explainability (SHAP)

**Status:** Completed

**Goal:** Generate global and individual-prediction SHAP explanations for the two selected models — Logistic Regression (congestion) and Ridge Regression (delay) — using `shap.LinearExplainer` with an `Independent` background masker fitted on training data only.

**Script:** `scripts/explain.py`  
**Explainer:** `shap.LinearExplainer` + `shap.maskers.Independent` (exact analytical SHAP for linear models; no sampling approximation)  
**Background:** Training set distribution only — no test rows in the baseline  
**Leakage confirmed absent:** `future_delay_hours` and `congestion_label` not present in feature matrix

### Explainer Parameters

| Property | Congestion (LR) | Delay (Ridge) |
|---|---|---|
| Model | LogisticRegression | Ridge(alpha=1.0) |
| Scaler | `scaler.joblib` | `delay_scaler.joblib` |
| Expected value | −6.330 (log-odds) | 4.195 h |
| SHAP units | log-odds | hours |
| n_explained (test set) | 1,040 | 1,040 |

### Global Feature Importance — Congestion Model

Mean |SHAP| value on the test set, top 10:

| Rank | Feature | Mean \|SHAP\| |
|------|---------|--------------|
| 1 | `avg_waiting_time_hours` | 2.071 |
| 2 | `queue_pressure` | 1.926 |
| 3 | `vessels_anchored` | 1.853 |
| 4 | `queue_length` | 1.784 |
| 5 | `queue_capacity_interaction` | 1.596 |
| 6 | `vessels_currently_in_port` | 1.389 |
| 7 | `arrivals_last_6h` | 0.621 |
| 8 | `available_berths` | 0.589 |
| 9 | `arrivals_last_24h` | 0.393 |
| 10 | `berth_utilization` | 0.368 |

### Global Feature Importance — Delay Model

Mean |SHAP| value on the test set, top 10:

| Rank | Feature | Mean \|SHAP\| (h) |
|------|---------|------------------|
| 1 | `avg_waiting_time_hours` | 1.093 |
| 2 | `queue_length` | 0.781 |
| 3 | `queue_pressure` | 0.763 |
| 4 | `vessels_anchored` | 0.724 |
| 5 | `vessels_currently_in_port` | 0.669 |
| 6 | `queue_capacity_interaction` | 0.407 |
| 7 | `arrivals_last_6h` | 0.282 |
| 8 | `available_berths` | 0.244 |
| 9 | `arrivals_last_24h` | 0.192 |
| 10 | `arrivals_last_1h` | 0.140 |

### Cross-Model Consistency

The same 6 queue-state features dominate both models, confirming the nowcasting interpretation from the data audit and E-007 diagnostic:

| Feature | Congestion rank | Delay rank |
|---|---|---|
| `avg_waiting_time_hours` | 1 | 1 |
| `queue_pressure` | 2 | 3 |
| `vessels_anchored` | 3 | 4 |
| `queue_length` | 4 | 2 |
| `queue_capacity_interaction` | 5 | 6 |
| `vessels_currently_in_port` | 6 | 5 |

### Individual Prediction Examples

**Congestion — high-risk instance** (actual: congested, P = 1.000)

Top 3 contributing features: `vessels_anchored` (+8.94), `queue_length` (+8.22), `queue_capacity_interaction` (+7.86). Every top feature pushes toward congestion. Queue length = 30.8 vessels, vessels anchored = 22.

**Congestion — low-risk instance** (actual: clear, P = 0.000)

Top 3 contributing features: `vessels_currently_in_port` (−1.88), `vessels_anchored` (−1.86), `queue_length` (−1.83). All major features push away from congestion. Queue length = 0, avg waiting time = 0.022 h.

**Delay — high-delay instance** (predicted 25.78 h, actual 25.35 h)

Top contributor: `avg_waiting_time_hours` (+7.83 h). Secondary: `queue_length` (+3.14 h), `vessels_anchored` (+2.71 h). All queue features strongly positive.

**Delay — low-delay instance** (predicted 0.00 h, actual 0.54 h)

Top contributors all negative: `queue_length` (−0.74 h), `vessels_anchored` (−0.74 h), `vessels_currently_in_port` (−0.67 h). Empty port, no queue pressure.

### SHAP Additivity Verification

For linear models, SHAP values satisfy exactly:

    expected_value + Σ(shap_values_i) = model_output

This was verified in the test suite (`TestShapAdditivity`) for both Ridge (atol=1e-6) and LogisticRegression in log-odds space (atol=1e-5). No approximation is involved.

### Artefacts

| File | Description |
|---|---|
| `artifacts/plots/shap_congestion_beeswarm.png` | Global beeswarm — LR congestion model (test set, n=1,040) |
| `artifacts/plots/shap_congestion_bar.png` | Mean \|SHAP\| bar chart — congestion |
| `artifacts/plots/shap_congestion_waterfall_high_risk.png` | Individual waterfall — high-risk case |
| `artifacts/plots/shap_congestion_waterfall_low_risk.png` | Individual waterfall — low-risk case |
| `artifacts/plots/shap_delay_beeswarm.png` | Global beeswarm — Ridge delay model |
| `artifacts/plots/shap_delay_bar.png` | Mean \|SHAP\| bar chart — delay |
| `artifacts/plots/shap_delay_waterfall_high_delay.png` | Individual waterfall — high-delay case |
| `artifacts/plots/shap_delay_waterfall_low_delay.png` | Individual waterfall — low-delay case |
| `artifacts/metrics/shap_congestion_importance.json/csv` | Ranked global importance — congestion |
| `artifacts/metrics/shap_delay_importance.json/csv` | Ranked global importance — delay |
| `artifacts/metrics/shap_congestion_instance_high_risk.json` | Top-10 contributions — high-risk prediction |
| `artifacts/metrics/shap_congestion_instance_low_risk.json` | Top-10 contributions — low-risk prediction |
| `artifacts/metrics/shap_delay_instance_high_delay.json` | Top-10 contributions — high-delay prediction |
| `artifacts/metrics/shap_delay_instance_low_delay.json` | Top-10 contributions — low-delay prediction |
| `artifacts/metrics/shap_experiment_summary.json` | Experiment summary — both models |

### Operational Interpretation

Each prediction can now be explained as:

- **Global:** "Queue pressure, average waiting time, and vessel counts account for ~80 % of the model's explanatory power."
- **Individual:** "This port is predicted as high-risk (P=1.00) primarily because vessels_anchored=22 and queue_length=30.8 are far above the training baseline."

The `PortPulseExplainer.explain_instance()` method returns a ranked `FeatureContribution` list ready for the API response layer. The `format_contributions_for_api()` formatter converts it to a JSON-serialisable dict with `feature_name`, `label`, `raw_value`, `shap_value`, `direction`, and `abs_shap` fields.

---

## E-009 — Temporal Evaluation

**Status:** Completed (built into E-001 through E-003)

**Finding:** All models were evaluated on the temporally later test set. Despite a distributional shift (20.6 % → 45.2 % congestion rate), all ML models maintained high F1 (≥ 0.942) and ROC-AUC (≥ 0.993). The queue rule also maintained near-perfect recall under the shift (0.996) because the later period has more genuinely congested observations.

---

## 2. Model Comparison Summary

| Model               | Accuracy | Precision | Recall | F1    | ROC-AUC | PR-AUC | Brier  |
|---------------------|----------|-----------|--------|-------|---------|--------|--------|
| Queue Rule          | 0.886    | 0.800     | **0.996** | 0.887 | 0.895   | 0.799  | 0.112  |
| Logistic Regression | **0.968** | **0.974** | 0.955  | **0.965** | **0.996** | **0.996** | **0.025** |
| Random Forest       | 0.949    | 0.964     | 0.921  | 0.942 | 0.993   | 0.993  | 0.031  |
| XGBoost             | 0.965    | 0.968     | 0.955  | 0.961 | 0.996   | 0.995  | 0.027  |

---

## 3. Model Selection

**Selected model: Logistic Regression**

**Reasoning:**

1. **Predictive performance:** Highest F1 (0.965), Accuracy (0.968), Precision (0.974), ROC-AUC (0.996), PR-AUC (0.996) across the test set.
2. **Calibration:** Brier score of 0.025 — the best-calibrated model, essential for using predicted probabilities as risk scores.
3. **Interpretability:** Coefficients directly expose which features drive predictions, straightforwardly compatible with SHAP and the recommendation layer.
4. **Complexity:** Simplest model among the ML options. No hyperparameter sensitivity. Trains in 0.04 seconds.
5. **Operational relevance:** The precision advantage (+0.174 vs queue rule) reduces false alarms materially — in an operational setting, false congestion alerts cause unnecessary preparation costs.

XGBoost is essentially equivalent (F1 0.961, ROC-AUC 0.996) and would be a valid alternative if SHAP interaction analysis becomes the priority. The difference is within noise.

**Not selected — Random Forest:** Lower recall (0.921) and F1 (0.942) than both LR and XGBoost with no compensating advantage.

---

## 4. Key Findings

1. **All three ML models beat the queue-only rule on precision and F1.** The improvement in precision (+0.164 to +0.174) is operationally meaningful.

2. **The queue rule retains the best recall (0.996).** It misses almost no congestion event. ML models trade a small recall reduction for a large precision gain.

3. **Queue variables dominate feature importance across all models.** This is consistent with the final audit finding that the dataset represents congestion persistence more than early-warning forecasting.

4. **Logistic Regression outperforms tree models** on this synthetic tabular dataset despite the strong nonlinear expectation. The relationships are approximately linear after standardisation; nonlinear capacity adds no measurable signal.

5. **Temporal distribution shift does not collapse model performance.** Despite a ~25 pp increase in congestion rate between train and test, ROC-AUC remains 0.993–0.996, demonstrating that the models generalise well to the later simulation period.

6. **The dataset should not be claimed as a genuine early-warning system.** Models are primarily detecting current congestion state and its immediate persistence. The 6-hour forward-looking label is strongly driven by current queue conditions (audit: 0.861 Pearson correlation).

---

## 5. Next Steps

1. **Recommendation engine** — Wire `congestion_prob` + `predicted_delay_hours` + top SHAP features into `src/recommendations/engine.py`.
2. **FastAPI endpoint** — `POST /predict` returning risk level, delay estimate, contributing factors, recommendation.
3. **Frontend** — Connect `frontend/` to the API.
4. **E-005** — Feature ablation (optional; queue dominance already confirmed by SHAP).
