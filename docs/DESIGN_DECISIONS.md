# Safiri PortPulse — Design Decisions

## 01 - Use Synthetic Data

**Decision:** Use a generated synthetic dataset instead of live AIS/port data.

**Reason:** The project has a 3-day implementation constraint. Synthetic data allows us to control the operational relationships, generate sufficient observations, and create reproducible experiments without depending on external data pipelines.

---

## 02 - Model Port Operations as a System

**Decision:** Generate operational variables through relationships rather than independently sampling every column.

**Reason:** Congestion should emerge from realistic interactions between vessel arrivals, capacity, service rate, queues, weather, and operational resources.

---

## 03 - Use a Multi-Factor Congestion Target

**Decision:** Define congestion using future operational outcomes rather than a direct threshold on a single input feature.

**Reason:** A single rule such as `queue_length > X` would make the prediction task trivial and introduce an unrealistic relationship between features and target.

---

## 04 - Six-Hour Prediction Horizon

**Decision:** Predict congestion risk for the following 6 hours.

**Reason:** A defined prediction horizon makes the problem operationally meaningful and prevents ambiguity about what "future congestion" means.

---

## 05 - Prevent Target Leakage

**Decision:** Only information available at prediction time can be used as model input.

**Reason:** Future queue, delay, waiting time, or congestion information would artificially improve model performance and would not be available in a real prediction scenario.

---

## 06 - Compare Multiple Classification Models

**Decision:** Evaluate Logistic Regression, Random Forest, and XGBoost rather than committing to one model beforehand.

**Reason:** Logistic Regression provides an interpretable baseline, while Random Forest and XGBoost can capture nonlinear relationships and feature interactions. The final model will be selected using predictive performance, calibration, interpretability, and complexity.

---

## 07 - Separate Congestion and Delay Prediction

**Decision:** Treat congestion classification and delay estimation as separate prediction tasks.

**Reason:** Congestion is a categorical risk state, while delay is a continuous quantity. Separate models allow each task to be evaluated using appropriate metrics.

---

## 08 - Use SHAP for Explainability

**Decision:** Use SHAP to explain individual model predictions.

**Reason:** The assignment requires understandable contributing factors. SHAP provides feature-level attribution that can be converted into operational explanations.

---

## 09 - Deterministic Recommendation Layer

**Decision:** Use a rule-based recommendation engine rather than an LLM for operational recommendations.

**Reason:** Recommendations should be predictable, testable, reproducible, and easy to justify within a short project. The system is intended to support human decisions rather than make autonomous decisions.

---

## 10 - FastAPI Backend

**Decision:** Use FastAPI for the prediction API.

**Reason:** It provides a lightweight interface between the trained ML pipeline and the frontend while keeping the implementation simple.

---

## 11 - Lightweight Frontend

**Decision:** Use HTML, CSS, and JavaScript for the initial frontend.

**Reason:** The project evaluation focuses primarily on the prediction and decision-support workflow. A lightweight interface is sufficient for demonstrating the system.

---

## 12 - Separate ML Code from API Code

**Decision:** Keep `src/` and `backend/` as separate layers.

**Reason:** `src/` contains data, feature engineering, models, explainability, and recommendation logic, while `backend/` exposes that functionality through an API. This separation keeps the architecture modular and easier to test.

---

## 13 - Reproducible Experiments

**Decision:** Use fixed random seeds and script-based pipelines.

**Reason:** The same dataset generation and training process should produce reproducible results, making experiments easier to compare and document.

---

## 14 - Human-in-the-Loop

**Decision:** Treat Safiri PortPulse as a decision-support system rather than an autonomous decision-maker.

**Reason:** Predictions and recommendations should assist an operator while leaving the final operational decision to a human.

---

## 15 - Temporal Train/Test Split

**Decision:** Split the dataset chronologically — training on the first 8 days (2026-01-01 to 2026-01-08) and evaluating on the final 2.1 days (2026-01-09 to 2026-01-11).

**Reason:** Randomly shuffling a time-series dataset would allow future information to leak into the training set via observations that are temporally adjacent. A strict chronological cut preserves the direction of causality and better approximates a real deployment scenario where the model is trained on historical data and evaluated on unseen future observations.

**Known distributional shift:** Congestion rate rises from 20.6 % in training to 45.2 % in the test set, reflecting queue build-up over the simulation window. This shift is documented rather than masked. All three ML models maintain F1 ≥ 0.942 and ROC-AUC ≥ 0.993 despite this shift.

---

## 16 - Queue-Only Baseline Before Any ML

**Decision:** Implement and evaluate a simple operational rule (queue > 50 % of berths → predict congestion) as a formal baseline, evaluated on the same test set as all ML models.

**Reason:** The final dataset audit found a Pearson correlation of 0.861 between queue pressure and the congestion label. A single-feature threshold achieves 85 % accuracy on the full dataset. Any ML model that does not clearly exceed this baseline would provide no operational value and should not be deployed. The baseline makes the ML value proposition explicit and testable.

---

## 17 - Logistic Regression as Selected Model

**Decision:** Select Logistic Regression as the primary model for the congestion prediction pipeline.

**Reason:** Logistic Regression achieved the best performance across the test set:
- Highest F1 (0.965), Accuracy (0.968), Precision (0.974), ROC-AUC (0.996), PR-AUC (0.996), Brier Score (0.025).
- Simplest model in the comparison — trains in under 0.05 seconds.
- Best probability calibration (Brier 0.025), essential for using its output as an operational risk score.
- Coefficients are directly interpretable and compatible with SHAP.
- Precision improvement over the queue rule (+0.174) reduces false operational alerts from 20 % to 2 % — the most important operational metric.

XGBoost is functionally equivalent (F1 0.961, ROC-AUC 0.996) and could be substituted if SHAP interaction values become the priority. Random Forest underperforms both alternatives.

---

## 18 - Acknowledge Nowcasting Limitation

**Decision:** Explicitly document that the system primarily performs congestion persistence detection ("nowcasting") rather than early-warning forecasting, and not claim early-warning capability that is not supported by the data.

**Reason:** The dataset audit established that only 2 of 4,880 valid rows show low current queue leading to future congestion. The congestion label is strongly driven by current operational state. The system is valuable for reducing false alarms and providing calibrated risk probabilities, but should not be described as detecting congestion risk before any operational stress is visible. This limitation is consistent with the 3-day proof-of-concept scope.
