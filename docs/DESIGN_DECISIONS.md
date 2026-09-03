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
