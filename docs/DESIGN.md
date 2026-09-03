# Safiri PortPulse — System Design

## 1. Problem

Port congestion and shipment delays are affected by vessel traffic, available capacity, queues, weather, equipment, and historical patterns.

**Safiri PortPulse** is an AI-powered decision-support system that estimates congestion and delay risk and explains the main factors influencing the prediction.

The system follows:

**Operational Data → Prediction → Explanation → Recommendation → Human Decision**

---

## 2. Objectives

The system will:

* Predict the **probability of port congestion**.
* Estimate **expected delay**.
* Provide a **confidence score**.
* Identify the **key contributing factors**.
* Provide an **actionable operational recommendation**.
* Present the results through a simple web interface.

---

## 3. Data

A synthetic dataset will be generated to simulate realistic port operations.

Each record represents the state of a port at a specific point in time.

The simulation will model relationships between:

* Vessel arrivals and traffic density
* Port and berth capacity
* Queue and anchorage conditions
* Weather
* Equipment and labor availability
* Historical port performance
* Temporal patterns

The congestion and delay outcomes will emerge from these operational relationships.

Detailed columns, ranges, relationships, and ground-truth generation are defined in `DATA_DESIGN.md`.

---

## 4. Prediction

The primary prediction is:

**Congestion probability over the next 6 hours.**

A separate model will estimate expected delay.

Only information available at prediction time will be used as model input to prevent data leakage.

---

## 5. Modeling

The system will compare:

* **Logistic Regression** — baseline
* **Random Forest** — primary model

A regression model will be used for delay estimation.

Model selection will be based on validation performance and interpretability.

---

## 6. Explainability

**SHAP** will be used to explain individual predictions.

The system will identify factors such as:

> High vessel arrivals + limited berth capacity → increased congestion risk.

This converts model output into understandable operational reasoning.

---

## 7. Recommendation Layer

A deterministic rule-based layer will convert predictions into operational guidance.

Examples:

* Monitor port conditions
* Prepare for increased queueing
* Consider delaying or rescheduling arrivals
* Prioritize operational resources

The system provides **decision support**, not autonomous decisions.

---

## 8. Architecture

```text
Synthetic Data
      ↓
Validation & Feature Engineering
      ↓
ML Models
      ↓
Prediction + Confidence
      ↓
SHAP Explanation
      ↓
Recommendation Engine
      ↓
FastAPI
      ↓
Web Interface
```

---

## 9. Evaluation

Classification will be evaluated using:

* Precision
* Recall
* F1-score
* ROC-AUC
* PR-AUC
* Brier Score

Delay prediction will use:

* MAE
* RMSE
* R²

A time-aware evaluation split will be used where appropriate.

---

## 10. Scope

This project is a **3-day proof-of-concept** focused on an end-to-end, interpretable decision-support workflow.

It does not attempt to implement:

* Real-time AIS infrastructure
* Live weather APIs
* Large-scale streaming systems
* Route optimization
* Autonomous decision-making
* Production deployment
