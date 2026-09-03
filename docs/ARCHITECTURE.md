# Safiri PortPulse — Architecture

## 1. System Overview

Safiri PortPulse is an AI-powered decision-support system that processes port operational data, predicts congestion and delay risk, explains the prediction, and provides an operational recommendation.

```text
                    ┌──────────────────┐
                    │  Synthetic Data  │
                    │     Generator     │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ Data Validation  │
                    │ & Preprocessing  │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ Feature          │
                    │ Engineering      │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ ML Prediction    │
                    │                  │
                    │ Congestion       │
                    │ + Delay          │
                    └────────┬─────────┘
                             ↓
               ┌─────────────┴─────────────┐
               ↓                           ↓
      ┌──────────────────┐       ┌──────────────────┐
      │   Explainability │       │ Confidence /     │
      │      (SHAP)      │       │ Calibration      │
      └────────┬─────────┘       └────────┬─────────┘
               └─────────────┬─────────────┘
                             ↓
                    ┌──────────────────┐
                    │ Recommendation   │
                    │     Engine       │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │    FastAPI       │
                    │      API         │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │   Web Frontend   │
                    └──────────────────┘
```

---

## 2. Data Layer

### Data Generator

**Location:** `src/data/generator.py`

Generates reproducible synthetic port-operation data according to `DATA_DESIGN.md`.

### Validator

**Location:** `src/data/validator.py`

Checks generated data for invalid values, missing data, duplicates, and impossible operational conditions.

### Preprocessing

**Location:** `src/data/preprocessing.py`

Handles categorical encoding, numerical preprocessing, and preparation of data for model training.

---

## 3. Feature Layer

**Location:** `src/features/engineering.py`

Transforms operational data into model-ready features.

Examples include:

* Traffic pressure
* Capacity pressure
* Queue pressure
* Weather pressure
* Arrival density
* Queue-capacity interaction
* Traffic-weather interaction

Feature engineering must use only prediction-time information.

---

## 4. Model Layer

### Congestion Classification

**Location:** `src/models/congestion.py`

Predicts the probability of congestion during the next 6 hours.

Candidate models:

* Logistic Regression
* Random Forest
* XGBoost

The final model will be selected through experimentation.

### Delay Prediction

**Location:** `src/models/delay.py`

Estimates expected future delay in hours.

### Baseline

**Location:** `src/models/baseline.py`

Provides the baseline implementation used for comparison.

### Calibration

**Location:** `src/models/calibration.py`

Evaluates and, where necessary, calibrates predicted probabilities so that reported risk better reflects observed outcomes.

### Evaluation

**Location:** `src/models/evaluation.py`

Calculates classification and regression metrics and generates evaluation results.

---

## 5. Explainability Layer

**Location:** `src/explainability/`

SHAP is used to identify which features contributed most to an individual prediction.

The output is converted into human-readable explanations such as:

> High arrival rate and limited berth availability are increasing congestion risk.

---

## 6. Recommendation Layer

**Location:** `src/recommendations/engine.py`

Converts prediction results and operational indicators into deterministic recommendations.

Example:

```text
High congestion risk
        +
High queue pressure
        +
Limited berth capacity
        ↓
Prepare for increased vessel waiting
and consider operational rescheduling.
```

The recommendation engine does not make autonomous decisions.

---

## 7. API Layer

**Location:** `backend/`

FastAPI provides the interface between the trained models and the frontend.

### Primary endpoint

```text
POST /predict
```

The endpoint accepts current port operational conditions and returns:

* Congestion probability
* Risk level
* Expected delay
* Confidence
* Key contributing factors
* Operational recommendation

---

## 8. Frontend

**Location:** `frontend/`

A lightweight HTML/CSS/JavaScript interface will allow a user to:

1. Enter or modify operational conditions.
2. Request a prediction.
3. View congestion risk.
4. View expected delay.
5. Understand the main contributing factors.
6. View the recommended action.

The frontend is intentionally simple because the primary focus is the ML and decision-support pipeline.

---

## 9. Model Artifact Flow

Trained models and supporting artifacts are stored under:

```text
artifacts/
├── models/
├── metrics/
└── plots/
```

This separates generated model outputs from source code and data.

---

## 10. Design Principle

The system follows a **human-in-the-loop decision-support architecture**:

```text
AI Prediction
     ↓
Explanation
     ↓
Recommendation
     ↓
Human Decision
```

The system assists an operator rather than replacing the operator.
