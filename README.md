# Safiri PortPulse

AI-powered decision-support prototype for predicting **port congestion risk and shipment delays**.

Safiri PortPulse takes the current operational state of a port — including vessel traffic, berth capacity, queues, resources, weather, and historical conditions then and estimates:

- **Likelihood of congestion**
- **Expected delay over the next 6 hours**
- **Key factors contributing to the prediction**
- **Recommended operational actions**

This is designed as a **human-in-the-loop decision-support tool**. It provides predictions and recommendations, but does not make autonomous operational decisions.

---

## Problem

Port congestion can result from a combination of:

- High vessel traffic
- Limited berth capacity
- Growing queues
- Vessel waiting times
- Equipment constraints
- Labour availability
- Weather conditions
- Historical operating patterns
- Temporal traffic patterns

The goal of PortPulse is to combine these signals into an interpretable risk assessment that can support operational planning.

---

## How It Works

```text
Current Port Conditions
          │
          ▼
   Feature Processing
          │
          ├──────────────────┐
          ▼                  ▼
 Congestion Model       Delay Model
 Logistic Regression    Ridge Regression
          │                  │
          ▼                  ▼
 Congestion Probability  Expected Delay
          │                  │
          └────────┬─────────┘
                   ▼
             SHAP Explanation
                   │
                   ▼
          Risk Classification
                   │
                   ▼
       Operational Recommendations
                   │
                   ▼
            Human Decision
````

The prediction pipeline is separated from the recommendation layer:

* **Machine learning models** estimate congestion and delay.
* **SHAP** explains the model prediction.
* A **deterministic recommendation engine** converts the prediction and operational conditions into suggested actions.
* A human operator remains responsible for the final decision.

---

## Key Features

### Congestion Prediction

A Logistic Regression model estimates the probability that the port will be congested within the prediction horizon.

The model was selected after comparing:

* Logistic Regression
* Random Forest
* XGBoost
* A queue-based baseline

Logistic Regression was selected because it provided the best overall synthetic evaluation performance while remaining simple, fast, and interpretable.

### Delay Prediction

A Ridge Regression model estimates the expected shipment/port delay in hours over the next 6 hours.

Ridge was selected after comparing it with Random Forest and XGBoost.

### Explainability

The system uses **SHAP (SHapley Additive exPlanations)** to identify which features contributed most to each prediction.

For example, high congestion risk may be associated with:

* More vessels currently in port
* More anchored vessels
* Longer waiting times
* Higher queue pressure
* Higher capacity pressure

The explanation is generated for each prediction rather than being hardcoded.

### Operational Recommendations

The recommendation engine uses deterministic rules based on:

* Congestion probability
* Expected delay
* Queue pressure
* Berth utilization
* Available capacity

Recommendations are intentionally separated from the ML model so that operational actions remain transparent and testable.

---

## Data

The project uses a **synthetic port-operations dataset** designed specifically for this assignment.

### Dataset characteristics

* ~5,000 hourly port observations
* 20 synthetic ports
* Multiple operational and environmental features
* 6-hour prediction horizon
* No future outcome variables are used as prediction inputs

The simulated data includes:

* Vessel arrivals
* Vessels currently in port
* Anchored and approaching vessels
* Berth capacity and utilization
* Queue length
* Waiting time
* Crane and equipment conditions
* Labour availability
* Weather indicators
* Temporal features
* Historical operating patterns
* Traffic/capacity interactions

The dataset was generated using a fixed random seed to make experiments reproducible.

### Important note

Because the dataset is synthetic, the reported model performance demonstrates the feasibility of the methodology within the simulated environment. It **does not represent real-world port prediction accuracy**.

Real deployment would require validation using historical AIS and port-operational data.

---

## Model Evaluation

The congestion models were evaluated using a temporal train/test split rather than randomly shuffling observations.

### Congestion classification

| Model               |  Accuracy | Precision |    Recall |        F1 |   ROC-AUC |    PR-AUC |     Brier |
| ------------------- | --------: | --------: | --------: | --------: | --------: | --------: | --------: |
| Queue baseline      |     88.6% |     80.0% |     99.6% |     0.887 |     0.895 |     0.799 |     0.112 |
| Logistic Regression | **96.8%** | **97.4%** | **95.5%** | **0.965** | **0.996** | **0.996** | **0.025** |
| Random Forest       |     94.9% |     96.4% |     92.1% |     0.942 |     0.993 |     0.993 |     0.031 |
| XGBoost             |     96.5% |     96.8% |     95.5% |     0.961 |     0.996 |     0.995 |     0.027 |

Logistic Regression was selected because it achieved the strongest overall balance of predictive performance, calibration, simplicity, and interpretability on the synthetic evaluation data.

### Delay prediction

| Model            |         MAE |        RMSE |        R² |
| ---------------- | ----------: | ----------: | --------: |
| Ridge Regression | **0.483 h** | **0.627 h** | **0.990** |
| Random Forest    |     0.586 h |     0.791 h |     0.985 |
| XGBoost          |     0.516 h |     0.698 h |     0.988 |

Ridge Regression was selected because it achieved the lowest MAE and RMSE while remaining lightweight and interpretable.

---

## Probability Calibration

The congestion model's probability estimates were evaluated separately from classification accuracy.

Test-set calibration metrics:

* **Brier score:** 0.0248
* **Log Loss:** 0.0812
* **Expected Calibration Error (ECE):** 0.0207

These results indicate that the Logistic Regression probabilities were reasonably calibrated on the synthetic evaluation dataset.

Calibration quality should not be interpreted as real-world reliability. A production system would require recalibration and validation using historical data from the target ports.

---

## Explainability

SHAP is used to provide both global and individual explanations.

Typical high-risk predictions are dominated by operational queue and vessel-state variables such as:

* Average waiting time
* Queue pressure
* Vessels anchored
* Queue length
* Vessels currently in port
* Queue/capacity interaction

The system exposes these contributions through the API and presents them in a natural language in the frontend.

A positive SHAP contribution indicates that a feature pushed the model toward a higher predicted risk relative to the model's baseline. A negative contribution indicates a contribution toward lower predicted risk.

SHAP contributions should be interpreted as **model explanations, not causal effects**.

---

## Decision Support

PortPulse follows a human-in-the-loop workflow:

```text
                 PORT CONDITIONS
                       │
                       ▼
              FEATURE PROCESSING
                       │
              ┌────────┴────────┐
              ▼                 ▼
       CONGESTION MODEL     DELAY MODEL
       Logistic Regression  Ridge Regression
              │                 │
              ▼                 ▼
       Congestion            Expected
       Probability           Delay (hours)
              │                 │
              └────────┬────────┘
                       ▼
                 SHAP EXPLANATION
                       │
                       ▼
                 RISK ASSESSMENT
                       │
                       ▼
             RECOMMENDATION ENGINE
                       │
                       ▼
                  HUMAN REVIEW
                       │
                       ▼
               FINAL DECISION
```

For example:

### LOW

Normal monitoring with no immediate intervention.

### MEDIUM

Potential actions include:

* Pre-positioning resources
* Alerting the duty officer
* Increasing queue monitoring

### HIGH

Potential actions include:

* Immediate operational review
* Reviewing contingency procedures
* Evaluating alternate-port or anchorage options
* Notifying relevant port authorities

The system does **not** automatically execute these actions.

---

## Running Locally

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
cd safiri-port-pulse
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 4. Run the API

```bash
uvicorn backend.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

### 5. Open the frontend

Run:

```text
python -m http.server 5500 --directory frontend --bind 127.0.0.1
```

and the browser opens while the API is running.

The frontend communicates with the FastAPI backend and sends predictions to the real `/predict` endpoint.

---

## API

### Health check

```http
GET /health
```

Returns the availability of the prediction models and explainability components.

### Prediction

```http
POST /predict
```

Accepts the current operational state of a port and returns:

* Congestion probability
* Expected delay
* Risk level
* SHAP-based risk factors
* Protective factors
* Recommended actions
* Human-in-the-loop guidance

The API validates the input schema and rejects unsupported fields, including future target values.

---

## Reproducing the Pipeline

Synthetic data can be generated using:

```bash
python scripts/generate_data.py
```

Data validation:

```bash
python scripts/validate_data.py
```

Train congestion models:

```bash
python scripts/train.py
```

Train delay models:

```bash
python scripts/train_delay.py
```

Run calibration:

```bash
python scripts/calibrate.py
```

Run evaluation:

```bash
python scripts/evaluate.py
```

Additional experiment and analysis scripts are available under `scripts/`.

---

## Testing

Run the complete test suite with:

```bash
pytest
```

The project includes tests covering:

* Data generation
* Feature engineering
* Model behavior
* Delay prediction
* Calibration
* SHAP explanations
* Recommendation logic
* API validation and prediction behavior

---

## Project Structure

```text
safiri-port-pulse/
│
├── artifacts/
│   ├── metrics/          # Evaluation and experiment results
│   ├── models/           # Trained model artifacts
│   └── plots/            # Evaluation and explainability plots
│
├── backend/
│   ├── main.py           # FastAPI application
│   ├── schemas.py        # API request/response schemas
│   └── services/
│       └── prediction_service.py
│
├── data/
│   └── raw/
│       └── port_operations.csv
│
├── docs/                 # Design, assumptions, audits and experiment documentation
│
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── style.css
│
├── scripts/              # Data generation, training and evaluation commands
│
├── src/
│   ├── data/
│   ├── explainability/
│   ├── features/
│   ├── models/
│   └── recommendations/
│
└── tests/
```

---

## Limitations

This is a **3-day proof-of-concept**, not a production forecasting system.

### Synthetic data

The dataset is simulated rather than collected from real port operations or AIS data. Therefore, model performance cannot be interpreted as real-world accuracy.

### Queue persistence

The synthetic environment contains strong persistence between the current queue state and future congestion/delay. As a result, the current model is better described as **operational-state prediction / nowcasting** than as a fully validated early-warning forecasting system.

### Simplified weather

Weather is represented using synthetic severity and pressure indicators rather than real meteorological measurements.

### Real-world uncertainty

A production system would need to handle:

* Missing AIS messages
* Sensor errors
* Unexpected vessel arrivals
* Equipment failures
* Labour disruptions
* Weather forecast uncertainty
* Port-specific operating differences
* Distribution shift

---

## Future Work

A production-oriented version could incorporate:

1. Historical AIS vessel tracking data
2. Real-time port operational data
3. Real weather and forecast feeds
4. Explicit unexpected-event modeling
5. Port-specific model calibration
6. Better early-warning forecasting
7. Robust handling of missing and noisy observations
8. Monitoring for distribution shift and model degradation

---

## Documentation

Additional design and experiment documentation is available in [`docs/`](docs/):

* `ARCHITECTURE.md` — system architecture
* `ASSUMPTIONS.md` — modeling assumptions and limitations
* `DATA_DESIGN.md` — dataset and target design
* `DESIGN.md` — overall technical design
* `DESIGN_DECISIONS.md` — major engineering decisions
* `EXPERIMENTS.md` — experiment history
* `EXPERIMENT_REPORT.md` — detailed experiment results

---

## Disclaimer

Safiri PortPulse is an internship take-home prototype developed for demonstrating an interpretable AI approach to port congestion and delay decision support.

Predictions and recommendations are intended to support human decision-making and should not be treated as autonomous operational instructions.