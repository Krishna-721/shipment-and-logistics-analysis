# Safiri PortPulse — Data Design

## 1. Purpose

This document defines the synthetic data used by this project - Safiri PortPulse.

The dataset is designed to simulate realistic port operations and create meaningful relationships between vessel traffic, port capacity, queues, weather, operational resources, delays, and congestion.

The generator must follow this document rather than generating independent random values.

---

## 2. Observation Unit

Each row represents the operational state of **one port at one point in time**.

Example:

```text
timestamp = 2026-01-15 14:00
port_id   = PORT_05
```

The dataset will contain approximately **5,000 port-hour observations** across multiple synthetic ports.

---

## 3. Prediction Target

The system predicts whether the port will experience significant congestion during the **following 6 hours**.

### Classification target

`congestion_label`

* `0` → No significant congestion
* `1` → Significant congestion

The label must be derived from simulated future operational outcomes, not directly from a simple threshold on an input feature.

### Delay target

`future_delay_hours`

Represents the expected operational delay associated with the following period.

---

## 4. Prediction-Time Features

Only information available at time `t` may be used to make the prediction.

Examples:

* Current vessel count
* Recent arrival rate
* Current queue
* Available berths
* Berth utilization
* Port throughput
* Weather conditions
* Equipment availability
* Labor availability
* Historical performance
* Temporal features

Future queue length, future waiting time, future delay, and future congestion state must **not** be used as input features.

This prevents target leakage.

---

## 5. Feature Groups

### 5.1 Port & Capacity

| Feature                  | Unit        |
| ------------------------ | ----------- |
| `port_id`                | Identifier  |
| `total_berths`           | Count       |
| `available_berths`       | Count       |
| `berth_utilization`      | %           |
| `daily_vessel_capacity`  | Vessels/day |
| `daily_teu_capacity`     | TEU/day     |
| `current_throughput_teu` | TEU         |
| `capacity_headroom`      | Ratio       |

---

### 5.2 Vessel Traffic

| Feature                     | Unit         |
| --------------------------- | ------------ |
| `vessels_currently_in_port` | Count        |
| `vessels_anchored`          | Count        |
| `vessels_approaching`       | Count        |
| `arrivals_last_1h`          | Count        |
| `arrivals_last_6h`          | Count        |
| `arrivals_last_24h`         | Count        |
| `departures_last_6h`        | Count        |
| `arrival_rate`              | Vessels/hour |
| `traffic_growth_rate`       | Ratio        |

---

### 5.3 Queue & Waiting

| Feature                       | Unit    |
| ----------------------------- | ------- |
| `queue_length`                | Vessels |
| `anchorage_vessel_count`      | Vessels |
| `avg_waiting_time_hours`      | Hours   |
| `max_waiting_time_hours`      | Hours   |
| `oldest_waiting_vessel_hours` | Hours   |
| `queue_growth_rate`           | Ratio   |

---

### 5.4 Cargo & Workload

| Feature                        | Unit            |
| ------------------------------ | --------------- |
| `vessel_type`                  | Category        |
| `cargo_type`                   | Category        |
| `cargo_volume_teu`             | TEU             |
| `cargo_weight_tons`            | Tons            |
| `estimated_moves`              | Container moves |
| `estimated_service_time_hours` | Hours           |

---

### 5.5 Equipment & Labor

| Feature                   | Unit       |
| ------------------------- | ---------- |
| `cranes_available`        | Count      |
| `cranes_operational`      | Count      |
| `crane_utilization`       | %          |
| `equipment_failure_count` | Count      |
| `avg_crane_productivity`  | Moves/hour |
| `labor_availability_pct`  | %          |
| `labor_disruption_flag`   | 0/1        |

---

### 5.6 Weather

| Feature            | Unit       |
| ------------------ | ---------- |
| `wind_speed_knots` | Knots      |
| `wave_height_m`    | Metres     |
| `visibility_km`    | Kilometres |
| `weather_severity` | 0–1        |
| `storm_flag`       | 0/1        |

---

### 5.7 Temporal & Historical Features

| Feature                          | Unit  |
| -------------------------------- | ----- |
| `hour`                           | 0–23  |
| `day_of_week`                    | 0–6   |
| `month`                          | 1–12  |
| `is_weekend`                     | 0/1   |
| `is_holiday`                     | 0/1   |
| `historical_avg_wait_time`       | Hours |
| `historical_avg_turnaround_time` | Hours |
| `historical_delay_rate`          | %     |
| `historical_congestion_rate`     | %     |

---

## 6. Derived Features

The following features will capture operational pressure:

* `traffic_pressure`
* `capacity_pressure`
* `queue_pressure`
* `equipment_pressure`
* `weather_pressure`
* `historical_pressure`
* `arrival_density`
* `queue_capacity_interaction`
* `traffic_weather_interaction`

These features should be calculated from the underlying operational variables rather than independently generated.

---

## 7. Synthetic Data Generation

The generator will follow a simplified port-operation simulation.

```text
Port Profile
     ↓
Vessel Arrivals
     ↓
Available Capacity
     ↓
Weather + Equipment + Labor
     ↓
Effective Service Capacity
     ↓
Queue Formation
     ↓
Waiting Time
     ↓
Delay
     ↓
Future Congestion
```

### Port profiles

Each synthetic port will have different characteristics such as:

* Number of berths
* Vessel handling capacity
* Typical traffic level
* Typical service rate
* Historical congestion tendency

This creates variation between ports.

### Vessel arrivals

Arrival demand will depend on:

* Port baseline traffic
* Hour of day
* Day of week
* Seasonal variation
* Random variation

Busy periods should naturally produce higher arrival volumes.

### Effective service capacity

Service capacity will be affected by:

* Available berths
* Equipment availability
* Labor availability
* Weather
* Vessel workload

Poor weather, equipment failures, or labor shortages should reduce effective service capacity.

### Queue formation

When incoming demand exceeds effective service capacity, the queue should increase.

When service capacity exceeds demand, the queue should gradually decrease.

### Delay

Delay should increase primarily when:

* Queue pressure increases
* Waiting time increases
* Capacity becomes constrained
* Service capacity decreases

Random noise should be added so that the relationship is not perfectly deterministic.

---

## 8. Ground Truth

The congestion label represents a **future operational state**.

A port is considered congested when the simulated future operating conditions indicate sustained operational pressure, such as elevated queueing, waiting time, and insufficient service capacity.

The exact threshold combination will be defined in the generator implementation and documented in the experiment results.

The important constraint is:

> Current input features influence future congestion, but the congestion label must not simply be a direct threshold of one current feature.

---

## 9. Realism & Noise

The generator should include controlled randomness to represent:

* Unpredictable vessel arrivals
* Weather variability
* Operational disruptions
* Equipment failures
* Variations in vessel workload
* Measurement noise

Random seeds will be fixed to make experiments reproducible.

---

## 10. Data Validation

Generated data must be checked for:

* Missing values
* Invalid ranges
* Negative physical quantities
* Impossible capacity relationships
* Invalid categorical values
* Duplicate observations
* Temporal ordering

Validation will be implemented in `src/data/validator.py`.

---

## 11. Leakage Prevention

The following must never be model inputs:

* Future queue length
* Future waiting time
* Future delay
* Future congestion label
* Any variable calculated using future observations

Feature engineering must only use information available at the prediction timestamp.

---

## 12. Dataset Output

The generator will produce:

```text
data/raw/port_operations.csv
```

The processed dataset used for training will be stored under:

```text
data/processed/
```

The generation process must use a fixed random seed and be reproducible through:

```text
python scripts/generate_data.py
```
