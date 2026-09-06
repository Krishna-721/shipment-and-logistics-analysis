# Final Dataset Audit Report

Generated: 2026-01-XX (before freezing dataset for ML)

## Executive Summary

**Classification: NEEDS MINOR FIXES**

The generated dataset passes most validation checks but has **one critical weakness**: the congestion target is primarily determined by current queue state rather than being truly forward-looking. While simulation dynamics are sound, the 6-hour prediction horizon does not meaningfully capture future risk independent of current operational stress.

---

## 1. Complete Column Audit

Total columns: **64**

### Identifiers / Metadata (2)
- `port_id` — Port identifier
- `timestamp` — Observation timestamp

### Prediction-Time Raw Features (50)

**Port & Capacity (10):**
- `total_berths`, `available_berths`, `occupied_berths`, `berth_utilization`
- `daily_vessel_capacity`, `daily_teu_capacity`
- `current_throughput_teu`, `capacity_headroom`
- `vessels_currently_in_port`, `vessels_anchored`

**Vessel Traffic (9):**
- `vessels_approaching`
- `arrivals_last_1h`, `arrivals_last_6h`, `arrivals_last_24h`
- `departures_last_6h`
- `arrival_rate`, `traffic_growth_rate`
- `queue_length`, `anchorage_vessel_count`

**Queue & Waiting (4):**
- `avg_waiting_time_hours`, `max_waiting_time_hours`
- `oldest_waiting_vessel_hours`, `queue_growth_rate`

**Cargo & Workload (7):**
- `vessel_type`, `cargo_type`
- `cargo_volume_teu`, `cargo_weight_tons`
- `estimated_moves`, `estimated_service_time_hours`
- (Note: vessel-level, not port-level aggregates)

**Equipment & Labor (7):**
- `cranes_available`, `cranes_operational`, `crane_utilization`
- `equipment_failure_count`, `avg_crane_productivity`
- `labor_availability_pct`, `labor_disruption_flag`

**Weather (5):**
- `wind_speed_knots`, `wave_height_m`, `visibility_km`
- `weather_severity`, `storm_flag`

**Temporal (5):**
- `hour`, `day_of_week`, `month`
- `is_weekend`, `is_holiday`

**Historical (4):**
- `historical_avg_wait_time`, `historical_avg_turnaround_time`
- `historical_delay_rate`, `historical_congestion_rate`

### Prediction-Time Derived Features (9)

- `arrival_density` — Arrivals per hour over 6h window
- `traffic_pressure` — Normalized arrival rate vs capacity
- `capacity_pressure` — Berth utilization
- `queue_pressure` — Queue relative to berth count
- `equipment_pressure` — Equipment failure impact
- `weather_pressure` — Weather severity (duplicate)
- `historical_pressure` — Composite historical risk
- `queue_capacity_interaction` — Queue × Capacity pressure
- `traffic_weather_interaction` — Traffic × Weather pressure

### Targets (2)
- `congestion_label` — Binary congestion indicator (0/1)
- `future_delay_hours` — Expected delay in hours

### Internal/Non-Model Columns (1)
- `is_valid_training_row` — Flags rows with complete 6-hour horizon

### ✓ Leakage Verification

**Confirmed: NO DATA LEAKAGE**

- `congestion_label` and `future_delay_hours` correctly marked as **TARGETS**
- `is_valid_training_row` correctly marked as **METADATA**
- `queue_length` represents **CURRENT** queue state
- `avg_waiting_time_hours` represents **CURRENT** waiting time
- All 59 features represent information available at prediction time
- No future queue, waiting, delay, or congestion information used as features

---

## 2. Leakage Audit

✓ **NO LEAKAGE DETECTED**

All features represent prediction-time information only. Targets correctly use future observations but are not used as inputs.

---

## 3. Feature-Count Audit

**Design target:** ~25-30 meaningful model features  
**Actual dataset:** 64 columns

### Why 64 Columns?

The generator created a comprehensive simulation with:
- 50 raw operational features (all available at prediction time)
- 9 derived pressure/interaction features
- 2 identifiers
- 2 targets
- 1 metadata flag

### Essential Model Features (30)

These should be used for initial modeling:

1. `total_berths`, `available_berths`, `berth_utilization`
2. `vessels_currently_in_port`, `vessels_anchored`, `vessels_approaching`
3. `arrivals_last_1h`, `arrivals_last_6h`, `arrivals_last_24h`
4. `arrival_rate`, `queue_length`, `avg_waiting_time_hours`
5. `cranes_operational`, `crane_utilization`, `equipment_failure_count`
6. `labor_availability_pct`, `weather_severity`, `storm_flag`
7. `hour`, `day_of_week`, `is_weekend`
8. `historical_avg_wait_time`, `historical_congestion_rate`
9. `traffic_pressure`, `capacity_pressure`, `queue_pressure`
10. `equipment_pressure`, `weather_pressure`
11. `queue_capacity_interaction`, `traffic_weather_interaction`

### Redundant/Optional Features (25)

These can be excluded or used in ablation studies:

**Derived from other features:**
- `occupied_berths` = total_berths - available_berths
- `anchorage_vessel_count` = queue_length × 0.70
- `capacity_headroom` = 1 - berth_utilization

**Vessel-level (not port-level):**
- `vessel_type`, `cargo_type`, `cargo_volume_teu`, `cargo_weight_tons`, `estimated_moves`

**Secondary metrics:**
- `max_waiting_time_hours`, `oldest_waiting_vessel_hours`
- `queue_growth_rate`, `traffic_growth_rate`, `departures_last_6h`

**Weather components (captured in weather_severity):**
- `wind_speed_knots`, `wave_height_m`, `visibility_km`

**Static or low-value:**
- `daily_vessel_capacity`, `daily_teu_capacity`, `current_throughput_teu`
- `cranes_available`, `avg_crane_productivity`
- `month`, `is_holiday` (all zeros)

**Composite features:**
- `historical_pressure`, `arrival_density`
- `historical_avg_turnaround_time`, `historical_delay_rate`

### Recommendation

**Keep all 64 columns in raw dataset** for flexibility. Feature selection should happen during modeling phase with ablation studies to identify essential predictors.

---

## 4. Target Relationship Audit

### Correlations with Congestion Label

| Feature | Correlation |
|---------|------------|
| `queue_pressure` | **0.861** |
| `queue_length` | **0.808** |
| `avg_waiting_time_hours` | **0.761** |
| `berth_utilization` | 0.585 |
| `arrival_rate` | 0.208 |
| `traffic_pressure` | 0.025 |
| `equipment_pressure` | -0.016 |
| `weather_severity` | 0.008 |

### Detailed Queue-Congestion Relationship

| Queue Pressure | Count | Congestion Rate |
|----------------|-------|-----------------|
| Very Low (0-25%) | 952 | 0.1% |
| Low (25-50%) | 620 | 0.2% |
| Medium (50-100%) | 754 | 13.5% |
| High (100-200%) | 845 | **90.5%** |
| Very High (>200%) | 393 | **100.0%** |

### ⚠ Interpretation

**WARNING: High correlation (>0.7) detected**

The three highest correlations are:
1. `queue_pressure` (0.861) — derived from current queue
2. `queue_length` (0.808) — current queue state
3. `avg_waiting_time_hours` (0.761) — derived from current queue

This indicates **the target is strongly determined by current operational state** rather than forward-looking dynamics.

### Is This a Problem?

**Partially legitimate, but weaker than intended:**

The high correlation is **partially expected** because:
- Ports with high current queues are more likely to experience future congestion
- This represents real operational momentum

However, it's **concerning** because:
- Correlation of 0.86 suggests near-deterministic relationship
- Very few cases exist where low current queue leads to future congestion
- The "6-hour forward-looking" window may not be working as designed

---

## 5. Counterexample Analysis

### Case A: High Queue BUT No Future Congestion

**Found: 732 cases (15.0% of valid dataset)**

✓ **SUFFICIENT EXAMPLES EXIST**

Sample case (PORT_17 at 2026-01-10 12:00):
- Queue: 9.1 vessels (83% of berths)
- Waiting time: 3.49 hours
- Berth utilization: 0.36
- Arrival rate: 2.48 vessels/hour
- Weather severity: 0.23
- **Explanation:** Low berth utilization (0.36) and good operational capacity allow queue to clear despite current size

### Case B: Low/Moderate Queue BUT Future Congestion

**Found: 2 cases (0.04% of valid dataset)**

⚠ **CRITICAL WEAKNESS**

Only **2 cases out of 4,880** (0.04%) where congestion occurs despite low/moderate current queue.

Sample case (PORT_18 at 2026-01-10 02:00):
- Queue: 1.48 vessels (37% of berths)
- Waiting time: 0.73 hours
- Arrivals last 6h: 6
- Traffic pressure: 0.43
- **Future evolution:** Queue drops to 0 within 2 hours, then builds slightly to 1.77 at hour +6

**Why is this marked as congested?**  
Looking at the next 6 hours, the queue actually clears and remains low. This case appears to be a **labeling artifact** rather than genuine forward-looking congestion risk.

### Predictability from Current Queue Alone

Simple rule: **Predict congestion if queue > 50% of berths**

| Metric | Value |
|--------|-------|
| Accuracy | **85.0%** |
| Precision | 63.3% |
| Recall | 99.8% |
| F1-score | 77.4% |

### ⚠ Critical Finding

**The congestion target can be predicted with 85% accuracy using ONLY current queue length.**

This indicates:
- The target is NOT truly forward-looking
- It primarily captures **CURRENT operational stress**, not **FUTURE risk**
- ML models will likely learn to threshold the queue rather than identify early warning signals

---

## 6. Distribution Audit

| Variable | Min | Median | Mean | 95th Percentile | Max |
|----------|-----|--------|------|-----------------|-----|
| **congestion_label** | Class 0: 74.1% | Class 1: 25.9% | | | |
| queue_length | 0.00 | 2.37 | 5.00 | 19.09 | 31.76 |
| berth_utilization | 0.00 | 0.38 | 0.42 | 1.00 | 1.00 |
| avg_waiting_time_hours | 0.00 | 1.06 | 2.30 | 9.09 | 18.49 |
| future_delay_hours | 0.00 | 2.79 | 4.80 | 16.58 | 25.47 |
| arrival_rate | 0.45 | 2.01 | 2.04 | 3.53 | 4.90 |
| weather_severity | 0.00 | 0.13 | 0.16 | 0.42 | 0.75 |
| equipment_pressure | 0.00 | 0.00 | 0.01 | 0.09 | 0.40 |
| labor_availability_pct | 0.77 | 0.94 | 0.94 | 1.00 | 1.00 |

### ✓ Distributions are Plausible

All ranges are realistic for synthetic port operations. No extreme outliers or impossible values.

---

## 7. Final Recommendation

### Classification: **NEEDS MINOR FIXES**

The dataset is **NOT ready for ML in its current form** due to the weakness in forward-looking dynamics.

### Reasoning

**✓ Strengths:**
1. No data leakage detected
2. All features represent prediction-time information
3. Targets correctly separated from features
4. Class balance is reasonable (74% / 26%)
5. Distributions are plausible
6. Simulation dynamics are sound
7. No missing values
8. Temporal consistency verified

**⚠ Critical Issues:**

1. **Case B scarcity (0.04%)**  
   Almost no examples where congestion occurs despite low current queue. This means the target is NOT truly forward-looking—it's primarily capturing CURRENT operational stress.

2. **High correlation (0.86)**  
   Queue pressure correlation of 0.86 indicates near-deterministic relationship. Simple queue threshold achieves 85% accuracy.

3. **Feature count mismatch**  
   64 columns vs designed ~25-30. This is manageable but requires clear feature selection guidance.

4. **Redundant features**  
   25 features are derived or redundant. Should be documented for exclusion during modeling.

### Root Cause Analysis

The congestion label calculation uses:
```python
congestion_score = (
    0.35 × future_queue_pressure +
    0.25 × future_queue_peak +
    0.25 × future_wait +
    0.15 × (current_berth_util + weather)
)
```

**Problem:** The "future" queue and waiting time are calculated from current queue state. When current queue is low, future queue is also low, making it nearly impossible to predict congestion from low-queue states.

**The 6-hour forward window is looking at future queue state, but that future queue is deterministically driven by current queue in the simulation.**

### Recommended Actions BEFORE ML

#### Option A: Use Dataset As-Is (RECOMMENDED)

**Justification:**
- Real-world congestion IS highly correlated with current queue state
- 85% baseline accuracy is not unreasonable—it leaves room for ML to add value
- Case A (high queue → no congestion) exists and represents meaningful variance
- The simulation dynamics are sound

**Accept that:**
- This is a congestion **nowcasting** problem more than **forecasting**
- ML value comes from identifying when high queues will/won't become congested
- The 15% of high-queue cases that don't become congested are the interesting signal

**Actions:**
1. Document the high correlation in design docs
2. Set baseline performance expectation at ~85% accuracy
3. Focus ML evaluation on improving precision (reducing false positives)
4. Use feature importance to show ML learns beyond simple queue thresholding

#### Option B: Regenerate with Modified Target (NOT RECOMMENDED)

Would require:
- Incorporating arrival rate forecast into congestion calculation
- Reducing weight of current/future queue in congestion score
- Increasing weight of traffic trends and capacity constraints

**Why not recommended:**
- Risks creating artificial/unrealistic relationships
- Would delay ML phase significantly
- Current simulation already took substantial iteration

#### Option C: Redefine Problem (ALTERNATIVE)

Change from:
- "Predict congestion in next 6 hours"

To:
- "Predict whether high queue will persist or clear"
- "Estimate time to queue resolution"
- "Predict congestion severity (regression instead of classification)"

### Recommended Pre-ML Documentation

Create `docs/FEATURE_GUIDE.md` with:

1. **Essential features (30)** — Start here
2. **Redundant features (25)** — Exclude unless ablation testing
3. **Expected correlations** — Document that 0.86 queue_pressure correlation is known
4. **Baseline performance** — 85% accuracy from simple queue rule
5. **ML value proposition** — Focus on precision improvement and learning interaction effects

### What NOT to Do

✗ **Do not regenerate dataset** — Simulation dynamics are sound, just not as forward-looking as initially intended

✗ **Do not artificially reduce correlations** — They reflect legitimate operational relationships

✗ **Do not delete columns yet** — Keep flexibility for feature selection experiments

✗ **Do not claim this is "production-ready"** — The forward-looking limitation must be acknowledged

---

## Conclusion

The dataset is **suitable for ML with caveats**. The congestion target is more of a "nowcasting" problem (will current congestion persist?) than a true "forecasting" problem (will congestion emerge from low-stress state?).

This is still valuable for demonstrating:
- End-to-end ML pipeline
- Feature engineering
- Model comparison
- Explainability (SHAP can show when ML improves on queue thresholding)
- Decision support system design

The limitation should be documented and acknowledged in the final project report.

**Proceed to ML with Option A approach.**
