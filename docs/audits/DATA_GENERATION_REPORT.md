# Data Generation Report

## Summary

Successfully improved the synthetic port-operation data generator to correctly implement the documented design requirements.

## Files Changed

### Modified Files

1. **`src/data/generator.py`**
   - Fixed service capacity calculation (resource factors now applied once, not twice)
   - Fixed RuntimeWarning from empty `previous_arrivals` at simulation start
   - Implemented proper 6-hour forward-looking congestion calculation
   - Added queue explosion prevention mechanism
   - Marked final 6 observations per port with `is_valid_training_row` flag
   - Improved congestion threshold calibration
   - Balanced port service rates relative to arrival rates

2. **`scripts/generate_data.py`**
   - Enhanced reporting with detailed diagnostics
   - Added validation checks and value range reporting

### New Files

3. **`scripts/analyze_data.py`**
   - Deep statistical analysis of generated data
   - Relationship validation between features and targets

4. **`scripts/validate_data.py`**
   - Comprehensive validation suite (13 checks)
   - Physical constraint verification
   - Temporal consistency checks
   - Relationship validation

5. **`scripts/test_queue_dynamics.py`**
   - Queue evolution analysis
   - Congestion episode tracking
   - Temporal pattern validation

## Dataset Characteristics

### Shape and Size
- **Total observations:** 5,000
- **Ports:** 20
- **Features:** 64
- **Valid training rows:** 4,880 (97.6%)
- **Incomplete horizon rows:** 120 (6 per port)

### Congestion Distribution
- **Class 0 (not congested):** 74.1%
- **Class 1 (congested):** 25.9%

This is a reasonable class balance that represents realistic operational conditions where congestion is a significant but not overwhelming problem.

### Key Value Ranges

| Metric | Min | Max | Mean | Median |
|--------|-----|-----|------|--------|
| Queue length | 0.00 | 31.76 | 4.97 | 2.91 |
| Avg waiting time (hours) | 0.00 | 18.49 | 2.29 | 1.04 |
| Future delay (hours) | 0.00 | 25.47 | 3.92 | 3.02 |
| Berth utilization | 0.00 | 1.00 | 0.42 | 0.36 |
| Arrivals (1h) | 0 | 10 | 2.02 | 2.00 |

## Validation Results

All 13 validation checks passed:

✓ **Data Quality**
- No missing values
- No negative physical quantities
- Berth utilization in valid range [0, 1]
- Queue lengths are plausible (<100)
- Waiting times are plausible (<100 hours)

✓ **Temporal Consistency**
- Timestamps monotonic within each port
- Queue evolution is smooth (no jumps >50)
- Correct number of invalid training rows (120)

✓ **Operational Relationships**
- High queue → more congestion (63.3% vs 0.1%)
- High berth utilization → more congestion (67.7% vs 17.8%)
- Congestion rate (25.9%) in target range (20-40%)

## Key Operational Relationships

### Queue Pressure Impact
- **High queue (>50% of berths):** 40.8% of rows → 63.3% congested
- **Low queue (≤50% of berths):** 59.2% of rows → 0.1% congested

This shows a strong but not deterministic relationship between queue buildup and future congestion.

### Berth Utilization Impact
- **High utilization (>75%):** 16.2% of rows → 67.7% congested
- **Low utilization (≤75%):** 83.8% of rows → 17.8% congested

High capacity utilization significantly increases congestion risk, but doesn't guarantee it.

### Temporal Patterns
- **Peak congestion hour:** 22:00 (34.0% congested)
- **Peak congestion day:** Friday (37.3% congested)
- **Weekend vs weekday:** 20.0% vs 26.0% congested
- **Weather impact:** High weather pressure → 27.7% vs 25.8% congested

These patterns reflect the built-in daily, weekly, and environmental variations.

## Queue Dynamics

Sample congestion episode from PORT_05 shows realistic buildup:

```
Time         Queue    Arrivals   Waiting    Congested
01-01 12:00  2.15     2          0.93       Yes (predicting future)
01-01 13:00  1.31     5          0.62       Yes
01-01 14:00  3.69     5          1.44       Yes
01-01 15:00  6.40     7          2.22       Yes ← CONGESTED
01-01 16:00  10.60    0          3.47       Yes ← CONGESTED
01-01 17:00  7.74     4          2.57       Yes ← CONGESTED
01-01 18:00  8.64     1          2.82       Yes ← CONGESTED
...
```

The congestion label correctly predicts the episode **before** the queue peaks, demonstrating the 6-hour forward-looking prediction window.

## Issues Fixed

### 1. ✓ Service Capacity Double Application
**Problem:** Resource factors (equipment, labor, weather) were applied twice to service capacity.

**Solution:** Removed duplicate application. Now calculated once:
```python
resource_factor = equipment_factor * labor_factor * weather_factor
effective_service_capacity = base_service_rate * resource_factor
```

### 2. ✓ RuntimeWarning from Empty History
**Problem:** `np.mean(previous_arrivals[-3:])` failed when `previous_arrivals` was empty at simulation start.

**Solution:** Added conditional check:
```python
if len(previous_arrivals) > 0:
    traffic_growth_rate = ...
else:
    traffic_growth_rate = 0.0
```

### 3. ✓ Congestion Not Forward-Looking
**Problem:** Original implementation used incorrect window shift logic, not properly capturing the next 6 hours.

**Solution:** Implemented proper rolling mean over future observations:
```python
future_queue_mean_6h = (
    df.groupby("port_id")["queue_length"]
    .transform(lambda s: s.shift(-1).rolling(window=6, min_periods=1).mean())
)
```

### 4. ✓ Queue Explosion Prevention
**Problem:** Queues could grow unbounded if arrivals consistently exceeded service capacity.

**Solution:** Added relief mechanism for very large queues:
```python
if next_queue_length > profile.total_berths * 3.0:
    extra_services = min(next_queue_length * 0.15, ...)
    next_queue_length = max(0.0, next_queue_length - extra_services * relief_factor)
```

### 5. ✓ Final 6 Observations Not Marked
**Problem:** No indicator for rows without complete 6-hour horizon.

**Solution:** Added `is_valid_training_row` column:
```python
df["_future_hours_available"] = df.groupby("port_id").cumcount(ascending=False)
df["is_valid_training_row"] = df["_future_hours_available"] >= 6
```

### 6. ✓ Poor Class Balance
**Problem:** Initial implementation produced heavily imbalanced classes (64% congested).

**Solution:** 
- Adjusted port service rates to be 1.05-1.45× arrival rates
- Calibrated congestion threshold to 1.05
- Result: 25.9% congested (within target 20-40% range)

### 7. ✓ No Validation
**Problem:** No systematic validation of generated data.

**Solution:** Created comprehensive validation suite with 13 checks covering data quality, temporal consistency, and operational relationships.

## Design Compliance

The generator correctly implements all documented requirements:

✓ Generates ~5,000 observations across 20 ports  
✓ Maintains independent temporal state per port  
✓ Models operational flow: arrivals → capacity → queue → waiting → delay → congestion  
✓ Arrival demand varies by time/day/season  
✓ Service capacity constrained by berth/equipment/labor/weather  
✓ Resource constraints applied once only  
✓ Queue dynamics temporally consistent  
✓ Waiting and delay scale with queue pressure  
✓ Queue explosion prevented  
✓ Congestion target represents FUTURE 6-hour window  
✓ Congestion NOT simply current_queue > threshold  
✓ Final 6 observations handled correctly  
✓ No RuntimeWarnings  
✓ No missing values in valid training rows  
✓ Internal simulation variables separated from features  
✓ No future information in features  
✓ Dataset aligned with DATA_DESIGN.md  
✓ Reproducible with seed=42  
✓ Project structure preserved  

## Remaining Considerations

### 1. Congestion Threshold Sensitivity
The congestion threshold (1.05) was empirically calibrated to achieve ~25% congestion rate. This could be adjusted if different class balance is desired, but the current value produces meaningful operational relationships.

### 2. Queue Explosion Relief
The mechanism that prevents permanent queue explosion (reducing queue when it exceeds 3× berth capacity) is a pragmatic simulation constraint. In reality, ports might implement similar emergency measures (extra shifts, weekend operations, vessel rescheduling).

### 3. Weather Impact
Weather shows modest impact on congestion (27.7% vs 25.8%). This is realistic—weather constrains capacity but doesn't dominate all other factors. If stronger weather impact is desired, increase weather_factor influence in service capacity calculation.

### 4. Temporal Patterns
The data shows realistic daily/weekly patterns with peak congestion on Friday at 22:00 and lower weekend congestion. These patterns emerge from the built-in traffic multipliers in `_daily_pattern()` and `_weekly_pattern()`.

## Recommendation

The generated dataset is **ready for model training**. All validation checks pass, operational relationships are sensible, temporal dynamics are realistic, and the data correctly implements the documented design.

Proceed with:
1. Feature engineering (already has derived pressure features)
2. Train/validation/test split (time-aware)
3. Baseline model training
4. Advanced model comparison

Do **not** modify the generator further unless a specific operational relationship needs adjustment based on model performance analysis.
