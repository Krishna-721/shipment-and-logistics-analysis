"""Project-wide configuration and constants."""

from pathlib import Path
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
ARTIFACTS_MODELS = ARTIFACTS_ROOT / "models"
ARTIFACTS_METRICS = ARTIFACTS_ROOT / "metrics"
ARTIFACTS_PLOTS = ARTIFACTS_ROOT / "plots"

# Ensure directories exist
for path in [DATA_PROCESSED, ARTIFACTS_MODELS, ARTIFACTS_METRICS, ARTIFACTS_PLOTS]:
    path.mkdir(parents=True, exist_ok=True)

# Dataset configuration
DATASET_FILE = DATA_RAW / "port_operations.csv"
TARGET_COLUMN = "congestion_label"
VALID_ROW_COLUMN = "is_valid_training_row"
TIMESTAMP_COLUMN = "timestamp"
PORT_ID_COLUMN = "port_id"

# Reproducibility
RANDOM_STATE = 42

# Train/test split configuration (temporal split)
# Dataset spans 2026-01-01 to 2026-01-11.
# Congestion builds over the simulation window (9.6% on day 1 → ~50% by day 9),
# creating a deliberate temporal distribution shift in the test set.
# This reflects a realistic evaluation scenario where later periods are harder to predict.
# Split after day 8 maximises training data (3,840 rows, 20.6% congested)
# while keeping a meaningful test window (1,040 rows, 45.2% congested, ~21% of valid set).
TRAIN_END_DATE = "2026-01-08 23:59:59"   # First 8 days for training
TEST_START_DATE = "2026-01-09 00:00:00"  # Final ~2.1 days for testing

# Queue baseline configuration (from audit)
QUEUE_THRESHOLD = 0.5  # Queue > 50% of berths predicts congestion

# Model configuration
MODELS_TO_TRAIN = ["logistic", "random_forest", "xgboost"]

# Feature categories (based on audit)
IDENTIFIER_COLUMNS = ["port_id", "timestamp"]

TARGET_COLUMNS = ["congestion_label", "future_delay_hours"]

METADATA_COLUMNS = ["is_valid_training_row"]

# Essential features for modeling (from audit report)
ESSENTIAL_FEATURES = [
    "total_berths", "available_berths", "berth_utilization",
    "vessels_currently_in_port", "vessels_anchored", "vessels_approaching",
    "arrivals_last_1h", "arrivals_last_6h", "arrivals_last_24h",
    "arrival_rate", "queue_length", "avg_waiting_time_hours",
    "cranes_operational", "crane_utilization", "equipment_failure_count",
    "labor_availability_pct", "weather_severity", "storm_flag",
    "hour", "day_of_week", "is_weekend",
    "historical_avg_wait_time", "historical_congestion_rate",
    "traffic_pressure", "capacity_pressure", "queue_pressure",
    "equipment_pressure", "weather_pressure",
    "queue_capacity_interaction", "traffic_weather_interaction"
]

# Categorical features that need encoding
CATEGORICAL_FEATURES = ["vessel_type", "cargo_type"]

def load_dataset() -> pd.DataFrame:
    """Load the main dataset."""
    df = pd.read_csv(DATASET_FILE)
    df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN])
    return df
