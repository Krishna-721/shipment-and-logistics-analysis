"""Quick check of temporal split before running full pipeline."""
import sys
sys.path.insert(0, ".")
import pandas as pd
from src.config import TRAIN_END_DATE, TEST_START_DATE

df = pd.read_csv("data/raw/port_operations.csv", parse_dates=["timestamp"])
valid = df[df["is_valid_training_row"]]

train = valid[valid["timestamp"] <= TRAIN_END_DATE]
test  = valid[valid["timestamp"] >= TEST_START_DATE]

cr_tr = train["congestion_label"].mean() * 100
cr_te = test["congestion_label"].mean() * 100

print(f"Total valid rows : {len(valid)}")
print(f"Train rows       : {len(train)} ({cr_tr:.1f}% congested)")
print(f"Test  rows       : {len(test)} ({cr_te:.1f}% congested)")
print(f"Train period     : {train['timestamp'].min()} -> {train['timestamp'].max()}")
print(f"Test  period     : {test['timestamp'].min()} -> {test['timestamp'].max()}")
print(f"Test fraction    : {len(test)/len(valid)*100:.1f}%")
print(f"No overlap       : {train['timestamp'].max() < test['timestamp'].min()}")
