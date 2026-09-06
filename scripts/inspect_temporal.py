"""Inspect congestion rate over time to find a stable split boundary."""
import pandas as pd

df = pd.read_csv("data/raw/port_operations.csv", parse_dates=["timestamp"])
valid = df[df["is_valid_training_row"]]
valid = valid.copy()
valid["date"] = valid["timestamp"].dt.date

by_day = valid.groupby("date").agg(
    rows=("congestion_label", "count"),
    congestion_rate=("congestion_label", "mean")
)
by_day["congestion_pct"] = (by_day["congestion_rate"] * 100).round(1)
print(by_day[["rows", "congestion_pct"]].to_string())

print()
# Check cumulative splits at each possible day boundary
dates = sorted(valid["date"].unique())
print(f"{'Split after':<15} {'Train N':>8} {'Train cong%':>12} {'Test N':>8} {'Test cong%':>11} {'Test frac%':>11}")
print("-"*70)
for i in range(4, len(dates)-2):
    cutoff = dates[i]
    tr = valid[valid["date"] <= cutoff]
    te = valid[valid["date"] > cutoff]
    print(
        f"{str(cutoff):<15} {len(tr):>8} {tr['congestion_label'].mean()*100:>11.1f}%"
        f" {len(te):>8} {te['congestion_label'].mean()*100:>10.1f}% {len(te)/len(valid)*100:>10.1f}%"
    )
