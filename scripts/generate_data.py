"""Script to generate synthetic port data."""

from pathlib import Path

import pandas as pd

from src.data.generator import generate_dataset


def main() -> None:
    output_path = Path("data/raw/port_operations.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic port-operation dataset...")
    df = generate_dataset(
        n_ports=20,
        hours_per_port=250,
        start_date="2026-01-01",
        seed=42,
    )

    df.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print("DATASET GENERATION COMPLETE")
    print(f"{'='*60}")
    
    print(f"\n[1] Dataset shape: {df.shape}")
    print(f"    Total observations: {len(df):,}")
    print(f"    Ports: {df['port_id'].nunique()}")
    print(f"    Features: {df.shape[1]}")
    
    print(f"\n[2] Valid training rows:")
    valid_rows = df["is_valid_training_row"].sum()
    print(f"    Valid: {valid_rows:,} ({valid_rows/len(df)*100:.1f}%)")
    print(f"    Incomplete horizon: {len(df) - valid_rows:,}")
    
    print(f"\n[3] Congestion distribution (all rows):")
    cong_dist = df["congestion_label"].value_counts(normalize=True).sort_index()
    for label, pct in cong_dist.items():
        print(f"    Class {label}: {pct*100:.1f}%")
    
    print(f"\n[4] Congestion distribution (valid training only):")
    valid_df = df[df["is_valid_training_row"]]
    cong_dist_valid = valid_df["congestion_label"].value_counts(normalize=True).sort_index()
    for label, pct in cong_dist_valid.items():
        print(f"    Class {label}: {pct*100:.1f}%")
    
    print(f"\n[5] Missing values:")
    missing_total = df.isna().sum().sum()
    print(f"    Total: {missing_total}")
    if missing_total > 0:
        missing_by_col = df.isna().sum()[df.isna().sum() > 0]
        print(f"    By column:")
        for col, count in missing_by_col.items():
            print(f"      {col}: {count}")
    
    print(f"\n[6] Key value ranges:")
    print(f"    Queue length: [{df['queue_length'].min():.2f}, {df['queue_length'].max():.2f}]")
    print(f"    Avg waiting time: [{df['avg_waiting_time_hours'].min():.2f}, {df['avg_waiting_time_hours'].max():.2f}]")
    print(f"    Future delay: [{df['future_delay_hours'].min():.2f}, {df['future_delay_hours'].max():.2f}]")
    print(f"    Berth utilization: [{df['berth_utilization'].min():.2f}, {df['berth_utilization'].max():.2f}]")
    print(f"    Arrivals (1h): [{df['arrivals_last_1h'].min()}, {df['arrivals_last_1h'].max()}]")
    
    print(f"\n[7] Sanity checks:")
    
    # Check for negative physical quantities
    negative_cols = ["queue_length", "avg_waiting_time_hours", "future_delay_hours", 
                     "berth_utilization", "arrivals_last_1h"]
    has_negatives = False
    for col in negative_cols:
        if col in df.columns and (df[col] < 0).any():
            print(f"    WARNING: Negative values in {col}")
            has_negatives = True
    if not has_negatives:
        print(f"    ✓ No negative values in key columns")
    
    # Check berth utilization range
    if (df["berth_utilization"] > 1.0).any():
        print(f"    WARNING: Berth utilization > 1.0")
    else:
        print(f"    ✓ Berth utilization in valid range [0, 1]")
    
    # Check queue growth is reasonable
    queue_stats = df.groupby("port_id")["queue_length"].agg(["mean", "max"])
    if (queue_stats["max"] > 100).any():
        print(f"    WARNING: Some ports have very large queues (max: {queue_stats['max'].max():.1f})")
    else:
        print(f"    ✓ Queue lengths are reasonable")
    
    print(f"\n[8] Output saved to: {output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
