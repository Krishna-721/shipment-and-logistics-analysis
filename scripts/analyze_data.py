"""Analyze generated dataset statistics."""

import pandas as pd
import numpy as np


def main():
    df = pd.read_csv("data/raw/port_operations.csv")
    
    print("Distribution analysis:")
    print(f"\nQueue pressure stats:")
    queue_pressure = df["queue_length"] / df["total_berths"]
    print(f"  Mean: {queue_pressure.mean():.3f}")
    print(f"  Median: {queue_pressure.median():.3f}")
    print(f"  75th percentile: {queue_pressure.quantile(0.75):.3f}")
    print(f"  90th percentile: {queue_pressure.quantile(0.90):.3f}")
    
    print(f"\nWaiting time stats:")
    print(f"  Mean: {df['avg_waiting_time_hours'].mean():.2f}")
    print(f"  Median: {df['avg_waiting_time_hours'].median():.2f}")
    print(f"  75th: {df['avg_waiting_time_hours'].quantile(0.75):.2f}")
    print(f"  90th: {df['avg_waiting_time_hours'].quantile(0.90):.2f}")
    
    print(f"\nBerth utilization stats:")
    print(f"  Mean: {df['berth_utilization'].mean():.3f}")
    print(f"  Median: {df['berth_utilization'].median():.3f}")
    print(f"  75th: {df['berth_utilization'].quantile(0.75):.3f}")
    print(f"  90th: {df['berth_utilization'].quantile(0.90):.3f}")
    
    print(f"\nRelationship check:")
    valid = df[df["is_valid_training_row"]]
    
    high_queue = valid[valid["queue_length"] > valid["total_berths"] * 0.5]
    print(f"\nHigh queue (>50% of berths): {len(high_queue)} rows ({len(high_queue)/len(valid)*100:.1f}%)")
    print(f"  Congested: {high_queue['congestion_label'].mean()*100:.1f}%")
    
    low_queue = valid[valid["queue_length"] <= valid["total_berths"] * 0.5]
    print(f"\nLow queue (<=50% of berths): {len(low_queue)} rows ({len(low_queue)/len(valid)*100:.1f}%)")
    print(f"  Congested: {low_queue['congestion_label'].mean()*100:.1f}%")
    
    print(f"\nBy berth utilization:")
    high_util = valid[valid["berth_utilization"] > 0.75]
    print(f"\nHigh berth util (>75%): {len(high_util)} rows ({len(high_util)/len(valid)*100:.1f}%)")
    print(f"  Congested: {high_util['congestion_label'].mean()*100:.1f}%")
    
    low_util = valid[valid["berth_utilization"] <= 0.75]
    print(f"\nLow berth util (<=75%): {len(low_util)} rows ({len(low_util)/len(valid)*100:.1f}%)")
    print(f"  Congested: {low_util['congestion_label'].mean()*100:.1f}%")


if __name__ == "__main__":
    main()
