"""Comprehensive validation of generated dataset."""

import pandas as pd
import numpy as np


def validate_dataset(df: pd.DataFrame) -> dict:
    """Run comprehensive validation checks."""
    
    results = {
        "passed": [],
        "warnings": [],
        "errors": []
    }
    
    # 1. Check for missing values
    missing = df.isna().sum().sum()
    if missing == 0:
        results["passed"].append("No missing values")
    else:
        results["errors"].append(f"Found {missing} missing values")
    
    # 2. Check physical constraints
    if (df["queue_length"] < 0).any():
        results["errors"].append("Negative queue length found")
    else:
        results["passed"].append("Queue length >= 0")
    
    if (df["avg_waiting_time_hours"] < 0).any():
        results["errors"].append("Negative waiting time found")
    else:
        results["passed"].append("Waiting time >= 0")
    
    if (df["future_delay_hours"] < 0).any():
        results["errors"].append("Negative delay found")
    else:
        results["passed"].append("Future delay >= 0")
    
    if (df["berth_utilization"] < 0).any() or (df["berth_utilization"] > 1.01).any():
        results["errors"].append("Berth utilization out of range [0, 1]")
    else:
        results["passed"].append("Berth utilization in valid range")
    
    # 3. Check temporal ordering
    for port in df["port_id"].unique():
        port_data = df[df["port_id"] == port]
        if not port_data["timestamp"].is_monotonic_increasing:
            results["errors"].append(f"Timestamps not monotonic for {port}")
            break
    else:
        results["passed"].append("Timestamps are monotonic within each port")
    
    # 4. Check valid training rows
    valid_count = df["is_valid_training_row"].sum()
    total = len(df)
    expected_invalid = df["port_id"].nunique() * 6
    actual_invalid = total - valid_count
    
    if actual_invalid == expected_invalid:
        results["passed"].append(f"Correct number of invalid training rows ({expected_invalid})")
    else:
        results["warnings"].append(
            f"Expected {expected_invalid} invalid rows, got {actual_invalid}"
        )
    
    # 5. Check congestion distribution
    valid_df = df[df["is_valid_training_row"]]
    cong_rate = valid_df["congestion_label"].mean()
    if 0.20 <= cong_rate <= 0.40:
        results["passed"].append(f"Congestion rate ({cong_rate:.1%}) in target range")
    else:
        results["warnings"].append(f"Congestion rate ({cong_rate:.1%}) outside target 20-40%")
    
    # 6. Check relationship: high queue → higher congestion
    high_queue = valid_df[valid_df["queue_length"] > valid_df["total_berths"] * 0.5]
    low_queue = valid_df[valid_df["queue_length"] <= valid_df["total_berths"] * 0.5]
    
    if len(high_queue) > 0 and len(low_queue) > 0:
        high_cong = high_queue["congestion_label"].mean()
        low_cong = low_queue["congestion_label"].mean()
        
        if high_cong > low_cong * 2:
            results["passed"].append(
                f"High queue → more congestion ({high_cong:.1%} vs {low_cong:.1%})"
            )
        else:
            results["warnings"].append(
                f"Weak queue-congestion relationship ({high_cong:.1%} vs {low_cong:.1%})"
            )
    
    # 7. Check relationship: high utilization → higher congestion
    high_util = valid_df[valid_df["berth_utilization"] > 0.75]
    low_util = valid_df[valid_df["berth_utilization"] <= 0.75]
    
    if len(high_util) > 0 and len(low_util) > 0:
        high_util_cong = high_util["congestion_label"].mean()
        low_util_cong = low_util["congestion_label"].mean()
        
        if high_util_cong > low_util_cong * 1.5:
            results["passed"].append(
                f"High utilization → more congestion ({high_util_cong:.1%} vs {low_util_cong:.1%})"
            )
        else:
            results["warnings"].append(
                f"Weak utilization-congestion relationship ({high_util_cong:.1%} vs {low_util_cong:.1%})"
            )
    
    # 8. Check value ranges are plausible
    if df["queue_length"].max() > 200:
        results["warnings"].append(f"Very large queue: {df['queue_length'].max():.1f}")
    else:
        results["passed"].append("Queue lengths are plausible")
    
    if df["avg_waiting_time_hours"].max() > 100:
        results["warnings"].append(f"Very long waiting time: {df['avg_waiting_time_hours'].max():.1f}h")
    else:
        results["passed"].append("Waiting times are plausible")
    
    # 9. Check queue dynamics (queue should evolve smoothly)
    for port in df["port_id"].unique()[:3]:  # Check first 3 ports
        port_data = df[df["port_id"] == port].sort_values("timestamp")
        queue_diff = port_data["queue_length"].diff().abs()
        if queue_diff.max() > 50:
            results["warnings"].append(f"Large queue jump in {port}: {queue_diff.max():.1f}")
            break
    else:
        results["passed"].append("Queue evolution is smooth")
    
    return results


def main():
    print("Loading dataset...")
    df = pd.read_csv("data/raw/port_operations.csv")
    
    print(f"\nValidating {len(df):,} observations across {df['port_id'].nunique()} ports...\n")
    
    results = validate_dataset(df)
    
    print("="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    
    if results["passed"]:
        print(f"\n✓ PASSED ({len(results['passed'])} checks):")
        for msg in results["passed"]:
            print(f"  • {msg}")
    
    if results["warnings"]:
        print(f"\n⚠ WARNINGS ({len(results['warnings'])} checks):")
        for msg in results["warnings"]:
            print(f"  • {msg}")
    
    if results["errors"]:
        print(f"\n✗ ERRORS ({len(results['errors'])} checks):")
        for msg in results["errors"]:
            print(f"  • {msg}")
    else:
        print(f"\n✓ No errors found")
    
    print("\n" + "="*60)
    
    # Summary
    total_checks = len(results["passed"]) + len(results["warnings"]) + len(results["errors"])
    print(f"\nSummary: {len(results['passed'])}/{total_checks} checks passed")
    
    if len(results["errors"]) == 0 and len(results["warnings"]) <= 2:
        print("Status: ✓ Dataset is valid and ready for training")
    elif len(results["errors"]) == 0:
        print("Status: ⚠ Dataset has some warnings but is usable")
    else:
        print("Status: ✗ Dataset has errors that must be fixed")
    
    print()


if __name__ == "__main__":
    main()
