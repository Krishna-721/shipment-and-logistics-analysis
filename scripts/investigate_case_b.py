"""Deep dive into Case B scarcity issue."""

import pandas as pd
import numpy as np


def main():
    df = pd.read_csv("data/raw/port_operations.csv")
    valid = df[df["is_valid_training_row"]].copy()
    
    print("="*80)
    print("INVESTIGATING CASE B SCARCITY")
    print("="*80)
    
    # Case B: Low/moderate current queue but FUTURE congestion
    print("\nCase B: Current queue ≤ 50% of berths BUT future congestion = 1")
    print("-" * 80)
    
    # Try different thresholds
    thresholds = [0.3, 0.5, 0.7, 1.0]
    
    print("\nCase B frequency by queue threshold:")
    print(f"{'Threshold':<15} {'Count':<10} {'% of Valid':<15} {'% of Congested':<20}")
    print("-" * 80)
    
    for threshold in thresholds:
        case_b = valid[
            (valid["queue_length"] <= valid["total_berths"] * threshold) &
            (valid["congestion_label"] == 1)
        ]
        
        pct_valid = len(case_b) / len(valid) * 100
        
        congested = valid[valid["congestion_label"] == 1]
        pct_congested = len(case_b) / len(congested) * 100 if len(congested) > 0 else 0
        
        print(f"{threshold*100:.0f}% of berths {len(case_b):<10} {pct_valid:<15.2f} {pct_congested:<20.2f}")
    
    # Analyze the 2 Case B instances in detail
    case_b = valid[
        (valid["queue_length"] <= valid["total_berths"] * 0.5) &
        (valid["congestion_label"] == 1)
    ]
    
    print(f"\n\nDetailed analysis of {len(case_b)} Case B instances:")
    print("="*80)
    
    for idx, row in case_b.iterrows():
        print(f"\nPort: {row['port_id']} at {row['timestamp']}")
        print(f"  Current queue: {row['queue_length']:.2f} vessels ({row['queue_length']/row['total_berths']*100:.0f}% of {row['total_berths']} berths)")
        print(f"  Current waiting: {row['avg_waiting_time_hours']:.2f} hours")
        print(f"  Berth util: {row['berth_utilization']:.2f}")
        print(f"  Arrival rate: {row['arrival_rate']:.2f} vessels/hour")
        print(f"  Arrivals last 6h: {row['arrivals_last_6h']}")
        print(f"  Traffic pressure: {row['traffic_pressure']:.2f}")
        print(f"  Queue pressure: {row['queue_pressure']:.2f}")
        print(f"  Weather severity: {row['weather_severity']:.2f}")
        print(f"  Equipment pressure: {row['equipment_pressure']:.2f}")
        
        # Look at the next 6 hours for this port
        port_data = df[df["port_id"] == row["port_id"]].sort_values("timestamp").reset_index(drop=True)
        current_idx = port_data[port_data["timestamp"] == row["timestamp"]].index[0]
        
        print(f"\n  Next 6 hours evolution:")
        print(f"  {'Hour':<6} {'Queue':<10} {'Arrivals':<10} {'Waiting':<10}")
        print("  " + "-"*40)
        
        for i in range(min(7, len(port_data) - current_idx)):
            future_row = port_data.iloc[current_idx + i]
            marker = " ← NOW" if i == 0 else ""
            print(f"  +{i:<5} {future_row['queue_length']:<10.2f} "
                  f"{future_row['arrivals_last_1h']:<10} "
                  f"{future_row['avg_waiting_time_hours']:<10.2f}{marker}")
    
    # Check overall predictability from current queue
    print("\n\n" + "="*80)
    print("PREDICTABILITY ANALYSIS")
    print("="*80)
    
    # Create confusion matrix based on simple queue threshold
    queue_threshold = 0.5
    
    valid["predicted_from_queue"] = (valid["queue_length"] > valid["total_berths"] * queue_threshold).astype(int)
    
    print(f"\nIf we predict congestion using: queue > {queue_threshold*100:.0f}% of berths")
    print("\nConfusion Matrix:")
    
    from sklearn.metrics import confusion_matrix, classification_report
    
    cm = confusion_matrix(valid["congestion_label"], valid["predicted_from_queue"])
    
    print(f"\n                     Predicted No    Predicted Yes")
    print(f"  Actual No          {cm[0,0]:<15} {cm[0,1]:<15}")
    print(f"  Actual Yes         {cm[1,0]:<15} {cm[1,1]:<15}")
    
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n  Accuracy:  {accuracy:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1-score:  {f1:.3f}")
    
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    
    print(f"\nThe congestion target can be predicted with {accuracy*100:.1f}% accuracy")
    print(f"using ONLY current queue length (> {queue_threshold*100:.0f}% of berths).")
    
    print("\nThis indicates:")
    
    if accuracy > 0.95:
        print("  ⚠ CRITICAL: Target is almost entirely determined by current queue")
        print("     The 6-hour forward-looking window may not be working correctly")
        print("     ML model may learn to simply threshold the queue")
    elif accuracy > 0.85:
        print("  ⚠ WARNING: Target is strongly determined by current queue")
        print("     Forward-looking dynamics exist but are weak")
        print("     Consider investigating congestion calculation")
    else:
        print("  ✓ Target has meaningful forward-looking component")
        print("     Current queue is predictive but not deterministic")
    
    # Check if Case B scarcity is a real problem
    print("\nCase B (low queue → congestion) represents the forward-looking signal:")
    print(f"  Found: {len(case_b)} cases ({len(case_b)/len(valid)*100:.2f}% of dataset)")
    print(f"  As % of congested: {len(case_b)/len(congested)*100:.2f}%")
    
    if len(case_b) < 10:
        print("\n  ⚠ CRITICAL WEAKNESS:")
        print("     Almost no cases where congestion occurs despite low current queue")
        print("     This means the target is NOT truly forward-looking")
        print("     It's mainly capturing CURRENT operational stress, not FUTURE risk")


if __name__ == "__main__":
    main()
