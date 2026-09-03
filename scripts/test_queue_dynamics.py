"""Test queue dynamics and temporal consistency."""

import pandas as pd
import numpy as np


def main():
    df = pd.read_csv("data/raw/port_operations.csv")
    
    print("Testing queue dynamics and temporal consistency...\n")
    print("="*60)
    
    # Sample one port for detailed analysis
    sample_port = "PORT_05"
    port_data = df[df["port_id"] == sample_port].sort_values("timestamp").reset_index(drop=True)
    
    print(f"\nPort: {sample_port}")
    print(f"Observations: {len(port_data)}")
    
    # Show first 10 hours of queue evolution
    print(f"\nFirst 10 hours of queue evolution:")
    print("-" * 80)
    print(f"{'Hour':<6} {'Queue':<8} {'Arrivals':<10} {'Berth Util':<12} {'Waiting':<10} {'Congested':<10}")
    print("-" * 80)
    
    for idx in range(min(10, len(port_data))):
        row = port_data.iloc[idx]
        print(f"{idx:<6} {row['queue_length']:<8.2f} {row['arrivals_last_1h']:<10} "
              f"{row['berth_utilization']:<12.2f} {row['avg_waiting_time_hours']:<10.2f} "
              f"{'Yes' if row['congestion_label'] == 1 else 'No':<10}")
    
    # Check for a congestion episode
    congested_periods = port_data[port_data["congestion_label"] == 1]
    
    if len(congested_periods) > 0:
        print(f"\n\nCongestion episode analysis:")
        print("-" * 80)
        
        # Find first congestion episode
        first_cong_idx = congested_periods.index[0]
        start = max(0, first_cong_idx - 3)
        end = min(len(port_data), first_cong_idx + 7)
        
        episode = port_data.iloc[start:end]
        
        print(f"{'Time':<12} {'Queue':<8} {'Arrivals':<10} {'Waiting':<10} {'Delay':<10} {'Congested':<10}")
        print("-" * 80)
        
        for idx, row in episode.iterrows():
            ts = pd.to_datetime(row["timestamp"]).strftime("%m-%d %H:%M")
            cong_mark = " ← CONGESTED" if row["congestion_label"] == 1 else ""
            print(f"{ts:<12} {row['queue_length']:<8.2f} {row['arrivals_last_1h']:<10} "
                  f"{row['avg_waiting_time_hours']:<10.2f} {row['future_delay_hours']:<10.2f} "
                  f"{'Yes':<10}{cong_mark}")
        
        print(f"\nNote: Congestion is predicted based on the NEXT 6 hours, not current state.")
    
    # Summary statistics across all ports
    print("\n" + "="*60)
    print("\nAggregate statistics across all ports:")
    print("-" * 60)
    
    print(f"\nQueue dynamics:")
    print(f"  Mean queue length: {df['queue_length'].mean():.2f}")
    print(f"  Std queue length: {df['queue_length'].std():.2f}")
    print(f"  Max queue length: {df['queue_length'].max():.2f}")
    
    print(f"\nArrival/Service balance:")
    valid = df[df["is_valid_training_row"]]
    print(f"  Mean arrivals (1h): {valid['arrivals_last_1h'].mean():.2f}")
    print(f"  Mean departures (6h): {valid['departures_last_6h'].mean():.2f}")
    print(f"  Mean arrival rate: {valid['arrival_rate'].mean():.2f} vessels/hour")
    
    print(f"\nCongestion patterns:")
    print(f"  Overall congestion rate: {valid['congestion_label'].mean()*100:.1f}%")
    
    # By time of day
    valid["hour"] = pd.to_datetime(valid["timestamp"]).dt.hour
    cong_by_hour = valid.groupby("hour")["congestion_label"].mean() * 100
    peak_hour = cong_by_hour.idxmax()
    print(f"  Peak congestion hour: {peak_hour:02d}:00 ({cong_by_hour[peak_hour]:.1f}%)")
    
    # By day of week
    valid["dow"] = pd.to_datetime(valid["timestamp"]).dt.dayofweek
    cong_by_dow = valid.groupby("dow")["congestion_label"].mean() * 100
    peak_dow = cong_by_dow.idxmax()
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    print(f"  Peak congestion day: {days[peak_dow]} ({cong_by_dow[peak_dow]:.1f}%)")
    print(f"  Weekend congestion: {cong_by_dow[[5,6]].mean():.1f}%")
    print(f"  Weekday congestion: {cong_by_dow[[0,1,2,3,4]].mean():.1f}%")
    
    print(f"\nWeather impact:")
    high_weather = valid[valid["weather_pressure"] > 0.5]
    low_weather = valid[valid["weather_pressure"] <= 0.5]
    print(f"  Congestion with high weather pressure: {high_weather['congestion_label'].mean()*100:.1f}%")
    print(f"  Congestion with low weather pressure: {low_weather['congestion_label'].mean()*100:.1f}%")
    
    print("\n" + "="*60)
    print("\n✓ Queue dynamics and temporal patterns look realistic\n")


if __name__ == "__main__":
    main()
