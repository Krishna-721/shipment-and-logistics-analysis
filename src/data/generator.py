"""
Synthetic port-operation data generator.

The generator simulates hourly port conditions and derives future congestion
from the simulated operational state.

A synthetic simulation, not a calibrated representation of any
specific real-world port.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PortProfile:
    """Static characteristics of a synthetic port."""

    port_id: str
    total_berths: int
    base_arrival_rate: float
    base_service_rate: float
    daily_teu_capacity: float
    typical_vessel_load_teu: float


def create_port_profiles(
    n_ports: int,
    rng: np.random.Generator,
) -> List[PortProfile]:
    """Create heterogeneous synthetic port profiles."""

    profiles = []

    for i in range(n_ports):
        # Ensure service rate is typically higher than arrival rate
        # but not so much that congestion never occurs
        base_arrival = float(rng.uniform(0.8, 3.0))
        base_service = float(rng.uniform(
            max(1.0, base_arrival * 1.05),
            max(3.0, base_arrival * 1.45)
        ))
        
        profiles.append(
            PortProfile(
                port_id=f"PORT_{i + 1:02d}",
                total_berths=int(rng.integers(4, 13)),
                base_arrival_rate=base_arrival,
                base_service_rate=base_service,
                daily_teu_capacity=float(rng.uniform(2500, 9000)),
                typical_vessel_load_teu=float(rng.uniform(800, 3000)),
            )
        )

    return profiles


def _daily_pattern(hour: int) -> float:
    """Represent recurring intra-day variation in vessel arrivals."""

    pattern = {
        0: 0.85,
        1: 0.80,
        2: 0.78,
        3: 0.80,
        4: 0.85,
        5: 0.95,
        6: 1.10,
        7: 1.20,
        8: 1.30,
        9: 1.35,
        10: 1.30,
        11: 1.20,
        12: 1.10,
        13: 1.05,
        14: 1.10,
        15: 1.20,
        16: 1.30,
        17: 1.35,
        18: 1.30,
        19: 1.20,
        20: 1.10,
        21: 1.00,
        22: 0.95,
        23: 0.90,
    }

    return pattern[hour]


def _weekly_pattern(day_of_week: int) -> float:
    """Represent weekday/weekend traffic differences."""

    return {
        0: 1.05,  # Monday
        1: 1.10,
        2: 1.08,
        3: 1.05,
        4: 1.12,
        5: 0.90,
        6: 0.82,
    }[day_of_week]


def _seasonal_pattern(month: int) -> float:
    """Represent mild seasonal variation."""

    return {
        1: 0.92,
        2: 0.95,
        3: 1.00,
        4: 1.03,
        5: 1.05,
        6: 1.08,
        7: 1.10,
        8: 1.08,
        9: 1.05,
        10: 1.02,
        11: 1.00,
        12: 1.08,
    }[month]


def _generate_weather(
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Generate correlated synthetic weather conditions."""

    wind_speed = float(np.clip(rng.normal(15, 7), 2, 45))
    wave_height = float(
        np.clip(0.4 + wind_speed * 0.045 + rng.normal(0, 0.25), 0.1, 4.5)
    )
    visibility = float(
        np.clip(
            10.0 - wind_speed * 0.08 + rng.normal(0, 1.5),
            1.0,
            15.0,
        )
    )

    wind_pressure = np.clip((wind_speed - 10) / 25, 0, 1)
    wave_pressure = np.clip((wave_height - 0.8) / 3.0, 0, 1)
    visibility_pressure = np.clip((8 - visibility) / 7, 0, 1)

    weather_severity = float(
        np.clip(
            0.45 * wind_pressure
            + 0.35 * wave_pressure
            + 0.20 * visibility_pressure,
            0,
            1,
        )
    )

    storm_flag = int(weather_severity >= 0.75)

    return {
        "wind_speed_knots": wind_speed,
        "wave_height_m": wave_height,
        "visibility_km": visibility,
        "weather_severity": weather_severity,
        "storm_flag": storm_flag,
    }


def _simulate_hour(
    profile: PortProfile,
    timestamp: pd.Timestamp,
    queue_length: float,
    previous_arrivals: List[int],
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Simulate one hour of port operations."""

    hour = timestamp.hour
    day_of_week = timestamp.dayofweek
    month = timestamp.month

    # ---------------------------------------------------------
    # 1. Weather
    # ---------------------------------------------------------
    weather = _generate_weather(rng)

    # ---------------------------------------------------------
    # 2. Vessel arrivals
    # ---------------------------------------------------------
    traffic_variation = float(rng.lognormal(mean=0.0, sigma=0.12))

    arrival_rate = (
        profile.base_arrival_rate
        * _daily_pattern(hour)
        * _weekly_pattern(day_of_week)
        * _seasonal_pattern(month)
        * traffic_variation
    )

    arrivals = int(rng.poisson(arrival_rate))

    # ---------------------------------------------------------
    # 3. Operational resources
    # ---------------------------------------------------------
    cranes_available = max(
        2,
        int(np.ceil(profile.total_berths * 1.2)),
    )

    equipment_failure_count = int(
        rng.poisson(0.08 + weather["storm_flag"] * 0.15)
    )

    cranes_operational = max(
        1,
        cranes_available - equipment_failure_count,
    )

    crane_utilization = float(
        np.clip(
            rng.normal(
                0.60 + 0.08 * min(queue_length / 10, 1),
                0.08,
            ),
            0.25,
            0.98,
        )
    )

    labor_availability_pct = float(
        np.clip(
            rng.normal(
                0.94 - 0.15 * weather["storm_flag"],
                0.05,
            ),
            0.55,
            1.0,
        )
    )

    labor_disruption_flag = int(labor_availability_pct < 0.75)

    # Calculate constraint factors (each applied once)
    equipment_factor = np.clip(
        cranes_operational / cranes_available,
        0.35,
        1.0,
    )

    labor_factor = np.clip(labor_availability_pct, 0.50, 1.0)

    weather_factor = np.clip(
        1.0 - 0.55 * weather["weather_severity"],
        0.45,
        1.0,
    )

    # ---------------------------------------------------------
    # 4. Berth capacity
    # ---------------------------------------------------------
    occupied_berths = min(
        profile.total_berths,
        int(round(queue_length * 0.25))
        + min(arrivals, profile.total_berths),
    )

    available_berths = max(
        0,
        profile.total_berths - occupied_berths,
    )

    berth_utilization = occupied_berths / profile.total_berths

    # ---------------------------------------------------------
    # 5. Effective service capacity
    # ---------------------------------------------------------
    # Combine all constraint factors (applied ONCE, not multiple times)
    resource_factor = (
        equipment_factor
        * labor_factor
        * weather_factor
    )

    # Base capacity constrained by resources
    handling_capacity = (
        profile.base_service_rate
        * resource_factor
    )

    # Prevent service capacity from becoming unrealistically low or high
    effective_service_capacity = np.clip(
        handling_capacity,
        0.10,
        profile.total_berths * 1.0,
    )

    # ---------------------------------------------------------
    # 6. Queue evolution
    # ---------------------------------------------------------
    services = min(
        queue_length + arrivals,
        effective_service_capacity,
    )

    next_queue_length = max(
        0.0,
        queue_length + arrivals - services,
    )

    # Prevent permanent queue explosion by adding occasional relief
    # when queues get very large
    if next_queue_length > profile.total_berths * 3.0:
        relief_factor = 1.0 + rng.uniform(0.1, 0.3)
        extra_services = min(
            next_queue_length * 0.15,
            next_queue_length - profile.total_berths * 2.5,
        )
        next_queue_length = max(
            0.0,
            next_queue_length - extra_services * relief_factor,
        )

    # ---------------------------------------------------------
    # 7. Waiting-time approximation
    # ---------------------------------------------------------
    if effective_service_capacity > 0.1 and queue_length > 0:
        waiting_time = queue_length / effective_service_capacity
    elif queue_length > 0:
        waiting_time = queue_length * 2.0
    else:
        waiting_time = 0.0

    waiting_time += float(
        rng.normal(0, 0.15)
    )

    waiting_time = max(0.0, waiting_time)

    max_waiting_time = max(
        waiting_time,
        waiting_time * float(rng.uniform(1.2, 2.0)),
    )

    oldest_waiting_vessel_hours = max(
        waiting_time,
        waiting_time * float(rng.uniform(1.2, 2.5)),
    )

    # ---------------------------------------------------------
    # 8. Cargo/workload
    # ---------------------------------------------------------
    vessel_type = rng.choice(
        ["container", "bulk", "tanker", "general_cargo"],
        p=[0.50, 0.20, 0.15, 0.15],
    )

    cargo_type = {
        "container": "containerized",
        "bulk": "dry_bulk",
        "tanker": "liquid_bulk",
        "general_cargo": "general",
    }[vessel_type]

    vessel_size_factor = float(
        rng.lognormal(mean=0.0, sigma=0.25)
    )

    cargo_volume_teu = (
        profile.typical_vessel_load_teu
        * vessel_size_factor
        if vessel_type == "container"
        else 0.0
    )

    cargo_weight_tons = float(
        np.clip(
            rng.normal(35000, 15000),
            5000,
            100000,
        )
    )

    estimated_moves = (
        int(max(50, cargo_volume_teu * rng.uniform(0.75, 1.0)))
        if vessel_type == "container"
        else int(rng.integers(50, 800))
    )

    avg_crane_productivity = float(
        np.clip(
            rng.normal(28, 5),
            12,
            45,
        )
    )

    estimated_service_time_hours = max(
        1.0,
        estimated_moves / (
            max(1, cranes_operational)
            * avg_crane_productivity
        ),
    )

    # ---------------------------------------------------------
    # 9. Throughput
    # ---------------------------------------------------------
    current_throughput_teu = float(
        max(
            0,
            rng.normal(
                profile.daily_teu_capacity / 24,
                profile.daily_teu_capacity / 100,
            ),
        )
    )

    capacity_headroom = float(
        np.clip(
            1.0
            - (
                current_throughput_teu
                / (profile.daily_teu_capacity / 24)
            ),
            -0.5,
            1.0,
        )
    )

    # ---------------------------------------------------------
    # 10. Historical context
    # ---------------------------------------------------------
    historical_avg_wait_time = float(
        np.clip(
            rng.normal(
                1.5 + 0.03 * profile.base_arrival_rate,
                0.35,
            ),
            0.3,
            6.0,
        )
    )

    historical_avg_turnaround_time = float(
        np.clip(
            rng.normal(
                18 + 2 * profile.base_arrival_rate,
                3,
            ),
            8,
            40,
        )
    )

    historical_delay_rate = float(
        np.clip(
            rng.normal(
                0.15 + 0.03 * profile.base_arrival_rate,
                0.05,
            ),
            0.03,
            0.50,
        )
    )

    historical_congestion_rate = float(
        np.clip(
            rng.normal(
                0.12 + 0.025 * profile.base_arrival_rate,
                0.04,
            ),
            0.02,
            0.45,
        )
    )

    return {
        "port_id": profile.port_id,
        "timestamp": timestamp,
        "total_berths": profile.total_berths,
        "available_berths": available_berths,
        "occupied_berths": occupied_berths,
        "berth_utilization": berth_utilization,
        "daily_vessel_capacity": profile.base_arrival_rate * 24 * 1.4,
        "daily_teu_capacity": profile.daily_teu_capacity,
        "current_throughput_teu": current_throughput_teu,
        "capacity_headroom": capacity_headroom,
        "vessels_currently_in_port": int(
            max(0, round(queue_length + arrivals))
        ),
        "vessels_anchored": int(
            max(0, round(queue_length * 0.70))
        ),
        "vessels_approaching": int(
            rng.poisson(max(0.1, arrival_rate * 0.8))
        ),
        "arrivals_last_1h": arrivals,
        "arrivals_last_6h": int(
            sum(previous_arrivals[-5:]) + arrivals
        ),
        "arrivals_last_24h": int(
            sum(previous_arrivals[-23:]) + arrivals
        ),
        "departures_last_6h": int(
            rng.poisson(max(0.1, effective_service_capacity * 0.7))
        ),
        "arrival_rate": arrival_rate,
        "traffic_growth_rate": float(
            np.clip(
                (
                    arrivals
                    - np.mean(previous_arrivals[-3:])
                )
                / max(
                    1.0,
                    np.mean(previous_arrivals[-3:]),
                )
                if len(previous_arrivals) > 0
                else 0.0,
                -2,
                2,
            )
        ),
        "queue_length": float(queue_length),
        "anchorage_vessel_count": int(
            max(0, round(queue_length * 0.70))
        ),
        "avg_waiting_time_hours": waiting_time,
        "max_waiting_time_hours": max_waiting_time,
        "oldest_waiting_vessel_hours": oldest_waiting_vessel_hours,
        "queue_growth_rate": float(
            next_queue_length - queue_length
        ),
        "vessel_type": vessel_type,
        "cargo_type": cargo_type,
        "cargo_volume_teu": cargo_volume_teu,
        "cargo_weight_tons": cargo_weight_tons,
        "estimated_moves": estimated_moves,
        "estimated_service_time_hours": estimated_service_time_hours,
        "cranes_available": cranes_available,
        "cranes_operational": cranes_operational,
        "crane_utilization": crane_utilization,
        "equipment_failure_count": equipment_failure_count,
        "avg_crane_productivity": avg_crane_productivity,
        "labor_availability_pct": labor_availability_pct,
        "labor_disruption_flag": labor_disruption_flag,
        **weather,
        "hour": hour,
        "day_of_week": day_of_week,
        "month": month,
        "is_weekend": int(day_of_week >= 5),
        "is_holiday": 0,
        "historical_avg_wait_time": historical_avg_wait_time,
        "historical_avg_turnaround_time": historical_avg_turnaround_time,
        "historical_delay_rate": historical_delay_rate,
        "historical_congestion_rate": historical_congestion_rate,
        # Internal simulation values.
        # These will not be used as model features.
        "_next_queue_length": next_queue_length,
        "_effective_service_capacity": effective_service_capacity,
    }


def _calculate_derived_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate pressure and interaction features."""

    df = df.copy()

    df["arrival_density"] = (
        df["arrivals_last_6h"] / 6.0
    )

    df["traffic_pressure"] = np.clip(
        df["arrival_rate"]
        / df["daily_vessel_capacity"].clip(lower=1)
        * 24,
        0,
        2,
    )

    df["capacity_pressure"] = np.clip(
        df["berth_utilization"],
        0,
        1,
    )

    df["queue_pressure"] = np.clip(
        df["queue_length"]
        / df["total_berths"].clip(lower=1),
        0,
        3,
    )

    df["equipment_pressure"] = np.clip(
        1
        - (
            df["cranes_operational"]
            / df["cranes_available"].clip(lower=1)
        ),
        0,
        1,
    )

    df["weather_pressure"] = df["weather_severity"]

    df["historical_pressure"] = (
        0.5 * df["historical_delay_rate"]
        + 0.5 * df["historical_congestion_rate"]
    )

    df["queue_capacity_interaction"] = (
        df["queue_pressure"]
        * df["capacity_pressure"]
    )

    df["traffic_weather_interaction"] = (
        df["traffic_pressure"]
        * df["weather_pressure"]
    )

    return df


def _add_future_targets(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Derive future congestion and delay targets.

    Future congestion is based on the simulated future state over the
    following six hours, rather than a direct threshold on current queue.
    
    The final six observations of each port do not have a complete 6-hour
    future horizon and are marked with is_valid_training_row=False.
    """

    df = df.copy()

    # For each port, calculate future queue statistics
    future_queue_mean_6h = (
        df.groupby("port_id")["queue_length"]
        .transform(
            lambda s: (
                s.shift(-1)
                .rolling(window=6, min_periods=1)
                .mean()
            )
        )
    )
    
    future_queue_max_6h = (
        df.groupby("port_id")["queue_length"]
        .transform(
            lambda s: (
                s.shift(-1)
                .rolling(window=6, min_periods=1)
                .max()
            )
        )
    )
    
    # Calculate how many valid future hours each row has
    df["_future_hours_available"] = (
        df.groupby("port_id")
        .cumcount(ascending=False)
    )
    
    # Mark rows with incomplete 6-hour horizon
    df["is_valid_training_row"] = (
        df["_future_hours_available"] >= 6
    )

    # Future waiting time estimate
    future_wait = (
        df["avg_waiting_time_hours"]
        + 0.50 * future_queue_mean_6h.fillna(df["queue_length"])
    )

    # Future queue pressure
    queue_pressure_future = (
        future_queue_mean_6h
        / df["total_berths"].clip(lower=1)
    )
    
    # Future peak queue pressure
    queue_pressure_peak = (
        future_queue_max_6h
        / df["total_berths"].clip(lower=1)
    )

    # Congestion is a combined future operational state
    # High future queue + high waiting + current capacity constraints
    # The threshold is calibrated so congestion represents sustained operational
    # pressure rather than any temporary queue buildup
    congestion_score = (
        0.35 * np.clip(queue_pressure_future, 0, 3.0)
        + 0.25 * np.clip(queue_pressure_peak, 0, 3.0)
        + 0.25 * np.clip(future_wait / 6.0, 0, 3.0)
        + 0.15 * np.clip(
            df["berth_utilization"]
            + df["weather_pressure"] * 0.30,
            0,
            2.0,
        )
    )

    # Threshold set to capture truly congested periods
    # targeting approximately 25-40% of observations
    df["congestion_label"] = (
        congestion_score >= 1.05
    ).astype(int)

    # Delay target combines future waiting pressure and service disruption
    delay_rng = np.random.default_rng(42)
    delay_noise = delay_rng.normal(0, 0.25, len(df))

    df["future_delay_hours"] = np.maximum(
        0,
        future_wait * 0.85
        + df["estimated_service_time_hours"] * 0.12
        + df["weather_pressure"] * 1.2
        + df["equipment_pressure"] * 0.8
        + delay_noise,
    )

    # Store diagnostic columns temporarily
    df["_future_queue_mean_6h"] = future_queue_mean_6h
    df["_future_queue_max_6h"] = future_queue_max_6h
    df["_congestion_score"] = congestion_score

    return df


def generate_dataset(
    n_ports: int = 20,
    hours_per_port: int = 250,
    start_date: str = "2026-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate the complete synthetic port-operation dataset.

    Parameters
    ----------
    n_ports:
        Number of synthetic ports.

    hours_per_port:
        Number of hourly observations generated for each port.

    start_date:
        Starting timestamp.

    seed:
        Random seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
        Synthetic port-operation dataset.
    """

    if n_ports <= 0:
        raise ValueError("n_ports must be greater than zero.")

    if hours_per_port < 10:
        raise ValueError(
            "hours_per_port must be at least 10."
        )

    rng = np.random.default_rng(seed)

    profiles = create_port_profiles(
        n_ports=n_ports,
        rng=rng,
    )

    timestamps = pd.date_range(
        start=start_date,
        periods=hours_per_port,
        freq="h",
    )

    records = []

    for profile in profiles:
        queue_length = float(
            rng.uniform(
                0,
                max(1.0, profile.total_berths * 0.50),
            )
        )

        previous_arrivals: List[int] = []

        for timestamp in timestamps:
            record = _simulate_hour(
                profile=profile,
                timestamp=timestamp,
                queue_length=queue_length,
                previous_arrivals=previous_arrivals,
                rng=rng,
            )

            records.append(record)

            queue_length = record["_next_queue_length"]

            previous_arrivals.append(
                record["arrivals_last_1h"]
            )

    df = pd.DataFrame(records)

    df = _calculate_derived_features(df)

    df = _add_future_targets(df)

    # Remove internal simulation columns from the public dataset
    # but keep is_valid_training_row for filtering
    internal_columns = [
        "_next_queue_length",
        "_effective_service_capacity",
        "_future_queue_mean_6h",
        "_future_queue_max_6h",
        "_congestion_score",
        "_future_hours_available",
    ]

    df = df.drop(
        columns=[
            column
            for column in internal_columns
            if column in df.columns
        ]
    )

    df = df.sort_values(
        ["port_id", "timestamp"]
    ).reset_index(drop=True)

    return df


if __name__ == "__main__":
    dataset = generate_dataset()

    print(
        f"Generated {len(dataset):,} observations "
        f"across {dataset['port_id'].nunique()} ports."
    )

    print("\nColumns:")
    print(dataset.columns.tolist())

    print("\nCongestion distribution:")
    print(
        dataset["congestion_label"]
        .value_counts(normalize=True)
        .sort_index()
    )

    print("\nSample:")
    print(dataset.head())