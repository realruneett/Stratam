"""
Generate synthetic train.csv and test.csv matching the Flipkart
Gridlock Hackathon 2.0 schema for local testing.

Columns: Index, geohash, day, timestamp, RoadType, NumberofLanes,
         LargeVehicles, Landmarks, Temperature, Weather, demand
"""

import csv
import math
import random
from datetime import datetime, timedelta


# Sample geohashes (Bengaluru area)
GEOHASHES = [
    "tdr1y2", "tdr1y3", "tdr1y4", "tdr1y5", "tdr1y6",
    "tdr1y7", "tdr1y8", "tdr1y9", "tdr1ya", "tdr1yb",
    "tdr2a0", "tdr2a1", "tdr2a2", "tdr2a3", "tdr2a4",
]

ROAD_TYPES = ["primary", "secondary", "tertiary", "residential"]
WEATHER_TYPES = ["Clear", "Rainy", "Cloudy", "Foggy"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def generate_data():
    random.seed(42)

    # Train: 14 days, hourly, across all geohashes
    start_train = datetime(2026, 5, 1, 0, 0, 0)
    train_hours = 14 * 24  # 336 hours

    # Test: 4 days, hourly, strictly future
    start_test = datetime(2026, 5, 15, 0, 0, 0)
    test_hours = 4 * 24  # 96 hours

    # ── Write train.csv ──────────────────────────────────────────
    with open("data/train.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Index", "geohash", "day", "timestamp", "RoadType",
            "NumberofLanes", "LargeVehicles", "Landmarks",
            "Temperature", "Weather", "demand",
        ])

        idx = 0
        for gh in GEOHASHES:
            gh_bias = GEOHASHES.index(gh) * 5
            road = random.choice(ROAD_TYPES)
            lanes = random.choice([2, 4, 6])
            large_vehicles = random.choice([0, 1])
            landmarks = random.choice([0, 1])

            for h in range(train_hours):
                dt = start_train + timedelta(hours=h)
                hour = dt.hour
                day_name = DAYS[dt.weekday()]

                # Realistic demand: daily cycle + location bias + noise
                daily = 15.0 * math.sin(2 * math.pi * hour / 24.0) + 20.0
                weekend_factor = 0.7 if dt.weekday() >= 5 else 1.0
                temp = round(25 + 5 * math.sin(2 * math.pi * hour / 24.0) + random.gauss(0, 2), 1)
                weather = random.choices(WEATHER_TYPES, weights=[0.5, 0.15, 0.25, 0.1])[0]
                weather_factor = 0.85 if weather == "Rainy" else 1.0
                noise = random.gauss(0, 3)

                demand = max(0, int(round(
                    (gh_bias + daily + lanes * 2) * weekend_factor * weather_factor + noise
                )))

                writer.writerow([
                    idx, gh, day_name, dt.strftime("%Y-%m-%d %H:%M:%S"),
                    road, lanes, large_vehicles, landmarks,
                    temp, weather, demand,
                ])
                idx += 1

    print(f"Generated data/train.csv: {idx} rows × 11 columns")

    # ── Write test.csv ───────────────────────────────────────────
    test_start_idx = idx
    with open("data/test.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Index", "geohash", "day", "timestamp", "RoadType",
            "NumberofLanes", "LargeVehicles", "Landmarks",
            "Temperature", "Weather",
        ])

        for gh in GEOHASHES:
            road = random.choice(ROAD_TYPES)
            lanes = random.choice([2, 4, 6])
            large_vehicles = random.choice([0, 1])
            landmarks = random.choice([0, 1])

            for h in range(test_hours):
                dt = start_test + timedelta(hours=h)
                hour = dt.hour
                day_name = DAYS[dt.weekday()]
                temp = round(25 + 5 * math.sin(2 * math.pi * hour / 24.0) + random.gauss(0, 2), 1)
                weather = random.choices(WEATHER_TYPES, weights=[0.5, 0.15, 0.25, 0.1])[0]

                writer.writerow([
                    idx, gh, day_name, dt.strftime("%Y-%m-%d %H:%M:%S"),
                    road, lanes, large_vehicles, landmarks,
                    temp, weather,
                ])
                idx += 1

    test_rows = idx - test_start_idx
    print(f"Generated data/test.csv: {test_rows} rows × 10 columns")

    # ── Write sample_submission.csv ──────────────────────────────
    with open("data/sample_submission.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Index", "demand"])
        for i in range(5):
            writer.writerow([test_start_idx + i, 0])
    print("Generated data/sample_submission.csv: 5 rows × 2 columns")


if __name__ == "__main__":
    generate_data()
