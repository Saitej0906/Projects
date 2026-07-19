"""
feature_engineering.py
-----------------------
Turns raw per-cycle sensor readings into rolling-window statistical
features that describe HOW a sensor is behaving recently, not just its
instantaneous value.

WHY THIS MATTERS (and the fatigue-life connection):
A single raw sensor reading is noisy - it bounces around due to sensor
noise and minor operational variation, even for a healthy engine. What
actually signals degradation is the TREND over recent cycles: is the mean
level drifting, is variability increasing, is there a slope developing?
This is the same logic as using a moving average or damage-accumulation
rate in fatigue analysis rather than a single stress reading - you care
about the accumulating trend, not one noisy data point.

For each sensor, over a rolling window of W past cycles, we compute:
  - rolling MEAN   -> the current baseline/level of the sensor
  - rolling STD    -> how noisy/unstable it's become (instability often
                       increases as components wear)
  - rolling TREND  -> the slope of a straight line fit through the window
                       (is this sensor climbing or falling right now, and
                       how fast?) - directly analogous to a damage rate.

Window size = 5 cycles. This is a design choice: small enough to react to
real degradation, large enough to smooth out cycle-to-cycle sensor noise.
"""

import pandas as pd
import numpy as np

WINDOW = 5


def rolling_trend(series):
    """
    Slope of a best-fit line through the window (least-squares), i.e. the
    rate of change of the sensor per cycle within the window. This is 0 for
    a flat signal, positive if climbing, negative if falling.
    """
    y = series.values
    x = np.arange(len(y))
    if len(y) < 2 or np.all(y == y[0]):
        return 0.0
    slope = np.polyfit(x, y, 1)[0]
    return slope


def add_rolling_features(df, sensor_cols, window=WINDOW):
    """
    Adds {sensor}_mean, {sensor}_std, {sensor}_trend columns for every
    sensor, computed per-engine (so one engine's window never bleeds into
    another engine's data) and ordered by cycle.

    min_periods=1 means early cycles (before a full window exists) still
    get a value, computed from whatever history is available so far - this
    avoids losing the first few rows of every engine.
    """
    df = df.sort_values(["unit", "cycle"]).copy()
    grouped = df.groupby("unit")

    for sensor in sensor_cols:
        roll = grouped[sensor].rolling(window=window, min_periods=1)
        df[f"{sensor}_mean"] = roll.mean().reset_index(level=0, drop=True)
        df[f"{sensor}_std"] = roll.std().reset_index(level=0, drop=True).fillna(0.0)
        df[f"{sensor}_trend"] = (
            grouped[sensor]
            .rolling(window=window, min_periods=1)
            .apply(rolling_trend, raw=False)
            .reset_index(level=0, drop=True)
        )
    return df


if __name__ == "__main__":
    train = pd.read_csv("data/train_with_rul.csv")
    test = pd.read_csv("data/test_with_rul.csv")

    with open("outputs/metrics/useful_sensors.txt") as f:
        useful_sensors = f.read().splitlines()

    print(f"Engineering rolling features (window={WINDOW}) for {len(useful_sensors)} sensors...")
    train_feat = add_rolling_features(train, useful_sensors)
    test_feat = add_rolling_features(test, useful_sensors)

    print("Train shape before:", train.shape, "-> after:", train_feat.shape)
    print("New feature columns added:", train_feat.shape[1] - train.shape[1])

    train_feat.to_csv("data/train_features.csv", index=False)
    test_feat.to_csv("data/test_features.csv", index=False)
    print("Saved data/train_features.csv and data/test_features.csv")

    print("\nExample - engine 1, first 7 cycles, sensor_11 raw vs engineered:")
    cols = ["unit", "cycle", "sensor_11", "sensor_11_mean", "sensor_11_std", "sensor_11_trend"]
    print(train_feat[train_feat["unit"] == 1][cols].head(7).to_string(index=False))
