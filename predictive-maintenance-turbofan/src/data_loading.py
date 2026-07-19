"""
data_loading.py
----------------
Loads the NASA CMAPSS FD001 turbofan dataset and explains its structure.

WHAT THIS DATASET IS:
Each row is one engine, at one point in time (one "cycle" = roughly one flight).
Engines start healthy and run until they fail. As an engine degrades, its
sensor readings (temperatures, pressures, fan speeds, etc.) drift away from
their healthy baseline.

- TRAIN set: 100 engines, each run all the way to failure. We know exactly
  how many cycles were left at every point, because we can see the end.
- TEST set: 100 different engines, each cut off BEFORE failure at some
  random point. We only see sensor data up to that cutoff.
- RUL file: for each test engine, the true "Remaining Useful Life" (in
  cycles) at the moment its data was cut off. This is the ground truth we
  try to predict.

WHAT "RUL" MEANS:
Remaining Useful Life = number of operating cycles left before the engine
would fail, counting forward from a given row. For a TRAINING engine that
fails at cycle 200, the row at cycle 150 has RUL = 200 - 150 = 50.
This is directly analogous to remaining fatigue life in your ME background:
instead of predicting cycles-to-crack-initiation from stress amplitude and
material S-N curves, here we predict cycles-to-failure from live sensor
"symptoms" of accumulating damage. Same idea (cumulative damage -> life left),
different evidence (sensors instead of a stress model).
"""

import pandas as pd
import numpy as np

# The raw .txt files have no header row and are space-delimited.
# There are 26 columns: unit number, cycle number, 3 operational settings,
# and 21 sensor readings. This layout is documented by NASA for CMAPSS.
COLUMN_NAMES = (
    ["unit", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def load_raw(path):
    """Load a raw CMAPSS space-delimited file into a clean DataFrame."""
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMN_NAMES)
    return df


def load_rul(path):
    """Load the RUL ground-truth file (one value per test engine, in order)."""
    rul = pd.read_csv(path, header=None, names=["RUL"])
    rul["unit"] = rul.index + 1  # engine IDs are 1-indexed, in file order
    return rul


def add_rul_to_train(train_df):
    """
    Compute RUL for every row of the TRAINING data.

    Since training engines run to failure, the last cycle recorded for each
    engine IS its failure point. RUL at any earlier cycle is simply:
        RUL = (max cycle seen for this engine) - (current cycle)
    """
    max_cycle_per_unit = train_df.groupby("unit")["cycle"].transform("max")
    train_df = train_df.copy()
    train_df["RUL"] = max_cycle_per_unit - train_df["cycle"]
    return train_df


def add_rul_to_test(test_df, rul_df):
    """
    Compute RUL for every row of the TEST data.

    Test engines are cut off before failure. The RUL file tells us how many
    cycles were left at the LAST recorded cycle for each engine. For any
    earlier row, we just add back the cycles that hadn't happened yet:
        RUL(row) = RUL(last row) + (last_cycle - this_row's_cycle)
    """
    test_df = test_df.copy()
    last_cycle = test_df.groupby("unit")["cycle"].transform("max")
    rul_at_cutoff = test_df["unit"].map(rul_df.set_index("unit")["RUL"])
    test_df["RUL"] = rul_at_cutoff + (last_cycle - test_df["cycle"])
    return test_df


if __name__ == "__main__":
    train = load_raw("data/raw/train_FD001.txt")
    test = load_raw("data/raw/test_FD001.txt")
    rul = load_rul("data/raw/RUL_FD001.txt")

    train = add_rul_to_train(train)
    test = add_rul_to_test(test, rul)

    print("TRAIN shape:", train.shape)
    print("TEST shape:", test.shape)
    print("Number of training engines:", train["unit"].nunique())
    print("Number of test engines:", test["unit"].nunique())
    print("\nCycles per training engine (life spans) - summary:")
    print(train.groupby("unit")["cycle"].max().describe())
    print("\nSample rows:")
    print(train.head())

    train.to_csv("data/train_with_rul.csv", index=False)
    test.to_csv("data/test_with_rul.csv", index=False)
    print("\nSaved data/train_with_rul.csv and data/test_with_rul.csv")
