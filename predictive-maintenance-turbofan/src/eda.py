"""
eda.py
------
Basic exploratory analysis: which sensors actually move as engines degrade,
and which ones are dead weight (constant, or pure noise around a fixed
operating point)?

CMAPSS FD001 is a single-operating-condition, single-fault-mode dataset
(engines all fly the same "mission" and degrade via one fault mechanism).
This matters: a handful of the 21 sensors and all 3 operational settings are
essentially constant in FD001 (they only vary because of a different fault
mode / multiple flight conditions in the other CMAPSS sub-datasets, FD002-4).
Constant columns carry zero predictive information and just add noise to a
model, so we drop them here -- same logic as dropping a non-significant
stress-concentration factor from a fatigue model because it doesn't vary
across your test matrix.
"""

import pandas as pd
import matplotlib.pyplot as plt

SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
OPSET_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]


def find_constant_columns(df, cols, std_threshold=1e-5):
    """Flag columns whose standard deviation is ~0 -> no useful signal."""
    stds = df[cols].std()
    return stds[stds < std_threshold].index.tolist()


def plot_sensor_trends(train_df, sensors_to_plot, n_engines=6, outpath="outputs/figures/sensor_trends.png"):
    """
    Plot raw sensor readings vs. RUL (counting down to failure) for a
    handful of engines, so degradation trends are visible by eye.
    We plot against RUL (not cycle) so all engines are aligned at "0 = failure",
    regardless of how long each one actually lived.
    """
    fig, axes = plt.subplots(len(sensors_to_plot), 1, figsize=(9, 3 * len(sensors_to_plot)), sharex=True)
    sample_units = train_df["unit"].unique()[:n_engines]

    for ax, sensor in zip(axes, sensors_to_plot):
        for unit in sample_units:
            eng = train_df[train_df["unit"] == unit].sort_values("cycle")
            ax.plot(eng["RUL"], eng[sensor], alpha=0.7, linewidth=1)
        ax.set_ylabel(sensor)
        ax.invert_xaxis()  # so RUL=0 (failure) is on the right, like a countdown
    axes[-1].set_xlabel("Remaining Useful Life (cycles)")
    axes[0].set_title("Sensor readings vs. cycles remaining before failure (6 sample engines)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=120)
    plt.close()
    print(f"Saved {outpath}")


def plot_correlation_with_rul(train_df, outpath="outputs/figures/sensor_rul_correlation.png"):
    """Bar chart: how linearly correlated is each sensor with RUL?"""
    corrs = train_df[SENSOR_COLS + ["RUL"]].corr()["RUL"].drop("RUL").sort_values()
    plt.figure(figsize=(8, 7))
    corrs.plot(kind="barh", color=["#c0392b" if v < 0 else "#2980b9" for v in corrs])
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("Correlation of each raw sensor with RUL")
    plt.xlabel("Pearson correlation")
    plt.tight_layout()
    plt.savefig(outpath, dpi=120)
    plt.close()
    print(f"Saved {outpath}")
    return corrs


if __name__ == "__main__":
    train = pd.read_csv("data/train_with_rul.csv")

    print("=== Missing values ===")
    print(train.isnull().sum().sum(), "missing cells total (should be 0 - CMAPSS is a clean simulated dataset)")

    print("\n=== Constant / near-constant columns (no signal, will drop) ===")
    dead_opsets = find_constant_columns(train, OPSET_COLS)
    dead_sensors = find_constant_columns(train, SENSOR_COLS)
    print("Operational settings:", dead_opsets)
    print("Sensors:", dead_sensors)

    useful_sensors = [s for s in SENSOR_COLS if s not in dead_sensors]
    print(f"\n{len(useful_sensors)} of {len(SENSOR_COLS)} sensors carry signal:", useful_sensors)

    corrs = plot_correlation_with_rul(train)
    print("\nTop 5 sensors most correlated (any direction) with RUL:")
    print(corrs.reindex(corrs.abs().sort_values(ascending=False).index).head())

    # Pick a few of the strongest, visually clean degradation sensors to plot
    plot_sensor_trends(train, ["sensor_2", "sensor_4", "sensor_7", "sensor_11", "sensor_12", "sensor_15"])

    # Save the list of useful sensors so downstream scripts use the same set
    with open("outputs/metrics/useful_sensors.txt", "w") as f:
        f.write("\n".join(useful_sensors))
    print("\nSaved outputs/metrics/useful_sensors.txt")
