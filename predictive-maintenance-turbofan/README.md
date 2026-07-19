# Turbofan Engine Predictive Maintenance (NASA CMAPSS FD001)

Predicting Remaining Useful Life (RUL) of jet engines from sensor time-series data, plus a binary early-warning classifier for maintenance scheduling.

## Overview

This project uses NASA's CMAPSS turbofan degradation simulation dataset to build two connected models:

1. A **regression model** that predicts how many operating cycles are left before an engine fails, from its current sensor readings.
2. A **classification model** that flags "this engine is likely to fail within 30 cycles" as a simple yes/no early-warning signal a maintenance team could act on.

The goal is a defensible, end-to-end predictive maintenance pipeline: load noisy sensor data → engineer features that actually capture degradation → predict remaining life → convert that into an operational maintenance decision → explain what's driving the predictions.

## Problem Statement

Unplanned equipment failure is expensive and, in aerospace, safety-critical. The alternative — fixed-interval scheduled maintenance regardless of actual condition — wastes serviceable life and still doesn't fully prevent failures between inspections. **Condition-based maintenance** uses live sensor data to estimate how much useful life a machine has left, so maintenance can be scheduled just in time rather than too early or too late.

This mirrors a problem I worked on from the materials side during a fatigue-life research internship: estimating cycles-to-failure of a component from a damage/stress history. There, the "sensor" is a stress model and an S-N curve; here, the sensors are literal temperature, pressure, and speed measurements, and the "damage accumulation" shows up as drift in those readings. Same underlying question — *how much life is left, and how do we know before it's too late?* — different evidence.

## Dataset

**NASA CMAPSS FD001** (Commercial Modular Aero-Propulsion System Simulation), a widely used public benchmark for predictive maintenance research.

- **Training set:** 100 simulated turbofan engines, each run from healthy until failure. Every cycle (roughly one flight) is recorded, so the exact RUL at every point is known by construction (failure cycle − current cycle).
- **Test set:** 100 different engines, each with sensor data cut off at some point *before* failure.
- **RUL ground truth file:** the true remaining life of each test engine at its cutoff point — this is what the models are evaluated against.
- **26 columns per row:** engine unit ID, cycle number, 3 operational settings, and 21 sensor channels (temperatures, pressures, fan/core speeds, fuel flow, etc.).
- FD001 is the simplest of the four CMAPSS sub-datasets: single operating condition, single fault mode. This is reflected in the data — 1 operational setting and 6 of the 21 sensors are constant and carry no signal (confirmed in EDA, not assumed).

## Methodology

**1. Data loading** (`src/data_loading.py`) — Parses the raw space-delimited files, assigns column names, and computes RUL for every row: directly (`max_cycle − current_cycle`) for training engines that run to failure, and by adding the known cutoff-RUL back onto earlier cycles for test engines.

**2. Exploratory analysis** (`src/eda.py`) — Confirms there are no missing values (CMAPSS is a clean simulated dataset). Identifies and drops 1 operational setting and 6 sensors that are constant (zero variance → zero signal), leaving 15 informative sensors. Plots raw sensor trajectories against cycles-remaining to visually confirm degradation patterns, and ranks all sensors by correlation with RUL. The strongest single sensors (sensor_11, sensor_4, sensor_12, sensor_7, sensor_15) all show |correlation| with RUL around 0.64–0.70.

**3. Feature engineering** (`src/feature_engineering.py`) — For each of the 15 informative sensors, computes a rolling 5-cycle **mean**, **standard deviation**, and **trend** (least-squares slope), per engine. Rationale: a single raw reading is noisy; the recent trend is what actually signals accumulating wear — the same logic as tracking a damage-accumulation rate rather than one stress reading in fatigue analysis. This turns 15 raw sensors into 15 raw + 45 engineered = 69 total features.

**4. Regression model** (`src/regression_model.py`) — A Random Forest Regressor predicts RUL from all 69 features. Chosen because sensor-degradation relationships are non-linear, sensors interact, and Random Forest needs no feature scaling and yields feature importances for free. Training RUL is **clipped at 125 cycles**: early in life, before real degradation starts, exact RUL isn't learnable from flat sensor readings and just adds noise — clipping tells the model "healthy = long life" without forcing an arbitrary precise guess. This is standard practice in CMAPSS literature.

**5. Classification layer** (`src/classification_model.py`) — A Random Forest Classifier (`class_weight="balanced"` to handle the ~15% positive rate) predicts whether an engine is within **30 cycles** of failure. The 30-cycle horizon was chosen because that's roughly where the EDA shows sensors actually start visibly drifting from baseline — close to the earliest point degradation is detectable, while leaving lead time to act.

**6. Evaluation** (built into steps 4–5) — Regression is scored with RMSE/MAE on the last recorded cycle of each test engine (mirrors real deployment: "given everything up to now, how much life is left?"). Classification is scored with precision, recall, F1, and ROC-AUC, at both a default 0.5 threshold and a **recall-prioritized threshold** chosen to guarantee ≥90% recall.

**7. Feature importance** (`src/feature_importance.py`) — Extracts and plots each model's top features by mean-decrease-in-impurity.

## Why Recall Matters More Than Precision Here

- A **false negative** (missed failure) means an engine fails in service with no warning — unplanned downtime and, in aviation, a genuine safety risk. This is the outcome the whole system exists to prevent.
- A **false positive** (false alarm) costs an unnecessary inspection — real but bounded and safe.

Given that asymmetry, the classifier is deliberately tuned toward high recall even at a real precision cost, the same way a conservative safety factor in fatigue design accepts some parts retired earlier than strictly necessary because missing a real crack is unacceptable. The threshold isn't picked by feel — it's chosen from the actual precision/recall trade-off curve (`outputs/figures/precision_recall_tradeoff.png`) to hit a specific, statable recall target.

## Results

**Regression (RUL prediction), evaluated on the final cycle of each of the 100 test engines:**

| Metric | Value |
|---|---|
| RMSE | 17.7 cycles |
| MAE | 12.6 cycles |

For context, test engines have a typical remaining life on the order of tens to ~100+ cycles at cutoff, so an average error of ~13 cycles is a reasonably tight estimate — in line with published Random-Forest-class benchmarks on this dataset.

**Classification (failure-within-30-cycles flag), evaluated cycle-by-cycle across the full test set:**

| Threshold | Precision | Recall | F1 | Missed failures (FN) | False alarms (FP) |
|---|---|---|---|---|---|
| 0.50 (default) | 0.68 | 0.75 | 0.71 | 84 / 332 | 119 |
| 0.19 (recall-prioritized, target ≥90% recall) | 0.45 | 0.90 | 0.60 | 33 / 332 | 368 |

ROC-AUC: **0.99** (threshold-independent separability is excellent — the trade-off above is purely about *where* to draw the operating line, not whether the model can tell healthy from failing engines apart).

Lowering the threshold from 0.50 to 0.19 cuts missed real failures from 84 to 33 (a 61% reduction) at the cost of roughly 3x more false alarms — a trade a maintenance team would very likely take, since a false alarm is an extra inspection and a missed failure is an unplanned breakdown.

**Feature importance:** Across both models, the rolling **mean** features dominate (~94% of total importance in the regressor) over raw instantaneous readings, rolling std, and rolling trend. The smoothed current level of a sensor turns out to be more informative than either a single noisy reading or a short-term (5-cycle) slope. `sensor_4`, `sensor_11`, `sensor_9`, `sensor_15`, and `sensor_12` are consistently the top contributors in both models — full breakdown in `outputs/metrics/feature_importance_notes.txt` and the two `*_feature_importance.png` plots.

## Project Structure

```
pdm_project
data
raw
src
data_loading.py
eda.py
feature_engineering.py
regression_model.py
classification_model.py
feature_importance.py
main.py
outputs
figures
metrics
requirements.txt
README.md

## How to Run

```bash
pip install -r requirements.txt
python src/main.py
```

This runs every stage in order and regenerates all files in `data/*.csv` and `outputs/`. Each script can also be run individually (e.g. `python src/eda.py`) once the earlier stages it depends on have been run at least once.

## Future Improvements

1. **Sequence models.** Random Forest treats each row independently aside from the hand-built rolling features. An LSTM/GRU or 1D-CNN could learn temporal degradation patterns directly from the raw sequences, likely improving on the hand-engineered rolling-window approach.
2. **Extend to FD002–FD004.** The other CMAPSS sub-datasets add multiple operating conditions and/or multiple fault modes — a much closer match to real fleets, and a good test of whether this pipeline's feature engineering (currently tuned for FD001's single condition) generalizes.
3. **Cost-based threshold optimization.** The 30-cycle horizon and 90%-recall threshold were chosen from visual/statistical justification. With real inspection and downtime costs, the threshold could instead be optimized to directly minimize expected total cost, which would make the recall/precision trade-off even more concretely defensible.
