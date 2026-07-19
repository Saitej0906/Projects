# Demand Forecasting & Inventory Optimization — Rossmann Retail Stores

An end-to-end supply chain analytics project: forecasting daily store-level
demand and turning those forecasts into actionable inventory decisions
(safety stock, reorder points, stockout risk).

## 1. Problem Statement

Retailers with hundreds of stores need to answer two linked questions every
day:
1. **How much will each store sell tomorrow / next week?**
2. **Given that forecast (and its uncertainty), how much stock should each
   store hold, and when should it reorder, to avoid both stockouts (lost
   sales) and overstock (wasted holding cost)?**

This project builds a forecasting pipeline that answers (1) at the
individual store level, and a set of inventory formulas that turn those
forecasts into (2): safety stock, reorder point, recommended inventory
level, and expected stockout risk for every store.

## 2. Dataset

[Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales)
(Kaggle), consisting of:
- **train.csv** — 1,017,209 daily records across 1,115 stores (2013–2015):
  Sales, Customers, Open/Closed, Promo, StateHoliday, SchoolHoliday.
- **store.csv** — 1,115 rows of store metadata: StoreType, Assortment,
  CompetitionDistance, CompetitionOpenSince{Month,Year}, Promo2 (recurring
  promotions) and its start week/year/interval.

After merging and cleaning (see Methodology), the modeling dataset has
~810,000 open-store-days across 1,115 stores.

## 3. Methodology

### 3.1 Data Cleaning (`src/01_data_prep.py`)
- Dropped duplicate rows; coerced/validated dates.
- **Removed closed-store days** (Open==0) before modeling — closed days have
  Sales=0 by definition, not by demand, and would corrupt both the EDA and
  the forecasts if left in.
- **Store-level IQR outlier flagging** (not deletion) on Sales, since a
  single global outlier threshold would unfairly flag big stores' normal
  sales days given the 10–20x size difference between stores.
- Imputed `store.csv` nulls using business logic, not blind mean-fill:
  e.g. missing `Promo2SinceWeek/Year` means "store doesn't run Promo2" (fill
  0/""), not a genuinely missing measurement.
- Confirmed `Promo`, `StateHoliday`, `SchoolHoliday` are clean, low-cardinality
  categorical/binary columns usable directly as model features.

### 3.2 EDA (`src/02_eda.py`)
Ten figures in `outputs/figures/` cover: overall trend, day-of-week pattern,
monthly seasonality, promo effect (~39% average sales lift on promo days),
holiday effect, store-type variation, and store-to-store heterogeneity. Each
chart's business interpretation is printed in the script's output and
summarized in the code comments.

### 3.3 Feature Engineering (`src/03_feature_engineering.py`)
- **Calendar features**: DayOfWeek, Month, Quarter, WeekOfYear, IsWeekend —
  capture recurring weekly/yearly seasonality without needing sales history.
- **Promo/Holiday indicators** — known-in-advance business events with large
  measured effects on sales.
- **Lag features** (`Sales_lag_1/7/14/30`) — "what did this store sell N
  days ago", with lag-7 specifically capturing "same weekday last week".
- **Rolling mean & std (7-day, 30-day)** — smoothed demand level and
  volatility, both computed with `.shift(1)` before rolling so no feature
  ever uses same-day or future information (leakage-free by construction).
- **Trend feature** (`DaysSinceStart`) — lets the model represent gradual
  growth/decline in a store's baseline that cycles alone don't explain.

### 3.4 Modeling (`src/04_baseline_model.py`, `src/05_xgboost_model.py`)
- **Baseline: SARIMA(1,1,1)(1,1,1,7)** on network-wide aggregate daily sales
  — the kind of forecast a planner could produce with classical statistics
  alone, with no external features.
- **Advanced: XGBoost**, trained per-store using all engineered features,
  with a small time-based hyperparameter search (max_depth, learning_rate)
  validated on a held-out time slice — never a random split, to avoid
  leaking future information into training.
- **Chronological train/test split**: last 42 days held out as the test set
  for both models.

### 3.5 Interpretability (`src/07_interpretability.py`)
SHAP (TreeExplainer) values on a 3,000-row sample of the test set, producing
a global feature-importance bar chart and a beeswarm summary plot. How to
read it: each dot is one prediction; its horizontal position shows how much
that feature pushed that specific forecast up or down; its color shows
whether the feature's value was high (red) or low (blue) for that row.

### 3.6 Inventory Optimization (`src/08_inventory_optimization.py`)
Assumptions (documented, adjustable): **7-day lead time**, **95% service
level** (Z ≈ 1.645), **7-day review period**.

- `Safety Stock = Z × σ_demand × √(lead time)`
- `Reorder Point = (avg daily demand × lead time) + Safety Stock`
- `Recommended Inventory = Reorder Point + (avg daily demand × review period)`
- `Stockout Risk = 1 − P(lead-time demand ≤ Reorder Point)`, computed from a
  Normal(mean, std) model of lead-time demand rather than just asserting
  the target 5%, so it reflects each store's actual forecast and volatility.

## 4. Results — Model Comparison

| Model | Level | MAE | RMSE | MAPE |
|---|---|---|---|---|
| SARIMA (baseline) | Network aggregate | 1,094,530 | 1,339,821 | **29.0%** |
| XGBoost (advanced) | Per-store | 660 | 938 | **10.0%** |

**XGBoost cuts MAPE roughly in third vs. the classical baseline.** Why:
XGBoost pools information across all 1,115 stores and uses rich exogenous
features (promo, holiday, store type, competition, lag/rolling stats) that
SARIMA in its basic form cannot use at all — SARIMA only ever looks at one
series' own past values. XGBoost also captures non-linear interactions
(e.g. promo lift differing by day-of-week and store type) automatically,
and produces store-level forecasts directly, which is what real inventory
decisions require (you don't ship a network-average number of units to one
store).

Top predictive features by mean |SHAP value|: `Sales_roll_mean_30` (recent
demand level), `Promo`, `Sales_lag_1`, `DayOfWeek`, `DaysSinceStart`.

## 5. Repository Structure

```
rossmann-forecast/
├── data/                       # raw train.csv, store.csv (gitignored)
├── src/
│   ├── 01_data_prep.py
│   ├── 02_eda.py
│   ├── 03_feature_engineering.py
│   ├── 04_baseline_model.py
│   ├── 05_xgboost_model.py
│   ├── 06_model_comparison.py
│   ├── 07_interpretability.py
│   └── 08_inventory_optimization.py
├── outputs/
│   ├── figures/                # all charts (EDA, SHAP, inventory plan)
│   ├── models/                 # saved xgboost_model.json
│   └── reports/                # CSV results: metrics, predictions, inventory plan
├── requirements.txt
└── README.md
```

Run in order: `python src/01_data_prep.py` → `02_eda.py` →
`03_feature_engineering.py` → `04_baseline_model.py` → `05_xgboost_model.py`
→ `06_model_comparison.py` → `07_interpretability.py` →
`08_inventory_optimization.py`.

## 6. Future Improvements

1. **Deep learning forecasting (LSTM / Temporal Fusion Transformer)** —
   could capture longer-range and cross-store temporal patterns that tree
   models miss, at the cost of more training data and compute.
2. **Real-time / streaming pipeline** — retrain or fine-tune forecasts daily
   as new sales data lands, feeding directly into an automated
   replenishment system rather than a static batch run.
3. **External demand drivers** — weather, local events, and macroeconomic
   indicators (e.g. regional unemployment, fuel prices) could explain
   demand swings that store/promo/calendar features alone don't capture.
4. **Non-normal demand modeling** — replace the Normal-distribution safety
   stock assumption with empirical or Poisson/negative-binomial demand
   distributions for low-volume/slow-moving SKUs, where the normal
   approximation breaks down.
