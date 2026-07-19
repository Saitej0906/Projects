"""
Step 4b: Advanced Model -- XGBoost (per-store, feature-rich)
==============================================================
WHY XGBoost over the SARIMA baseline for THIS data:
- We have ~1,115 parallel time series (one per store) plus rich side
  information (promo, holiday, store type, competition distance...).
  XGBoost can learn ONE model across all stores simultaneously, sharing
  statistical strength ("stores of StoreType b behave like this") while
  still differentiating individual stores through the Store ID and
  store-level engineered features -- something a single-series SARIMA
  fundamentally cannot do.
- Tree models handle non-linear interactions natively (e.g. "promo effect
  is bigger on Mondays for StoreType a") without us having to manually
  specify them, unlike ARIMA which assumes a fairly rigid linear structure.
- Demand here is driven by many discrete, known-in-advance events (promo
  flags, holiday flags, day-of-week) rather than smooth autocorrelation
  alone -- exactly the setting where gradient boosting on tabular features
  tends to outperform classical time-series models.

Train/test split: chronological (train = data.max_date -42d, test = last
42 days) -- NEVER a random split, since a random split would let the model
"see the future" via random rows from the test period sitting in the lag
features of adjacent training rows, which silently inflates accuracy.

Hyperparameter tuning: a small, deliberately limited random/grid search
over a few key parameters (max_depth, learning_rate, n_estimators) using a
time-based validation slice, since exhaustive tuning is not the point of
this project -- demonstrating a sound, leak-free evaluation process is.
"""
import pandas as pd
import numpy as np
import os
import json
import itertools
import xgboost as xgb

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

df = pd.read_parquet(os.path.join(OUT_DIR, "features.parquet"))
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

FEATURES = ["Store", "DayOfWeek", "Promo", "StateHoliday", "SchoolHoliday",
            "StoreType", "Assortment", "CompetitionDistance",
            "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
            "Promo2", "Promo2SinceWeek", "Promo2SinceYear",
            "Month", "Quarter", "WeekOfYear", "Year", "IsWeekend",
            "IsStateHoliday", "IsSchoolHoliday", "IsPromo", "DaysSinceStart",
            "Sales_lag_1", "Sales_lag_7", "Sales_lag_14", "Sales_lag_30",
            "Sales_roll_mean_7", "Sales_roll_std_7",
            "Sales_roll_mean_30", "Sales_roll_std_30"]
TARGET = "Sales"

HOLDOUT_DAYS = 42
max_date = df["Date"].max()
split_date = max_date - pd.Timedelta(days=HOLDOUT_DAYS)
val_split_date = split_date - pd.Timedelta(days=HOLDOUT_DAYS)  # extra slice for tuning

train_full = df[df["Date"] <= split_date]
test = df[df["Date"] > split_date]

tune_train = train_full[train_full["Date"] <= val_split_date]
tune_val = train_full[train_full["Date"] > val_split_date]

print(f"Tuning train: {len(tune_train):,} rows | Tuning val: {len(tune_val):,} rows")
print(f"Final train: {len(train_full):,} rows | Test (holdout): {len(test):,} rows")

def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# ---------------- Small hyperparameter search ----------------
param_grid = {
    "max_depth": [6, 8],
    "learning_rate": [0.05, 0.1],
    "n_estimators": [300],
}
combos = list(itertools.product(*param_grid.values()))
best_rmse, best_params = np.inf, None

Xt, yt = tune_train[FEATURES], tune_train[TARGET]
Xv, yv = tune_val[FEATURES], tune_val[TARGET]

print("\n--- Hyperparameter tuning (time-based validation) ---")
for depth, lr, n_est in combos:
    m = xgb.XGBRegressor(max_depth=depth, learning_rate=lr, n_estimators=n_est,
                          subsample=0.9, colsample_bytree=0.9,
                          objective="reg:squarederror", tree_method="hist",
                          random_state=42, n_jobs=-1)
    m.fit(Xt, yt)
    preds = m.predict(Xv)
    rmse = np.sqrt(np.mean((yv.values - preds) ** 2))
    print(f"  depth={depth}, lr={lr}, n_est={n_est} -> val RMSE={rmse:,.0f}")
    if rmse < best_rmse:
        best_rmse, best_params = rmse, dict(max_depth=depth, learning_rate=lr, n_estimators=n_est)

print(f"\nBest params: {best_params} (val RMSE={best_rmse:,.0f})")

# ---------------- Fit final model on full training window with best params ----------------
final_model = xgb.XGBRegressor(**best_params, subsample=0.9, colsample_bytree=0.9,
                                objective="reg:squarederror", tree_method="hist",
                                random_state=42, n_jobs=-1)
final_model.fit(train_full[FEATURES], train_full[TARGET])

test_preds = final_model.predict(test[FEATURES])
test_preds = np.clip(test_preds, 0, None)

mae = np.mean(np.abs(test[TARGET].values - test_preds))
rmse = np.sqrt(np.mean((test[TARGET].values - test_preds) ** 2))
mape_val = mape(test[TARGET].values, test_preds)

print(f"\nXGBoost final model (per-store, {HOLDOUT_DAYS}-day holdout):")
print(f"  MAE:  {mae:,.0f}")
print(f"  RMSE: {rmse:,.0f}")
print(f"  MAPE: {mape_val:.2f}%")

# Save model, predictions, and results
final_model.save_model(os.path.join(OUT_DIR, "models", "xgboost_model.json"))

test_out = test[["Store", "Date", "Sales"]].copy()
test_out["Prediction"] = test_preds
test_out.to_csv(os.path.join(OUT_DIR, "reports", "xgboost_test_predictions.csv"), index=False)

results = pd.DataFrame({
    "model": ["XGBoost"], "level": ["per_store"],
    "MAE": [mae], "RMSE": [rmse], "MAPE": [mape_val]
})
results.to_csv(os.path.join(OUT_DIR, "reports", "xgboost_results.csv"), index=False)

with open(os.path.join(OUT_DIR, "reports", "xgboost_best_params.json"), "w") as f:
    json.dump(best_params, f, indent=2)

print("\nSaved model, test predictions, and results to outputs/")
