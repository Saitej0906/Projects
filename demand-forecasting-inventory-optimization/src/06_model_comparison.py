"""
Step 4c: Model Comparison
==========================
Puts the SARIMA baseline and XGBoost results side by side.

NOTE on comparability: SARIMA was evaluated on NETWORK-WIDE aggregate daily
sales (one number per day, summed across all stores), while XGBoost was
evaluated PER STORE (one number per store per day). Their absolute MAE/RMSE
are on different scales for that reason (aggregate totals are much bigger
numbers than single-store sales) -- so MAPE (a scale-free percentage error)
is the fair number to compare here, and it clearly favors XGBoost.

WHY XGBoost wins for this dataset (explained for an interview):
1. Cross-sectional information: XGBoost pools patterns across 1,115 stores
   and learns how features like Promo, StoreType, and CompetitionDistance
   interact with demand. SARIMA only ever looks at one series' own past.
2. Rich exogenous features: Promo and holiday effects are known FUTURE
   information (we know if a promo is planned) -- XGBoost uses this
   directly as an input feature. SARIMA in this basic form doesn't use any
   exogenous regressors at all, so it's blind to promo/holiday timing.
3. Non-linear interactions: e.g., the promo lift differs by day-of-week and
   store type. Tree splits capture this automatically; SARIMA's linear
   structure cannot.
4. Granularity: retail inventory decisions must be made per store (you
   don't ship a network-average number of units to a specific store).
   XGBoost naturally produces store-level forecasts; SARIMA at this level
   would require 1,115 separate models, each with less data and no shared
   learning.

WHEN SARIMA (or SARIMAX with exogenous regressors) would still be a
reasonable choice: very short, clean, single-series problems with strong
autocorrelation and few external drivers, or when full interpretability of
a simple statistical model matters more than raw accuracy.
"""
import pandas as pd
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "reports")

baseline = pd.read_csv(os.path.join(OUT_DIR, "baseline_results.csv"))
xgb_res = pd.read_csv(os.path.join(OUT_DIR, "xgboost_results.csv"))

comparison = pd.concat([baseline, xgb_res], ignore_index=True)
print("=== Model Comparison ===")
print(comparison.to_string(index=False))

comparison.to_csv(os.path.join(OUT_DIR, "model_comparison.csv"), index=False)
print(f"\nSaved comparison table to {OUT_DIR}/model_comparison.csv")

improvement = (baseline["MAPE"].values[0] - xgb_res["MAPE"].values[0])
print(f"\nXGBoost improves MAPE by {improvement:.1f} percentage points vs the "
      f"SARIMA baseline ({baseline['MAPE'].values[0]:.1f}% -> {xgb_res['MAPE'].values[0]:.1f}%).")
