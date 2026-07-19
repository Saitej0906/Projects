"""
Step 6: Inventory Optimization
================================
We turn XGBoost demand forecasts into concrete inventory decisions using
classic, industry-standard formulas. All formulas assume demand during the
lead time is approximately normally distributed -- a standard, defensible
simplifying assumption for a resume project (real systems often use more
detailed distributions, but this is the textbook starting point and is
what most interviewers will expect you to know).

Key concepts, defined simply:
- Lead time (L): how many days it takes from placing a replenishment order
  to the stock arriving on the shelf. We ASSUME L = 7 days (typical for
  grocery/retail DC-to-store replenishment); this is a documented
  assumption, not a value from the data.
- Service level: the probability we do NOT run out of stock before the
  next delivery arrives. We assume a 95% service level, a common retail
  default (higher service level = more safety stock = higher holding cost
  but fewer lost sales).
- Z-score: the number of standard deviations above the mean that
  corresponds to our target service level under a normal distribution.
  For 95% service level, Z ~= 1.645.

Formulas:
1. Safety Stock = Z * sigma_demand * sqrt(L)
   -> Extra buffer stock to absorb demand variability during lead time.
      sigma_demand is the forecast demand's rolling standard deviation
      (from Sales_roll_std_30, our best available volatility estimate).
      We scale by sqrt(L) because uncertainty compounds over multiple days
      independently (variance adds; std dev grows with the square root of
      time), assuming day-to-day demand variability is roughly independent.

2. Reorder Point (ROP) = (avg daily demand * L) + Safety Stock
   -> The stock level at which a new order should be triggered so it
      arrives (on average) just as stock would otherwise run out, PLUS
      the safety buffer for variability.

3. Recommended Inventory Level = ROP + (avg daily demand * review_period)
   -> How much stock to hold/order up to, covering the reorder point plus
      one additional review cycle of demand (we assume a weekly, 7-day,
      review period -- i.e., store managers/DCs check and reorder weekly).

4. Expected Stockout Risk = 1 - service level (by construction, ~5% here)
   -> Or, more informatively, we back-calculate the ACTUAL probability of
      stockout given the CURRENT on-hand-equivalent (using the forecast +
      its uncertainty) vs the ROP, so it's not just a fixed assumption but
      a per-store number the model computes for the forecast period.

We use the XGBoost forecasts (test period) as our "future demand" input,
and Sales_roll_std_30 as the volatility input for safety stock -- this
directly ties Part 6 back to Part 4's model outputs.
"""
import pandas as pd
import numpy as np
import os
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
FIG_DIR = os.path.join(OUT_DIR, "figures")

# ---------------- Assumptions (documented, adjustable) ----------------
LEAD_TIME_DAYS = 7
SERVICE_LEVEL = 0.95
REVIEW_PERIOD_DAYS = 7
Z = norm.ppf(SERVICE_LEVEL)  # ~1.645

print(f"Assumptions: lead time = {LEAD_TIME_DAYS} days, "
      f"service level = {SERVICE_LEVEL*100:.0f}% (Z={Z:.3f}), "
      f"review period = {REVIEW_PERIOD_DAYS} days")

preds = pd.read_csv(os.path.join(OUT_DIR, "reports", "xgboost_test_predictions.csv"),
                     parse_dates=["Date"])

feat = pd.read_parquet(os.path.join(OUT_DIR, "features.parquet"),
                        columns=["Store", "Date", "Sales_roll_std_30", "Sales_roll_mean_30"])
feat["Date"] = pd.to_datetime(feat["Date"])

merged = preds.merge(feat, on=["Store", "Date"], how="left")

# Volatility input: use the store's rolling 30-day std as our estimate of
# demand variability (fallback to a small positive floor to avoid zero
# safety stock for very stable/new stores, which would be unrealistic).
merged["sigma_demand"] = merged["Sales_roll_std_30"].fillna(merged["Sales_roll_std_30"].median())
merged["sigma_demand"] = merged["sigma_demand"].clip(lower=1)

merged["avg_daily_demand"] = merged["Prediction"]

merged["Safety_Stock"] = Z * merged["sigma_demand"] * np.sqrt(LEAD_TIME_DAYS)
merged["Reorder_Point"] = merged["avg_daily_demand"] * LEAD_TIME_DAYS + merged["Safety_Stock"]
merged["Recommended_Inventory"] = merged["Reorder_Point"] + merged["avg_daily_demand"] * REVIEW_PERIOD_DAYS

# Expected stockout risk: probability that ACTUAL demand over lead time
# exceeds the reorder point, given our normal-demand assumption.
# demand_over_LT ~ N(avg_daily_demand * L, sigma_demand^2 * L)
lt_mean = merged["avg_daily_demand"] * LEAD_TIME_DAYS
lt_std = merged["sigma_demand"] * np.sqrt(LEAD_TIME_DAYS)
merged["Stockout_Risk_pct"] = (1 - norm.cdf(merged["Reorder_Point"], loc=lt_mean, scale=lt_std)) * 100

inventory_plan = merged[["Store", "Date", "Sales", "Prediction", "sigma_demand",
                          "Safety_Stock", "Reorder_Point", "Recommended_Inventory",
                          "Stockout_Risk_pct"]].rename(columns={"Prediction": "Forecast_Demand",
                                                                  "Sales": "Actual_Demand"})
inventory_plan.to_csv(os.path.join(OUT_DIR, "reports", "inventory_plan.csv"), index=False)
print(f"\nSaved per-store, per-day inventory plan for {inventory_plan['Store'].nunique()} "
      f"stores to outputs/reports/inventory_plan.csv")

print("\nSample inventory plan (Store 1):")
print(inventory_plan[inventory_plan.Store == 1].head(10).to_string(index=False))

# ---------------- Visualization: forecast demand vs reorder point over time ----------------
example_store = 1
store_data = inventory_plan[inventory_plan["Store"] == example_store].sort_values("Date")

plt.figure(figsize=(13, 5))
plt.plot(store_data["Date"], store_data["Forecast_Demand"], label="Forecasted Demand",
         color="#4C72B0", linewidth=1.5)
plt.plot(store_data["Date"], store_data["Actual_Demand"], label="Actual Demand",
         color="#999999", linestyle="--", linewidth=1, alpha=0.7)
plt.plot(store_data["Date"], store_data["Reorder_Point"], label="Reorder Point",
         color="#C44E52", linewidth=1.5)
plt.plot(store_data["Date"], store_data["Recommended_Inventory"], label="Recommended Inventory Level",
         color="#55A868", linestyle=":", linewidth=1.5)
plt.title(f"Store {example_store}: Forecasted Demand vs Inventory Recommendations")
plt.xlabel("Date"); plt.ylabel("Units (Sales)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "10_inventory_plan_store1.png"), dpi=110)
plt.close()
print(f"\nSaved forecast-vs-inventory chart for Store {example_store} to outputs/figures/")

# ---------------- Network-level summary ----------------
summary = inventory_plan.groupby("Store").agg(
    avg_forecast_demand=("Forecast_Demand", "mean"),
    avg_safety_stock=("Safety_Stock", "mean"),
    avg_reorder_point=("Reorder_Point", "mean"),
    avg_recommended_inventory=("Recommended_Inventory", "mean"),
    avg_stockout_risk_pct=("Stockout_Risk_pct", "mean"),
).reset_index()
summary.to_csv(os.path.join(OUT_DIR, "reports", "inventory_summary_by_store.csv"), index=False)
print(f"\nSaved network-wide per-store inventory summary "
      f"({summary.shape[0]} stores) to outputs/reports/inventory_summary_by_store.csv")
print("\nNetwork averages:")
print(summary.drop(columns="Store").mean().to_string())
