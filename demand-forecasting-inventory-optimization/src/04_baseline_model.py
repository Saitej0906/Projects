"""
Step 4a: Baseline Model -- SARIMA on total daily sales
========================================================
WHY a classical time-series model as the baseline, and why aggregate:
- ARIMA/SARIMA models a SINGLE series using only its own past values and
  a fixed seasonal cycle. They cannot natively take in hundreds of extra
  columns (promo, holiday, store type, etc.), so per-store ARIMA for 1000+
  stores individually is both slow and doesn't use the cross-store
  information a tree model can exploit.
- We therefore fit SARIMA on the NETWORK-WIDE total daily sales series.
  This gives a fair, honest "simple statistical method" baseline: the kind
  of forecast a planner could produce in Excel/traditional stats software
  before reaching for machine learning. Any ML model we build afterward
  needs to beat this to justify its added complexity.
- Seasonal order (1,1,1,7) captures a weekly (7-day) seasonal cycle, which
  the EDA showed is the dominant repeating pattern.

We evaluate on the same holdout window (last 6 weeks) that the XGBoost
model uses, aggregated the same way, so the comparison is apples-to-apples.
"""
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")
from statsmodels.tsa.statespace.sarimax import SARIMAX

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

df = pd.read_parquet(os.path.join(OUT_DIR, "merged_clean.parquet"))
df["Date"] = pd.to_datetime(df["Date"])

daily = df.groupby("Date")["Sales"].sum().asfreq("D").ffill()

HOLDOUT_DAYS = 42  # 6 weeks, matches the XGBoost test split
train = daily.iloc[:-HOLDOUT_DAYS]
test = daily.iloc[-HOLDOUT_DAYS:]

print(f"Training SARIMA on {len(train)} days, testing on {len(test)} days...")

model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7),
                 enforce_stationarity=False, enforce_invertibility=False)
fit = model.fit(disp=False)

forecast = fit.get_forecast(steps=HOLDOUT_DAYS).predicted_mean
forecast.index = test.index

def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

mae = np.mean(np.abs(test.values - forecast.values))
rmse = np.sqrt(np.mean((test.values - forecast.values) ** 2))
mape_val = mape(test.values, forecast.values)

print(f"\nSARIMA baseline (aggregate daily sales, {HOLDOUT_DAYS}-day holdout):")
print(f"  MAE:  {mae:,.0f}")
print(f"  RMSE: {rmse:,.0f}")
print(f"  MAPE: {mape_val:.2f}%")

results = pd.DataFrame({
    "model": ["SARIMA_baseline"],
    "level": ["aggregate_network_total"],
    "MAE": [mae], "RMSE": [rmse], "MAPE": [mape_val]
})
results.to_csv(os.path.join(OUT_DIR, "reports", "baseline_results.csv"), index=False)

fc_df = pd.DataFrame({"Date": test.index, "actual": test.values, "forecast": forecast.values})
fc_df.to_csv(os.path.join(OUT_DIR, "reports", "baseline_forecast.csv"), index=False)
print("\nSaved baseline results and forecast to outputs/reports/")
