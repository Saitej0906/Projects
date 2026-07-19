"""
Step 5: Interpretability with SHAP
====================================
WHY interpretability matters for a supply-chain/ops audience: a forecast
that "just works" isn't good enough to act on -- planners need to know
WHICH drivers pushed a forecast up or down so they can sanity-check it and
explain inventory decisions to stakeholders (e.g. "we're ordering extra
stock because of the upcoming promo, not because the model is guessing").

How to read a SHAP plot in plain language:
- Each row in a SHAP summary plot is one FEATURE (e.g. Sales_lag_7).
- Each dot is one PREDICTION (one store-day) from the test set.
- The dot's horizontal position (SHAP value) says how much that feature
  pushed THIS SPECIFIC prediction up (right, positive) or down (left,
  negative) relative to the average forecast.
- The dot's COLOR shows whether the feature's actual value was high (red)
  or low (blue) for that row.
- So "red dots clustered on the right for Sales_lag_7" reads as: "when
  recent sales were high, the model pushed today's forecast up too" --
  i.e., strong short-term momentum in demand, which is intuitive and
  reassuring (if it looked backwards, that would be a red flag).

We also report plain global feature importance (mean absolute SHAP value)
as a simple ranked bar chart -- "which features matter most, on average,
across all predictions" -- since that's often the first thing an
interviewer or stakeholder will ask for.
"""
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
import shap

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
FIG_DIR = os.path.join(OUT_DIR, "figures")

FEATURES = ["Store", "DayOfWeek", "Promo", "StateHoliday", "SchoolHoliday",
            "StoreType", "Assortment", "CompetitionDistance",
            "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
            "Promo2", "Promo2SinceWeek", "Promo2SinceYear",
            "Month", "Quarter", "WeekOfYear", "Year", "IsWeekend",
            "IsStateHoliday", "IsSchoolHoliday", "IsPromo", "DaysSinceStart",
            "Sales_lag_1", "Sales_lag_7", "Sales_lag_14", "Sales_lag_30",
            "Sales_roll_mean_7", "Sales_roll_std_7",
            "Sales_roll_mean_30", "Sales_roll_std_30"]

model = xgb.XGBRegressor()
model.load_model(os.path.join(OUT_DIR, "models", "xgboost_model.json"))

df = pd.read_parquet(os.path.join(OUT_DIR, "features.parquet"))
df["Date"] = pd.to_datetime(df["Date"])
max_date = df["Date"].max()
test = df[df["Date"] > max_date - pd.Timedelta(days=42)]

# Sample for SHAP speed (SHAP on the full 40k-row test set is unnecessary;
# a representative random sample gives essentially the same picture much faster).
sample = test.sample(n=min(3000, len(test)), random_state=42)
X_sample = sample[FEATURES]

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# ---------------- Global feature importance (mean |SHAP|) ----------------
mean_abs_shap = np.abs(shap_values).mean(axis=0)
importance = pd.Series(mean_abs_shap, index=FEATURES).sort_values(ascending=False)

plt.figure(figsize=(8, 8))
importance.head(15).sort_values().plot(kind="barh", color="#4C72B0")
plt.title("Top 15 Features by Mean |SHAP value|\n(average impact on predicted sales)")
plt.xlabel("Mean |SHAP value| (impact on model output, in Sales units)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "08_shap_feature_importance.png"), dpi=110)
plt.close()

# ---------------- SHAP summary (beeswarm) plot ----------------
plt.figure()
shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "09_shap_summary_beeswarm.png"), dpi=110, bbox_inches="tight")
plt.close()

importance.to_csv(os.path.join(OUT_DIR, "reports", "feature_importance.csv"))

print("Top 10 features driving predictions (mean |SHAP value|):")
print(importance.head(10).to_string())
print("\nSaved SHAP importance bar chart and beeswarm summary plot to outputs/figures/")
