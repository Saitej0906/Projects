"""
Step 3: Feature Engineering
============================
We build features that give the model memory of the past (lags, rolling
stats) and awareness of calendar structure (day/month/quarter/trend), since
tree models like XGBoost have no built-in notion of "time" -- they only see
whatever numeric columns we hand them.

Feature groups and WHY each helps:

1. Calendar features (DayOfWeek, Month, Quarter, WeekOfYear, IsWeekend)
   -> Capture recurring, predictable seasonality (weekly + yearly cycles)
      seen in the EDA charts, without needing any sales history.

2. Promo / Holiday indicators (already in data, reformatted)
   -> Encode the known, controllable business events that shift demand by
      double digits (see EDA promo effect). Since these are usually known
      in advance (promos are planned, holidays are on the calendar), they
      are legitimate features even for forecasting the future.

3. Lag features (Sales_lag_1, _lag_7, _lag_14, _lag_30)
   -> "What did this exact store sell N days ago?" Lag-7 in particular
      captures "same weekday last week", which is a very strong predictor
      because of the weekly cycle we saw in EDA.

4. Rolling mean / std (7-day, 30-day windows)
   -> Rolling mean smooths out day-to-day noise to show the underlying
      demand LEVEL; rolling std captures how VOLATILE that store's demand
      currently is, which feeds directly into safety stock later (more
      volatile stores need bigger safety buffers).

5. Trend feature (days since start of series, per store)
   -> Lets the model represent slow, gradual growth/decline in a store's
      baseline demand that isn't explained by weekly/yearly cycles alone.

IMPORTANT leakage rule: every lag/rolling feature is computed using only
PAST data relative to each row (shift(1) before rolling), so the model
never "sees the future" when predicting a given day -- this mirrors exactly
what would be available at real prediction time.
"""
import pandas as pd
import numpy as np
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

df = pd.read_parquet(os.path.join(OUT_DIR, "merged_clean.parquet"))
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Store", "Date"]).reset_index(drop=True)

# ---------------- Calendar features ----------------
df["DayOfWeek"] = df["Date"].dt.dayofweek + 1  # keep Rossmann's 1-7 Mon-Sun convention
df["Month"] = df["Date"].dt.month
df["Quarter"] = df["Date"].dt.quarter
df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
df["Year"] = df["Date"].dt.year
df["IsWeekend"] = df["DayOfWeek"].isin([6, 7]).astype(int)

# ---------------- Holiday / Promo indicators ----------------
df["IsStateHoliday"] = (df["StateHoliday"] != "0").astype(int)
df["IsSchoolHoliday"] = df["SchoolHoliday"].astype(int)
df["IsPromo"] = df["Promo"].astype(int)

# ---------------- Trend feature (per store, days since first record) ----------------
first_date = df.groupby("Store")["Date"].transform("min")
df["DaysSinceStart"] = (df["Date"] - first_date).dt.days

# ---------------- Lag features (per store, using only past data) ----------------
g = df.groupby("Store")["Sales"]
for lag in [1, 7, 14, 30]:
    df[f"Sales_lag_{lag}"] = g.shift(lag)

# ---------------- Rolling mean / std (shifted by 1 so "today" isn't in its own window) ----------------
shifted = df.groupby("Store")["Sales"].shift(1)
df["Sales_roll_mean_7"] = shifted.groupby(df["Store"]).transform(lambda s: s.rolling(7, min_periods=3).mean())
df["Sales_roll_std_7"] = shifted.groupby(df["Store"]).transform(lambda s: s.rolling(7, min_periods=3).std())
df["Sales_roll_mean_30"] = shifted.groupby(df["Store"]).transform(lambda s: s.rolling(30, min_periods=7).mean())
df["Sales_roll_std_30"] = shifted.groupby(df["Store"]).transform(lambda s: s.rolling(30, min_periods=7).std())

# ---------------- Encode remaining categoricals numerically ----------------
for col in ["StoreType", "Assortment", "StateHoliday"]:
    df[col] = df[col].astype("category").cat.codes

# Drop the earliest rows per store where lag/rolling features are NaN
# (there is no "past" for them yet -- keeping them would force the model
# to learn from fabricated/imputed history, which is worse than dropping).
before = len(df)
df = df.dropna(subset=["Sales_lag_30", "Sales_roll_mean_30"])
print(f"Dropped {before - len(df)} early-history rows lacking full lag/rolling context.")

df.to_parquet(os.path.join(OUT_DIR, "features.parquet"), index=False)
print(f"Saved engineered feature set: {df.shape[0]:,} rows, {df.shape[1]} columns")
print("Feature columns:", [c for c in df.columns if c not in
      ["Sales", "Date", "Customers", "PromoInterval"]])
