"""
Step 1: Data Preparation
=========================
Loads train.csv (daily sales per store) and store.csv (store metadata),
merges them, and cleans the result.

WHY these specific cleaning decisions:
- We only keep rows where the store was Open==1. Closed-store days have
  Sales==0 by definition, not because of low demand. If we leave them in,
  the model learns "day of week X = zero sales" instead of learning real
  demand patterns, which corrupts both EDA and forecasting.
- CompetitionDistance / Promo2 nulls are structural, not random. A null in
  CompetitionOpenSinceYear usually means "no known competitor", not a
  missing measurement. So we impute with reasonable business-logic values
  instead of the column mean.
- We keep Sales==0 while Open==1 only if very rare (real stockout/anomaly
  days); we flag but do not delete them, since deleting real days breaks
  the time series continuity that lag/rolling features need.
- Outliers in Sales are handled with a store-level IQR cap rather than a
  global cap, because store sizes vary by 10-20x; a single global threshold
  would wrongly flag big stores' normal days as outliers.
"""
import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

def load_data():
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"),
                         parse_dates=["Date"], low_memory=False,
                         dtype={"StateHoliday": str})
    store = pd.read_csv(os.path.join(DATA_DIR, "store.csv"))
    return train, store

def clean_train(train):
    before = len(train)
    train = train.drop_duplicates()
    print(f"Dropped {before - len(train)} duplicate rows.")

    # Dates: coerce anything unparsable to NaT, then drop (should be ~0 rows
    # for Rossmann, but this makes the pipeline robust to messier real data).
    train["Date"] = pd.to_datetime(train["Date"], errors="coerce")
    bad_dates = train["Date"].isna().sum()
    if bad_dates:
        print(f"Dropping {bad_dates} rows with unparsable dates.")
        train = train.dropna(subset=["Date"])

    # Business rule: exclude closed-store days from the modeling set.
    # We keep a copy of the raw (incl. closed days) for EDA context only.
    train_open = train[train["Open"] == 1].copy()

    # Sales should never be negative; clip any negative noise to 0.
    neg = (train_open["Sales"] < 0).sum()
    if neg:
        print(f"Clipping {neg} negative Sales values to 0.")
        train_open.loc[train_open["Sales"] < 0, "Sales"] = 0

    # Store-level IQR outlier flagging (not deletion) -- we keep the data
    # point but add a flag column, so modeling can decide whether to
    # downweight it later, and so we don't silently erase real demand spikes
    # (e.g. a genuine promo-driven surge).
    q1 = train_open.groupby("Store")["Sales"].transform(lambda s: s.quantile(0.25))
    q3 = train_open.groupby("Store")["Sales"].transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1
    lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
    train_open["is_sales_outlier"] = ((train_open["Sales"] < lower) | (train_open["Sales"] > upper)).astype(int)
    print(f"Flagged {train_open['is_sales_outlier'].sum()} store-level Sales outliers "
          f"({train_open['is_sales_outlier'].mean()*100:.2f}% of open-day rows).")

    train_open["StateHoliday"] = train_open["StateHoliday"].replace({0: "0"}).astype(str)
    return train_open

def clean_store(store):
    store = store.drop_duplicates()

    # CompetitionDistance: a genuine missing value (few rows). Median impute
    # since distance is right-skewed and median is robust to that skew.
    med_dist = store["CompetitionDistance"].median()
    store["CompetitionDistance"] = store["CompetitionDistance"].fillna(med_dist)

    # CompetitionOpenSince{Month,Year}: null == "no competitor on record".
    # We don't want to impute a fake competitor open-date, so instead we
    # derive a binary "has_competitor_info" flag downstream and fill these
    # with a neutral placeholder that won't create a false "long-open
    # competitor" signal.
    store["CompetitionOpenSinceMonth"] = store["CompetitionOpenSinceMonth"].fillna(0)
    store["CompetitionOpenSinceYear"] = store["CompetitionOpenSinceYear"].fillna(0)

    # Promo2SinceWeek/Year and PromoInterval are null exactly when Promo2==0
    # (store doesn't run the recurring promo) -- fill with 0 / "" which is
    # the correct business meaning, not a guess.
    store["Promo2SinceWeek"] = store["Promo2SinceWeek"].fillna(0)
    store["Promo2SinceYear"] = store["Promo2SinceYear"].fillna(0)
    store["PromoInterval"] = store["PromoInterval"].fillna("")

    return store

def merge_and_save():
    train, store = load_data()
    print(f"Raw train shape: {train.shape}, store shape: {store.shape}")

    train_clean = clean_train(train)
    store_clean = clean_store(store)

    merged = train_clean.merge(store_clean, on="Store", how="left")
    print(f"Merged shape: {merged.shape}")

    # Sanity check: confirm holiday/promo columns are usable as features
    print("\n--- Feature usability check ---")
    print("StateHoliday values:", merged["StateHoliday"].unique())
    print("SchoolHoliday values:", merged["SchoolHoliday"].unique())
    print("Promo values:", merged["Promo"].unique())
    print("Promo2 values:", merged["Promo2"].unique())
    print("Nulls remaining per column:\n", merged.isna().sum()[merged.isna().sum() > 0])

    os.makedirs(OUT_DIR, exist_ok=True)
    merged.to_parquet(os.path.join(OUT_DIR, "merged_clean.parquet"), index=False)
    print(f"\nSaved cleaned merged dataset to outputs/merged_clean.parquet "
          f"({merged.shape[0]:,} rows, {merged.shape[1]} cols)")
    return merged

if __name__ == "__main__":
    merge_and_save()
