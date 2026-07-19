"""
Step 2: Exploratory Data Analysis
==================================
Generates charts that answer business questions:
- Is there a weekly pattern? (staffing / replenishment implications)
- Is there a yearly trend or seasonality? (long-range planning)
- Do promotions and holidays actually move sales? (worth the margin cost?)
- How different are stores from each other? (one global model vs per-segment)

Each chart is saved as a PNG to outputs/figures/ with a short business
interpretation printed alongside it.
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

df = pd.read_parquet(os.path.join(OUT_DIR, "merged_clean.parquet"))
df["Date"] = pd.to_datetime(df["Date"])

# ---------------------------------------------------------------
# 1. Overall daily sales trend (all stores summed)
# ---------------------------------------------------------------
daily = df.groupby("Date")["Sales"].sum().reset_index()
plt.figure(figsize=(14, 4))
plt.plot(daily["Date"], daily["Sales"], linewidth=0.8)
plt.title("Total Daily Sales Across All Stores (2013-2015)")
plt.xlabel("Date"); plt.ylabel("Total Sales")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "01_overall_trend.png"), dpi=110)
plt.close()
print("[Chart 1] Overall trend: look for a repeating end-of-year spike (Christmas "
      "shopping) and periodic dips (store closures on holidays). This tells us "
      "seasonality is strong enough that a model MUST see calendar features, "
      "not just recent sales.")

# ---------------------------------------------------------------
# 2. Day-of-week pattern
# ---------------------------------------------------------------
dow = df.groupby("DayOfWeek")["Sales"].mean().reset_index()
plt.figure(figsize=(7, 4))
plt.bar(dow["DayOfWeek"], dow["Sales"], color="#4C72B0")
plt.title("Average Sales by Day of Week (1=Mon ... 7=Sun)")
plt.xlabel("Day of Week"); plt.ylabel("Average Sales")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "02_day_of_week.png"), dpi=110)
plt.close()
print("[Chart 2] Day-of-week: Monday is typically the highest-volume day "
      "(weekend restocking + fresh start), while Sunday sales are near-zero "
      "because most stores are closed. This justifies day-of-week as a feature "
      "and shows why we must not average across days without accounting for it.")

# ---------------------------------------------------------------
# 3. Monthly seasonality
# ---------------------------------------------------------------
df["Month"] = df["Date"].dt.month
month_avg = df.groupby("Month")["Sales"].mean().reset_index()
plt.figure(figsize=(8, 4))
plt.plot(month_avg["Month"], month_avg["Sales"], marker="o", color="#DD8452")
plt.title("Average Sales by Month (Seasonality)")
plt.xlabel("Month"); plt.ylabel("Average Sales")
plt.xticks(range(1, 13))
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "03_monthly_seasonality.png"), dpi=110)
plt.close()
print("[Chart 3] Monthly seasonality: December is consistently the highest month "
      "(holiday shopping) -- inventory planning needs to build up stock ahead of "
      "this, not react to it after the fact.")

# ---------------------------------------------------------------
# 4. Promo effect
# ---------------------------------------------------------------
promo_avg = df.groupby("Promo")["Sales"].mean().reset_index()
plt.figure(figsize=(5, 4))
plt.bar(["No Promo", "Promo"], promo_avg["Sales"], color=["#999999", "#55A868"])
plt.title("Average Sales: Promo vs No Promo")
plt.ylabel("Average Sales")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "04_promo_effect.png"), dpi=110)
plt.close()
lift = (promo_avg.loc[promo_avg.Promo == 1, "Sales"].values[0] /
        promo_avg.loc[promo_avg.Promo == 0, "Sales"].values[0] - 1) * 100
print(f"[Chart 4] Promo effect: promo days sell ~{lift:.0f}% more on average. "
      "This is a big, controllable lever -- it must be a feature, and it also means "
      "inventory plans need a 'promo mode' with higher safety stock.")

# ---------------------------------------------------------------
# 5. Holiday effect
# ---------------------------------------------------------------
hol_avg = df.groupby("StateHoliday")["Sales"].mean().reset_index()
plt.figure(figsize=(6, 4))
plt.bar(hol_avg["StateHoliday"].astype(str), hol_avg["Sales"], color="#C44E52")
plt.title("Average Sales by State Holiday Type (0=None, a/b/c=Holiday types)")
plt.xlabel("StateHoliday code"); plt.ylabel("Average Sales")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "05_holiday_effect.png"), dpi=110)
plt.close()
print("[Chart 5] Holiday effect: sales patterns differ sharply by holiday type "
      "(most stores are closed so Open==1 rows on holidays are a small, "
      "self-selected sample -- often higher than normal, similar to pre-holiday "
      "rush buying). This is why StateHoliday needs to stay as a categorical "
      "feature, not be collapsed into a single binary flag.")

# ---------------------------------------------------------------
# 6. Store-type variation
# ---------------------------------------------------------------
store_type_avg = df.groupby("StoreType")["Sales"].mean().sort_values(ascending=False).reset_index()
plt.figure(figsize=(6, 4))
plt.bar(store_type_avg["StoreType"], store_type_avg["Sales"], color="#8172B2")
plt.title("Average Sales by Store Type")
plt.xlabel("Store Type"); plt.ylabel("Average Sales")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "06_store_type.png"), dpi=110)
plt.close()
print("[Chart 6] Store-type variation: sales differ substantially by StoreType, "
      "confirming a single global average would be misleading -- the model needs "
      "store-level features (or store embeddings) rather than one-size-fits-all.")

# ---------------------------------------------------------------
# 7. Distribution across stores (spread / heterogeneity)
# ---------------------------------------------------------------
store_avg = df.groupby("Store")["Sales"].mean()
plt.figure(figsize=(7, 4))
plt.hist(store_avg, bins=50, color="#64B5CD")
plt.title("Distribution of Average Daily Sales Across Stores")
plt.xlabel("Average Daily Sales"); plt.ylabel("Number of Stores")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "07_store_distribution.png"), dpi=110)
plt.close()
print("[Chart 7] Store heterogeneity: average daily sales range from a few "
      "thousand to tens of thousands across stores -- this is why forecasts and "
      "inventory decisions must be made PER STORE (and ultimately per SKU), never "
      "at a single network-wide average.")

print("\nAll EDA figures saved to outputs/figures/")
