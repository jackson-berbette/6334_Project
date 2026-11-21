
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load datasets
sales = pd.read_csv("Data/filtered_sales_evaluation.csv")
calendar = pd.read_csv("Data/calendar.csv")
prices = pd.read_csv("Data/filtered_prices.csv")

calendar["date"] = pd.to_datetime(calendar["date"])

# Convert wide sales data into long format
day_cols = [col for col in sales.columns if col.startswith("d_")]

sales_long = sales.melt(
    id_vars=["item_id", "store_id", "state_id"],
    value_vars=day_cols,
    var_name="d",
    value_name="sales"
)

# Merge with calendar information
df = sales_long.merge(calendar, on="d", how="left")

# Merge weekly selling prices
df = df.merge(
    prices,
    on=["store_id", "item_id", "wm_yr_wk"],
    how="left"
)

# Exclude reserved test period
test_start = pd.Timestamp("2016-02-01")
df = df[df["date"] < test_start].copy()

# Create useful features
df["day_of_week"] = df["date"].dt.day_name()
df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year

# California SNAP indicator
df["snap"] = df["snap_CA"]

# Holiday/event indicator
df["holiday"] = (
    df["event_name_1"].notna() |
    df["event_name_2"].notna()
).astype(int)

df = df.sort_values(["item_id", "date"]).reset_index(drop=True)

products = df["item_id"].unique()

# calculate summary stats
summary = df.groupby("item_id")["sales"].agg(
    mean_sales="mean",
    median_sales="median",
    std_sales="std",
    max_sales="max",
    total_sales="sum",
    zero_days=lambda x: (x == 0).sum(),
    zero_pct=lambda x: 100 * (x == 0).mean()
)

summary["cv"] = (
    summary["std_sales"] / summary["mean_sales"]
)

print(summary.round(2).to_string())


# plot daily sales over time
fig, axes = plt.subplots(
    5, 1, figsize=(14, 16), sharex=True
)

for ax, product in zip(axes, products):
    data = df[df["item_id"] == product].copy()

    data["rolling_mean"] = (
        data["sales"].rolling(30, min_periods=1).mean()
    )

    ax.plot(
        data["date"], data["sales"],
        alpha=0.35, linewidth=0.7,
        label="Daily sales"
    )

    ax.plot(
        data["date"], data["rolling_mean"],
        color="red", linewidth=2,
        label="30-day moving average"
    )

    ax.set_title(product)
    ax.set_ylabel("Units sold")
    ax.legend()
    ax.grid(alpha=0.3)

axes[-1].set_xlabel("Date")
plt.suptitle("Daily Sales by Product", fontsize=16)
plt.tight_layout()
plt.show()


# analyze weekly seasonality

day_order = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday"
]

weekly = (
    df.groupby(["item_id", "day_of_week"])["sales"]
    .mean()
    .unstack()
    .reindex(columns=day_order)
)

fig, axes = plt.subplots(5, 1, figsize=(12, 16))

for ax, product in zip(axes, products):
    weekly.loc[product].plot(
        kind="bar", ax=ax, color="steelblue"
    )

    ax.set_title(product)
    ax.set_ylabel("Average daily sales")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.3)

plt.suptitle("Average Sales by Day of Week", fontsize=16)
plt.tight_layout()
plt.show()

print(weekly.round(2))


# analyze yearly seasonality

monthly = (
    df.groupby(["item_id", "month"])["sales"]
    .mean()
    .unstack()
    .reindex(columns=range(1, 13))
)

fig, axes = plt.subplots(5, 1, figsize=(12, 16))

for ax, product in zip(axes, products):
    ax.plot(
        monthly.columns,
        monthly.loc[product],
        marker="o"
    )

    ax.set_title(product)
    ax.set_ylabel("Average daily sales")
    ax.set_xticks(range(1, 13))
    ax.grid(alpha=0.3)

plt.suptitle("Monthly Sales Patterns", fontsize=16)
plt.tight_layout()
plt.show()

# Monthly patterns separated by year
monthly_year = (
    df.groupby(["item_id", "year", "month"])["sales"]
    .mean()
    .reset_index()
)

fig, axes = plt.subplots(5, 1, figsize=(12, 17))

for ax, product in zip(axes, products):
    data = monthly_year[
        monthly_year["item_id"] == product
    ]

    for year, group in data.groupby("year"):
        ax.plot(
            group["month"], group["sales"],
            marker="o", label=str(year)
        )

    ax.set_title(product)
    ax.set_ylabel("Average daily sales")
    ax.set_xticks(range(1, 13))
    ax.legend(ncol=6, fontsize=8)
    ax.grid(alpha=0.3)

plt.suptitle("Monthly Sales by Year", fontsize=16)
plt.tight_layout()
plt.show()


# analyze holiday effects

holiday_summary = (
    df.groupby(["item_id", "holiday"])["sales"]
    .agg(["mean", "count"])
    .reset_index()
)

holiday_pivot = holiday_summary.pivot(
    index="item_id",
    columns="holiday",
    values="mean"
)

holiday_pivot.columns = [
    "Non-Holiday", "Holiday"
]

holiday_pivot["pct_change"] = (
    (holiday_pivot["Holiday"] /
     holiday_pivot["Non-Holiday"] - 1) * 100
)

print(holiday_pivot.round(2))

holiday_pivot[["Non-Holiday", "Holiday"]].plot(
    kind="bar", figsize=(11, 5)
)

plt.title("Average Daily Sales: Holiday vs Non-Holiday")
plt.ylabel("Average units sold")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


# Use first event name when present, otherwise second
df["event_name"] = df["event_name_1"].fillna(
    df["event_name_2"]
)

event_summary = (
    df[df["holiday"] == 1]
    .groupby(["item_id", "event_name"])["sales"]
    .agg(["mean", "count"])
    .reset_index()
)

# Only show events observed at least 3 times
event_summary = event_summary[
    event_summary["count"] >= 3
]

for product in products:
    print(f"\n{product}")

    product_events = event_summary[
        event_summary["item_id"] == product
    ].sort_values("mean", ascending=False)

    print(product_events.head(10).round(2).to_string(index=False))


# analyze SNAP effects

snap_summary = (
    df.groupby(["item_id", "snap"])["sales"]
    .mean()
    .unstack()
)

snap_summary.columns = ["Non-SNAP", "SNAP"]

snap_summary["pct_change"] = (
    (snap_summary["SNAP"] /
     snap_summary["Non-SNAP"] - 1) * 100
)

print(snap_summary.round(2))

snap_summary[["Non-SNAP", "SNAP"]].plot(
    kind="bar", figsize=(11, 5)
)

plt.title("Average Daily Sales: SNAP vs Non-SNAP Days")
plt.ylabel("Average units sold")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


# consolidated comparison

comparison = summary[[
    "mean_sales",
    "std_sales",
    "cv",
    "zero_pct"
]].copy()

comparison["weekday_range_pct"] = (
    (weekly.max(axis=1) - weekly.min(axis=1))
    / weekly.mean(axis=1) * 100
)

comparison["monthly_range_pct"] = (
    (monthly.max(axis=1) - monthly.min(axis=1))
    / monthly.mean(axis=1) * 100
)

comparison["holiday_effect_pct"] = (
    holiday_pivot["pct_change"]
)

comparison["snap_effect_pct"] = (
    snap_summary["pct_change"]
)

comparison = comparison.round(2)

print(comparison.to_string())

comparison.to_csv(
    "demand_exploratory_summary.csv"
)
