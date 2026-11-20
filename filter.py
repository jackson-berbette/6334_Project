import pandas as pd

# Read original CSV
df_evaluation = pd.read_csv("Data/sales_train_evaluation.csv")
df_validation = pd.read_csv("Data/sales_train_validation.csv")
df_prices = pd.read_csv("Data/sell_prices.csv")

# IDs to keep
ids_to_keep = [
    "FOODS_3_668",
    "FOODS_3_744",
    "FOODS_3_327",
    "FOODS_3_156",
    "HOUSEHOLD_1_532"
]

# Filter rows
filtered_evaluation = df_evaluation[(df_evaluation["item_id"].isin(ids_to_keep)) & (df_evaluation["store_id"] == "CA_1")]
filtered_validation = df_validation[(df_validation["item_id"].isin(ids_to_keep)) & (df_validation["store_id"] == "CA_1")]
filtered_prices = df_prices[(df_prices["item_id"].isin(ids_to_keep)) & (df_prices["store_id"] == "CA_1")]

# Save to new CSV
filtered_evaluation.to_csv("Data/filtered_sales_evaluation.csv", index=False)
filtered_validation.to_csv("Data/filtered_sales_validation.csv", index=False)
filtered_prices.to_csv("Data/filtered_prices.csv", index=False)


print(f"Original rows: {len(df_evaluation)}")
print(f"Filtered rows: {len(filtered_evaluation)}")

print(f"Original rows: {len(df_validation)}")
print(f"Filtered rows: {len(filtered_validation)}")

print(f"Original rows: {len(df_prices)}")
print(f"Filtered rows: {len(filtered_prices)}")