import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

df = pd.read_csv("data/Delivery_Logistics.csv")

print("=" * 60)
print("LOGISTICS DELIVERY DATA PREPROCESSING")
print("=" * 60)

print("\nOriginal Dataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())


# --------------------------------------------------
# 2. DATA QUALITY CHECK
# --------------------------------------------------

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nData Types:")
print(df.dtypes)


# --------------------------------------------------
# 3. CHECK DUPLICATE DELIVERY IDs
# --------------------------------------------------

print("\nUnique Delivery IDs:")
print(df["delivery_id"].nunique())

print("\nTotal Records:")
print(len(df))

duplicate_ids = df["delivery_id"].duplicated().sum()

print("\nDuplicate Delivery IDs:")
print(duplicate_ids)


# --------------------------------------------------
# 4. FIX DELIVERY TIME COLUMNS
# --------------------------------------------------

# Convert timestamp-like values into numeric hours
df["delivery_time_hours"] = pd.to_datetime(
    df["delivery_time_hours"],
    errors="coerce"
).dt.nanosecond

df["expected_time_hours"] = pd.to_datetime(
    df["expected_time_hours"],
    errors="coerce"
).dt.nanosecond


# --------------------------------------------------
# 5. HANDLE MISSING VALUES CREATED DURING CONVERSION
# --------------------------------------------------

numeric_columns = [
    "distance_km",
    "package_weight_kg",
    "delivery_time_hours",
    "expected_time_hours",
    "delivery_cost"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")
    df[column] = df[column].fillna(df[column].median())


# --------------------------------------------------
# 6. CREATE UNIQUE DELIVERY ID
# --------------------------------------------------

df["delivery_id_original"] = df["delivery_id"]

df["delivery_id"] = range(1, len(df) + 1)


# --------------------------------------------------
# 7. OUTLIER DETECTION USING IQR
# --------------------------------------------------

print("\n" + "=" * 60)
print("OUTLIER ANALYSIS")
print("=" * 60)

outlier_summary = {}

for column in numeric_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_limit) |
        (df[column] > upper_limit)
    ]

    outlier_summary[column] = len(outliers)

    print(f"\n{column}")
    print(f"Q1: {Q1:.2f}")
    print(f"Q3: {Q3:.2f}")
    print(f"IQR: {IQR:.2f}")
    print(f"Lower Limit: {lower_limit:.2f}")
    print(f"Upper Limit: {upper_limit:.2f}")
    print(f"Outliers: {len(outliers)}")


# --------------------------------------------------
# 8. NORMALIZATION
# --------------------------------------------------

scaler = MinMaxScaler()

normalization_columns = [
    "distance_km",
    "package_weight_kg",
    "delivery_cost"
]

df[normalization_columns] = scaler.fit_transform(
    df[normalization_columns]
)

print("\n" + "=" * 60)
print("NORMALIZATION COMPLETED")
print("=" * 60)

print(df[normalization_columns].head())


# --------------------------------------------------
# 9. FINAL DATA QUALITY CHECK
# --------------------------------------------------

print("\nFinal Dataset Shape:")
print(df.shape)

print("\nFinal Missing Values:")
print(df.isnull().sum().sum())

print("\nFinal Duplicate Rows:")
print(df.duplicated().sum())


# --------------------------------------------------
# 10. SAVE CLEANED DATASET
# --------------------------------------------------

df.to_csv(
    "data/Delivery_Logistics_cleaned.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")

print("\nPreprocessing completed successfully.")