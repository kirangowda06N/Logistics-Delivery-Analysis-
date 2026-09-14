import pandas as pd
import matplotlib.pyplot as plt
import os

# -----------------------------------------
# 1. Load cleaned logistics dataset
# -----------------------------------------

df = pd.read_csv("data/Delivery_Logistics_cleaned.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)

# -----------------------------------------
# 2. Convert numerical columns
# -----------------------------------------

numeric_columns = [
    "distance_km",
    "package_weight_kg",
    "delivery_time_hours",
    "expected_time_hours",
    "delivery_rating",
    "delivery_cost"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Create delay indicator
df["delay_flag"] = (
    df["delayed"].astype(str).str.lower() == "yes"
).astype(int)

# -----------------------------------------
# 3. Descriptive Statistics
# -----------------------------------------

print("\nDESCRIPTIVE STATISTICS")
print(df[numeric_columns].describe())

# -----------------------------------------
# 4. Correlation Analysis
# -----------------------------------------

print("\nCORRELATION MATRIX")

correlation_columns = [
    "distance_km",
    "package_weight_kg",
    "delivery_time_hours",
    "expected_time_hours",
    "delivery_cost",
    "delivery_rating",
    "delay_flag"
]

correlation_matrix = df[correlation_columns].corr()

print(correlation_matrix)

# -----------------------------------------
# Create output folder
# -----------------------------------------

output_folder = "outputs/charts"
os.makedirs(output_folder, exist_ok=True)

# -----------------------------------------
# 5. Delivery Time Distribution
# -----------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    df["delivery_time_hours"].dropna(),
    bins=20,
    edgecolor="black"
)

plt.title("Distribution of Delivery Time")
plt.xlabel("Delivery Time (Hours)")
plt.ylabel("Number of Deliveries")

plt.tight_layout()

plt.savefig(
    f"{output_folder}/delivery_time_distribution.png",
    dpi=200
)

plt.show()

# -----------------------------------------
# 6. Delivery Cost Distribution
# -----------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    df["delivery_cost"].dropna(),
    bins=30,
    edgecolor="black"
)

plt.title("Distribution of Delivery Cost")
plt.xlabel("Delivery Cost")
plt.ylabel("Number of Deliveries")

plt.tight_layout()

plt.savefig(
    f"{output_folder}/delivery_cost_distribution.png",
    dpi=200
)

plt.show()

# -----------------------------------------
# 7. Distance vs Delivery Cost
# -----------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["distance_km"],
    df["delivery_cost"],
    alpha=0.2,
    s=10
)

plt.title("Distance vs Delivery Cost")
plt.xlabel("Distance (km)")
plt.ylabel("Delivery Cost")

plt.tight_layout()

plt.savefig(
    f"{output_folder}/distance_vs_cost.png",
    dpi=200
)

plt.show()

# -----------------------------------------
# 8. Distance vs Delivery Time
# -----------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["distance_km"],
    df["delivery_time_hours"],
    alpha=0.2,
    s=10
)

plt.title("Distance vs Delivery Time")
plt.xlabel("Distance (km)")
plt.ylabel("Delivery Time (Hours)")

plt.tight_layout()

plt.savefig(
    f"{output_folder}/distance_vs_delivery_time.png",
    dpi=200
)

plt.show()

# -----------------------------------------
# 9. Package Weight vs Delivery Cost
# -----------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["package_weight_kg"],
    df["delivery_cost"],
    alpha=0.2,
    s=10
)

plt.title("Package Weight vs Delivery Cost")
plt.xlabel("Package Weight (kg)")
plt.ylabel("Delivery Cost")

plt.tight_layout()

plt.savefig(
    f"{output_folder}/weight_vs_cost.png",
    dpi=200
)

plt.show()

# -----------------------------------------
# 10. Correlation Heatmap
# -----------------------------------------

plt.figure(figsize=(9, 7))

plt.imshow(
    correlation_matrix.values,
    aspect="auto"
)

plt.xticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns
)

for i in range(len(correlation_matrix.columns)):
    for j in range(len(correlation_matrix.columns)):
        plt.text(
            j,
            i,
            f"{correlation_matrix.iloc[i, j]:.2f}",
            ha="center",
            va="center"
        )

plt.title("Correlation Heatmap of Logistics Metrics")

plt.tight_layout()

plt.savefig(
    f"{output_folder}/correlation_heatmap.png",
    dpi=200
)

plt.show()

# -----------------------------------------
# 11. Delay Analysis
# -----------------------------------------

total_deliveries = len(df)

delayed_deliveries = df["delay_flag"].sum()

delay_rate = (
    delayed_deliveries / total_deliveries
) * 100

print("\nDELAY ANALYSIS")

print("Total deliveries:", total_deliveries)
print("Delayed deliveries:", delayed_deliveries)
print("Delay rate:", round(delay_rate, 2), "%")

# -----------------------------------------
# 12. Delivery Mode Analysis
# -----------------------------------------

mode_analysis = df.groupby("delivery_mode").agg(
    deliveries=("delivery_id", "count"),
    average_delivery_time=("delivery_time_hours", "mean"),
    average_cost=("delivery_cost", "mean"),
    delay_rate=("delay_flag", "mean")
)

mode_analysis["delay_rate"] *= 100

print("\nDELIVERY MODE ANALYSIS")
print(mode_analysis)

# -----------------------------------------
# 13. Region Analysis
# -----------------------------------------

region_analysis = df.groupby("region").agg(
    deliveries=("delivery_id", "count"),
    average_delivery_time=("delivery_time_hours", "mean"),
    average_cost=("delivery_cost", "mean"),
    delay_rate=("delay_flag", "mean")
)

region_analysis["delay_rate"] *= 100

print("\nREGION ANALYSIS")
print(region_analysis)

# -----------------------------------------
# 14. Save analytical results
# -----------------------------------------

results_folder = "outputs/results"
os.makedirs(results_folder, exist_ok=True)

mode_analysis.to_csv(
    f"{results_folder}/delivery_mode_analysis.csv"
)

region_analysis.to_csv(
    f"{results_folder}/region_analysis.csv"
)

correlation_matrix.to_csv(
    f"{results_folder}/correlation_matrix.csv"
)

print("\nWeek 3 analysis completed successfully!")
print("Charts saved in:", output_folder)
print("Results saved in:", results_folder)