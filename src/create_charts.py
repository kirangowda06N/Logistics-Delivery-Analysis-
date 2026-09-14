import pandas as pd
import matplotlib.pyplot as plt
import os

print("=" * 60)
print("LOGISTICS DELIVERY OUTLIER CHARTS")
print("=" * 60)

# Load original dataset
df = pd.read_csv("data/Delivery_Logistics.csv")

# Convert columns to numeric
columns = [
    "distance_km",
    "package_weight_kg",
    "delivery_time_hours",
    "expected_time_hours",
    "delivery_cost"
]

for col in columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Create output charts folder
os.makedirs("outputs/charts", exist_ok=True)


# Function to create boxplot
def create_boxplot(column, title, filename, ylabel):
    plt.figure(figsize=(8, 5))

    plt.boxplot(df[column].dropna())

    plt.title(title)
    plt.ylabel(ylabel)

    plt.savefig(
        f"outputs/charts/{filename}",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Created: {filename}")


# Create 4 required charts

create_boxplot(
    "distance_km",
    "Distance (km) Boxplot",
    "distance_boxplot.png",
    "Distance (km)"
)

create_boxplot(
    "package_weight_kg",
    "Package Weight (kg) Boxplot",
    "package_weight_boxplot.png",
    "Package Weight (kg)"
)

create_boxplot(
    "delivery_time_hours",
    "Delivery Time (Hours) Boxplot",
    "delivery_time_boxplot.png",
    "Delivery Time (Hours)"
)

create_boxplot(
    "delivery_cost",
    "Delivery Cost Boxplot",
    "delivery_cost_boxplot.png",
    "Delivery Cost"
)


print("=" * 60)
print("ALL 4 CHARTS CREATED SUCCESSFULLY")
print("Location: outputs/charts/")
print("=" * 60)