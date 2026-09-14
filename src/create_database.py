import pandas as pd
import sqlite3
import os

# File paths
input_file = "data/Delivery_Logistics_cleaned.csv"
database_folder = "database"
database_file = "database/logistics.db"

# Create database folder if it doesn't exist
os.makedirs(database_folder, exist_ok=True)

# Load cleaned dataset
df = pd.read_csv(input_file)

print("Dataset loaded successfully!")
print("Rows:", len(df))
print("Columns:", len(df.columns))

# Connect to SQLite database
connection = sqlite3.connect(database_file)

# Create main deliveries table
df.to_sql(
    "deliveries",
    connection,
    if_exists="replace",
    index=False
)

# Create indexes for commonly analyzed columns
connection.execute(
    "CREATE INDEX IF NOT EXISTS idx_delivery_partner "
    "ON deliveries(delivery_partner)"
)

connection.execute(
    "CREATE INDEX IF NOT EXISTS idx_region "
    "ON deliveries(region)"
)

connection.execute(
    "CREATE INDEX IF NOT EXISTS idx_delayed "
    "ON deliveries(delayed)"
)

connection.commit()

# Check number of records
result = connection.execute(
    "SELECT COUNT(*) FROM deliveries"
).fetchone()

print("Records in database:", result[0])

# Display first 5 records
sample = connection.execute(
    "SELECT * FROM deliveries LIMIT 5"
).fetchall()

print("\nFirst 5 database records:")
for row in sample:
    print(row)

connection.close()

print("\n===================================")
print("DATABASE CREATED SUCCESSFULLY")
print("Database:", database_file)
print("===================================")