import pandas as pd

# load dataset
df = pd.read_csv("data/Delivery_Logistics.csv")


#Disply first 5 rows
print("First 5 Rows:")
print(df.head())

#Disply dataset size
print("\nDataSet Shape:")
print(df.shape)

#Disply column names
print("\nColumn Names:")
print(df.columns.to_list())

#Disply Data Types
print("\ndata Types:")
print(df.dtypes)

#Disply missing values
print("\nMissing Values:")
print(df.isnull().sum())

#Disply Duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())