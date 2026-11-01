DMV Experiment 3 - Data Wrangling

Files:
1. DMV_Experiment_3_Data_Wrangling.ipynb
2. train.csv

Dataset:
This is the exact train.csv file supplied by the student for this practical.
It contains 80,000 rows and 18 columns.

How to run:
1. Extract this ZIP into one folder.
2. Open the notebook in Jupyter Notebook or JupyterLab.
3. Keep train.csv in the same folder as the notebook.
4. Run cells from top to bottom.

Install if required:
pip install pandas numpy

Task 3 note:
No separate verified neighbourhood dataset was supplied, so no artificial merge dataset
has been added. The notebook explains when and how a merge could be performed.

# --- BEGIN EXTRACTED CODE ---

import pandas as pd
import numpy as np

df = pd.read_csv("train.csv")

print(df.head())
print("Dataset shape:", df.shape)

# Convert column names to lowercase and replace spaces with underscores.
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

print("Cleaned column names:")
print(df.columns.tolist())

print("Missing values in each column:")
print(df.isnull().sum())

# Fill missing numerical values with the median.
num_cols = df.select_dtypes(include="number").columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill missing text values with the most common value.
cat_cols = df.select_dtypes(include="object").columns

for col in cat_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

print("Total missing values after cleaning:", df.isnull().sum().sum())

print("Available columns for a possible merge:")
print(df.columns.tolist())

# No merge is performed because no second verified dataset is supplied.

# Select properties in Delhi that are Apartments or Villas.
df_filtered = df[
    (df["city"] == "Delhi") &
    (df["property_type"].isin(["Apartment", "Villa"]))
].copy()

print(df_filtered.head())
print("Number of selected properties:", len(df_filtered))

# Convert categorical columns into numerical columns using one-hot encoding.
df_encoded = pd.get_dummies(
    df_filtered,
    columns=["city", "locality_type", "property_type", "furnishing_status"],
    dtype=int
)

print(df_encoded.head())
print("Encoded dataset shape:", df_encoded.shape)

# Calculate average, minimum and maximum prices by property type.
summary = df_filtered.groupby("property_type").agg(
    Average_Price=("price_inr_lakhs", "mean"),
    Minimum_Price=("price_inr_lakhs", "min"),
    Maximum_Price=("price_inr_lakhs", "max"),
    Number_of_Properties=("property_id", "count")
).round(2)

print(summary)

Q1 = df_filtered["price_inr_lakhs"].quantile(0.25)
Q3 = df_filtered["price_inr_lakhs"].quantile(0.75)
IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

outliers = df_filtered[
    (df_filtered["price_inr_lakhs"] < lower_limit) |
    (df_filtered["price_inr_lakhs"] > upper_limit)
]

print("Number of price outliers:", len(outliers))

# Cap extreme prices at the IQR limits.
df_filtered["price_inr_lakhs"] = df_filtered["price_inr_lakhs"].clip(
    lower=lower_limit,
    upper=upper_limit
)

print("Outlier handling completed.")

df_filtered.to_csv("Wrangled_RealEstate_Prices.csv", index=False)
df_encoded.to_csv("Encoded_RealEstate_Prices.csv", index=False)

print("Final datasets saved successfully.")

# --- END EXTRACTED CODE ---
