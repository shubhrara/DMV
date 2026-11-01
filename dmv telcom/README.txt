DMV Experiment 3 – Data Cleaning and Preparation
Problem: Analyzing Customer Churn in a Telecommunications Company

Files included:
- DMV_Experiment_3_Data_Cleaning.ipynb: clean notebook to run
- DMV_Experiment_3_Data_Cleaning_Executed.ipynb: notebook with outputs
- Telecom_Customer_Churn.csv: source dataset (renamed from the Kaggle archive filename)
- Cleaned_Telecom_Customer_Churn.csv: cleaned dataset with engineered features
- X_train.csv, X_test.csv, y_train.csv, y_test.csv: model-ready train/test split

Requirements: pandas, numpy, matplotlib, seaborn, scikit-learn, jupyter.
Install if needed: pip install pandas numpy matplotlib seaborn scikit-learn jupyter

Run the notebook from this folder so it can find Telecom_Customer_Churn.csv.

Note: The supplied ZIP contained WA_Fn-UseC_-Telco-Customer-Churn.csv. It is the standard Telco Customer Churn dataset and has been renamed to Telecom_Customer_Churn.csv for the practical.
Outliers were checked using IQR and domain validation. Plausible high charges were retained rather than clipped automatically. Scaling was fitted only on the training split to avoid data leakage.

# --- BEGIN EXTRACTED CODE ---

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path

DATA_PATH = Path("Telecom_Customer_Churn.csv")


df = pd.read_csv(DATA_PATH)
print("Dataset imported successfully")
print("Shape:", df.shape)
display(df.head())

print("Rows and columns:", df.shape)
print("\nColumn names:\n", df.columns.tolist())
print("\nDataset information:")
df.info()
print("\nNumerical summary:")
display(df.describe())
print("\nMissing values in each column:")
print(df.isna().sum())
print("\nDuplicate rows:", df.duplicated().sum())

# Remove extra spaces and convert blank strings to missing values
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print("Missing values before treatment:")
print(df.isna().sum()[df.isna().sum() > 0])

# A customer with zero tenure has not accumulated monthly charges
zero_tenure = df["tenure"].eq(0)
df.loc[zero_tenure & df["TotalCharges"].isna(), "TotalCharges"] = 0

# Median for remaining numeric gaps; mode for categorical gaps
for col in df.select_dtypes(include=np.number).columns:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())
for col in df.select_dtypes(include="object").columns:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing values after treatment:", int(df.isna().sum().sum()))
print("TotalCharges type:", df["TotalCharges"].dtype)

rows_before = len(df)
duplicate_rows = int(df.duplicated().sum())
df = df.drop_duplicates().copy()
# Customer ID is a unique identifier; keep the first record if an ID is repeated
duplicate_ids = int(df["customerID"].duplicated().sum())
df = df.drop_duplicates(subset="customerID", keep="first").copy()
print("Duplicate rows found:", duplicate_rows)
print("Repeated customer IDs found:", duplicate_ids)
print("Rows removed:", rows_before - len(df))
print("Shape after duplicate removal:", df.shape)

# Normalize text formatting
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()

# Standardize capitalization for categorical labels
df["gender"] = df["gender"].str.title()
for col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]:
    df[col] = df[col].str.title()

print("Unique values after standardization:")
for col in df.select_dtypes(include="object").columns:
    if col != "customerID":
        print(col, ":", sorted(df[col].dropna().unique().tolist()))

df["SeniorCitizen"] = df["SeniorCitizen"].astype("int64")
df["tenure"] = df["tenure"].astype("int64")
df["MonthlyCharges"] = df["MonthlyCharges"].astype("float64")
df["TotalCharges"] = df["TotalCharges"].astype("float64")
print(df.dtypes)

numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, numeric_cols):
    sns.boxplot(y=df[col], ax=ax)
    ax.set_title(col)
plt.tight_layout()
plt.show()

for col in numeric_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    count = int(((df[col] < lower) | (df[col] > upper)).sum())
    print(f"{col}: {count} IQR-flagged values (bounds: {lower:.2f} to {upper:.2f})")

# Domain checks: tenure 0-72 months; charges cannot be negative
invalid = (df["tenure"].lt(0) | df["tenure"].gt(72) |
           df["MonthlyCharges"].lt(0) | df["TotalCharges"].lt(0))
print("Impossible domain values:", int(invalid.sum()))
# Keep legitimate high values; remove only impossible records if any
df = df.loc[~invalid].copy()
print("Rows after domain validation:", len(df))

df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 24, 48, 72],
    labels=["New (0-12 months)", "1-2 years", "2-4 years", "4-6 years"]
)
service_cols = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
df["TotalServicesCount"] = df[service_cols].eq("Yes").sum(axis=1)
df["AvgMonthlySpend"] = np.where(
    df["tenure"] > 0, df["TotalCharges"] / df["tenure"], 0
)
df["Churn_Binary"] = df["Churn"].map({"No": 0, "Yes": 1}).astype(int)
display(df[["tenure", "TenureGroup", "TotalServicesCount", "AvgMonthlySpend", "Churn", "Churn_Binary"]].head())

X = df.drop(columns=["customerID", "Churn", "Churn_Binary"])
y = df["Churn_Binary"]
X = pd.get_dummies(X, drop_first=True, dtype=int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scale_cols = ["tenure", "MonthlyCharges", "TotalCharges", "TotalServicesCount", "AvgMonthlySpend"]
scaler = StandardScaler()
X_train = X_train.copy()
X_test = X_test.copy()
X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
X_test[scale_cols] = scaler.transform(X_test[scale_cols])

print("X_train:", X_train.shape, " y_train:", y_train.shape)
print("X_test :", X_test.shape, " y_test :", y_test.shape)
print("Training churn rate:", round(y_train.mean(), 3))
print("Testing churn rate :", round(y_test.mean(), 3))
print("Training scaled means (approximately 0):")
print(X_train[scale_cols].mean().round(2))

# Export human-readable cleaned data (keeps customerID and original Churn label)
df.to_csv("Cleaned_Telecom_Customer_Churn.csv", index=False)

# Export model-ready train/test features and labels
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_frame("Churn_Binary").to_csv("y_train.csv", index=False)
y_test.to_frame("Churn_Binary").to_csv("y_test.csv", index=False)

print("Exported: Cleaned_Telecom_Customer_Churn.csv, X_train.csv, X_test.csv, y_train.csv, y_test.csv")

print("Final cleaned dataset shape:", df.shape)
print("Missing values:", int(df.isna().sum().sum()))
print("Duplicate rows:", int(df.duplicated().sum()))
print("Duplicate customer IDs:", int(df["customerID"].duplicated().sum()))
print("Churn distribution:")
print(df["Churn"].value_counts())
display(df.head())

print("\nConclusion:")
print("The telecom dataset was explored, cleaned, standardized and converted to suitable data types.")
print("Valid extreme values were retained after domain checks. New customer features were created.")
print("Categorical variables were encoded, numerical features were scaled using training data only,")
print("and the data was split into training and testing sets for future churn modeling.")

# --- END EXTRACTED CODE ---
