DMV - Experiment 6: Data Aggregation

Files:
1. DMV_Experiment_6_Data_Aggregation.ipynb
2. Global Superstore Dataset.csv

How to run:
- Keep both files in the same folder.
- Open the notebook in Jupyter Notebook or Google Colab.
- Run the cells from top to bottom.

Libraries:
pandas
matplotlib

Dataset:
External Global Superstore retail transactions dataset (9,800 rows, 17 columns).

Important:
The dataset contains Region, Category and Sales, but it does not contain a quantity-sold column. The notebook completes the required regional and product-category sales aggregation tasks using the available fields.

# --- BEGIN EXTRACTED CODE ---

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Global Superstore Dataset.csv")
print("Dataset imported successfully")

print("Shape:", df.shape)
print("\\nColumns:")
print(df.columns)
print("\\nFirst five rows:")
print(df.head())
print("\\nDataset information:")
df.info()

print("Relevant columns:")
print(df[["Region", "Category", "Sales"]].head())

print("\\nMissing values:")
print(df[["Region", "Category", "Sales"]].isnull().sum())

region_sales = df.groupby("Region")["Sales"].sum().sort_values(ascending=False)
print(region_sales)

region_sales.plot(kind="bar", title="Total Sales by Region", ylabel="Sales", xlabel="Region")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

region_sales.plot(kind="pie", autopct="%1.1f%%", title="Sales Distribution by Region")
plt.ylabel("")
plt.tight_layout()
plt.show()

top_region = region_sales.idxmax()
top_sales = region_sales.max()

print("Top-performing region:", top_region)
print("Total sales:", round(top_sales, 2))

region_category_sales = df.groupby(["Region", "Category"])["Sales"].sum().unstack()
print(region_category_sales)

region_category_sales.plot(kind="bar", title="Sales by Region and Product Category")
plt.xlabel("Region")
plt.ylabel("Sales")
plt.xticks(rotation=0)
plt.legend(title="Product Category")
plt.tight_layout()
plt.show()

region_category_sales.plot(kind="bar", stacked=True, title="Stacked Sales by Region and Product Category")
plt.xlabel("Region")
plt.ylabel("Sales")
plt.xticks(rotation=0)
plt.legend(title="Product Category")
plt.tight_layout()
plt.show()

# --- END EXTRACTED CODE ---
