DMV – EXPERIMENT 5: AQI VISUALIZATION

Files in this pack
1. DMV_Experiment_5_AQI_Visualization.ipynb
   The complete Jupyter Notebook containing the data exploration,
   preprocessing, and visualization tasks.
2. City_Air_Quality.csv
   The fixed historical Pune air-quality dataset used by the notebook.

How to run
1. Extract this ZIP file into one folder.
2. Open the folder in Jupyter Notebook or JupyterLab.
3. Open DMV_Experiment_5_AQI_Visualization.ipynb.
4. Run the cells from top to bottom.

Required libraries
pandas
numpy
matplotlib

If required, install them in a notebook cell using:
%pip install pandas numpy matplotlib

Dataset and preprocessing notes
- The notebook uses the included historical Pune dataset. No separate
  dataset download is required.
- The dataset contains US AQI and six pollutant measurements.
- Missing values in the selected measurement columns are filled using
  time-based interpolation in both directions. Boundary-filled values
  are estimates, not actual new observations.
- The dataset does not contain a Dominant_Pollutant category. Therefore,
  the AQI box plot compares distributions across months rather than
  using a fabricated dominant-pollutant category.
- A linear trendline is included in the PM2.5 versus AQI scatter plot.

The notebook is provided with cleared outputs so it can be run afresh
on another computer using the included CSV file.

# --- BEGIN EXTRACTED CODE ---

# Run this only if the libraries are not installed
# %pip install pandas numpy matplotlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = (12, 5)
plt.rcParams["font.size"] = 10

df = pd.read_csv("City_Air_Quality.csv")

print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset information:")
df.info()

print("\nFirst five rows:")
display(df.head())

print("\nDescriptive statistics:")
display(df.describe().round(2))

print("\nMissing values:")
print(df.isnull().sum())

# Convert date strings into datetime values and sort by date
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# Select AQI and pollutant columns available in this dataset
aqi_col = "us_aqi"
pollutant_cols = [
    "pm2_5", "pm10", "carbon_monoxide",
    "nitrogen_dioxide", "sulphur_dioxide", "ozone"
]

# Fill missing measurements using time-based interpolation.
# limit_direction='both' also fills missing values at the beginning/end.
measurement_cols = pollutant_cols + [aqi_col]
df = df.set_index("date")
df[measurement_cols] = df[measurement_cols].interpolate(
    method="time", limit_direction="both"
)
df = df.reset_index()

# Create month columns for monthly comparisons
df["month"] = df["date"].dt.to_period("M").astype(str)
df["month_name"] = df["date"].dt.strftime("%b %Y")

print("Prepared variables:")
print(["date", aqi_col] + pollutant_cols)
print("\nMissing values after interpolation:")
print(df[measurement_cols].isnull().sum())
assert df[measurement_cols].isnull().sum().sum() == 0, "Some measurement values are still missing."
print("\nAll selected AQI and pollutant measurement columns are complete.")

# Calculate a 7-day moving average to make the trend easier to see
df["aqi_7_day_average"] = df[aqi_col].rolling(window=7, min_periods=1).mean()

plt.figure(figsize=(13, 5))
plt.plot(df["date"], df[aqi_col], label="Daily US AQI", alpha=0.55)
plt.plot(df["date"], df["aqi_7_day_average"], label="7-day moving average", linewidth=2.5)

plt.title("Daily AQI Trend in Pune")
plt.xlabel("Date")
plt.ylabel("US AQI")
plt.legend()
plt.grid(True, alpha=0.3)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

pollutant_labels = {
    "pm2_5": "PM2.5",
    "pm10": "PM10",
    "carbon_monoxide": "Carbon Monoxide",
    "nitrogen_dioxide": "Nitrogen Dioxide",
    "sulphur_dioxide": "Sulphur Dioxide",
    "ozone": "Ozone"
}

fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True)
axes = axes.flatten()

for i, col in enumerate(pollutant_cols):
    axes[i].plot(df["date"], df[col], linewidth=1.4, label=pollutant_labels[col])
    axes[i].set_title(pollutant_labels[col] + " Trend")
    axes[i].set_xlabel("Date")
    axes[i].set_ylabel("Measured level")
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

fig.suptitle("Pollutant Levels in Pune Over Time", fontsize=15)
plt.tight_layout()
plt.show()

monthly = df.groupby("month", sort=True).agg(
    month_name=("month_name", "first"),
    average_aqi=(aqi_col, "mean"),
    average_pm25=("pm2_5", "mean"),
    average_pm10=("pm10", "mean"),
    average_no2=("nitrogen_dioxide", "mean")
).reset_index()

# 6A: Monthly average AQI
plt.figure(figsize=(11, 5))
plt.bar(monthly["month_name"], monthly["average_aqi"])
plt.title("Monthly Average US AQI in Pune")
plt.xlabel("Month")
plt.ylabel("Average US AQI")
plt.xticks(rotation=35)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# 6B: Stacked bar chart of monthly average pollutant measurements
plt.figure(figsize=(11, 5))
plt.bar(monthly["month_name"], monthly["average_pm25"], label="PM2.5")
plt.bar(monthly["month_name"], monthly["average_pm10"],
        bottom=monthly["average_pm25"], label="PM10")
plt.bar(monthly["month_name"], monthly["average_no2"],
        bottom=monthly["average_pm25"] + monthly["average_pm10"],
        label="Nitrogen Dioxide")
plt.title("Monthly Average Pollutant Levels (Stacked)")
plt.xlabel("Month")
plt.ylabel("Measured level")
plt.xticks(rotation=35)
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# 7A: Box plot of AQI values for each month
month_groups = [
    df.loc[df["month"] == month, aqi_col].dropna().values
    for month in sorted(df["month"].unique())
]
month_labels = [
    pd.Period(month).strftime("%b %Y")
    for month in sorted(df["month"].unique())
]

plt.figure(figsize=(13, 5))
plt.boxplot(month_groups, tick_labels=month_labels)
plt.title("Distribution of AQI Values Across Months")
plt.xlabel("Month")
plt.ylabel("US AQI")
plt.xticks(rotation=35)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# 7B: Violin plot of pollutant measurement distributions
violin_data = [df[col].dropna().values for col in pollutant_cols]
plt.figure(figsize=(12, 5))
plt.violinplot(violin_data, showmeans=True, showmedians=True)
plt.title("Distribution of Pollutant Measurements")
plt.xlabel("Pollutant")
plt.ylabel("Measured level")
plt.xticks(range(1, len(pollutant_cols) + 1),
           [pollutant_labels[col] for col in pollutant_cols],
           rotation=20)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# 8A: Scatter plot of PM2.5 against AQI with a linear trendline
plt.figure(figsize=(7, 5))
scatter = plt.scatter(
    df["pm2_5"], df[aqi_col],
    c=df[aqi_col], cmap="plasma", alpha=0.75, label="Daily observations"
)

# Fit and draw a straight trendline
x_values = df["pm2_5"].to_numpy()
y_values = df[aqi_col].to_numpy()
slope, intercept = np.polyfit(x_values, y_values, 1)
trend_x = np.linspace(x_values.min(), x_values.max(), 100)
plt.plot(trend_x, slope * trend_x + intercept,
         color="black", linewidth=2, label="Linear trendline")

plt.colorbar(scatter, label="US AQI")
plt.title("Relationship Between PM2.5 and US AQI")
plt.xlabel("PM2.5")
plt.ylabel("US AQI")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 8B: Bubble chart of PM10 against AQI; bubble size represents CO
plt.figure(figsize=(8, 5))
bubble_sizes = df["carbon_monoxide"].clip(lower=0) / df["carbon_monoxide"].max() * 250 + 10
bubble = plt.scatter(
    df["pm10"], df[aqi_col],
    s=bubble_sizes,
    c=df["pm2_5"], cmap="viridis",
    alpha=0.65, edgecolors="black", linewidth=0.3
)
plt.colorbar(bubble, label="PM2.5 level")
plt.title("PM10 vs US AQI (Bubble Size = Carbon Monoxide)")
plt.xlabel("PM10")
plt.ylabel("US AQI")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- END EXTRACTED CODE ---
