DMV EXPERIMENT 3 - TIME SERIES DATA ANALYSIS

Files:
1. DMV_Experiment_3_Time_Series.ipynb
   Clean notebook to run yourself.
2. DMV_Experiment_3_Time_Series_Executed.ipynb
   Same notebook with outputs already saved.
3. Stock_Prices.csv
   Apple (AAPL) historical stock data extracted from the Kaggle dataset you uploaded.

Dataset details:
- Company: Apple Inc. (AAPL)
- Rows: 1,259
- Columns: date, open, high, low, close, volume, Name
- Date range: 2013-02-08 to 2018-02-07
- Missing values: none in the selected file

How to run:
1. Extract this ZIP folder.
2. Open the notebook in Jupyter Notebook or JupyterLab.
3. Make sure Stock_Prices.csv is in the same folder as the notebook.
4. Run the cells from top to bottom.

Required Python packages:
pandas, numpy, matplotlib, statsmodels, scikit-learn, jupyter

If a package is missing, run:
pip install pandas numpy matplotlib statsmodels scikit-learn jupyter

Note:
The source Kaggle archive contained a separate AAPL_data.csv file. It has been renamed to Stock_Prices.csv to match the practical problem statement. The dataset is historical (2013-2018), so the forecast is for demonstration/learning and is not current investment guidance.

# --- BEGIN EXTRACTED CODE ---

# Import the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the dataset
df = pd.read_csv("Stock_Prices.csv")

# Display the first five rows
df.head()

# Understand the structure and contents of the dataset
print("Shape of dataset:", df.shape)
print("\\nColumn names:")
print(df.columns.tolist())

print("\\nDataset information:")
df.info()

print("\\nStatistical summary:")
display(df.describe())

print("\\nMissing values:")
print(df.isnull().sum())

# Convert date into datetime format
df["date"] = pd.to_datetime(df["date"])

# Sort data by date and use date as the index
df = df.sort_values("date")
df = df.set_index("date")

print("Date column data type:", df.index.dtype)
print("Date range:", df.index.min(), "to", df.index.max())
df.head()

# Plot daily closing prices
plt.figure(figsize=(12, 5))
plt.plot(df.index, df["close"], label="Closing Price")
plt.title("Apple Stock Closing Price Over Time")
plt.xlabel("Date")
plt.ylabel("Closing Price (USD)")
plt.legend()
plt.grid(True)
plt.show()

# Calculate 7-day and 30-day moving averages
df["MA_7"] = df["close"].rolling(window=7).mean()
df["MA_30"] = df["close"].rolling(window=30).mean()

# Plot closing price with moving averages
plt.figure(figsize=(12, 5))
plt.plot(df.index, df["close"], label="Closing Price", alpha=0.5)
plt.plot(df.index, df["MA_7"], label="7-day Moving Average")
plt.plot(df.index, df["MA_30"], label="30-day Moving Average")
plt.title("Closing Price with Moving Averages")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.show()

# Create calendar-related columns
season = df.copy()
season["Weekday"] = season.index.day_name()
season["Month"] = season.index.month
season["Month_Name"] = season.index.strftime("%b")
season["Year"] = season.index.year

# Average closing price by weekday
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
weekday_avg = season.groupby("Weekday")["close"].mean().reindex(weekday_order)

plt.figure(figsize=(9, 4))
weekday_avg.plot(kind="bar")
plt.title("Average Closing Price by Day of the Week")
plt.xlabel("Day")
plt.ylabel("Average Closing Price (USD)")
plt.xticks(rotation=30)
plt.grid(axis="y")
plt.show()

# Average closing price by month
month_avg = season.groupby("Month")["close"].mean()

plt.figure(figsize=(9, 4))
month_avg.plot(kind="bar")
plt.title("Average Closing Price by Month")
plt.xlabel("Month")
plt.ylabel("Average Closing Price (USD)")
plt.xticks(range(12), ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], rotation=0)
plt.grid(axis="y")
plt.show()

# Average closing price by year
year_avg = season.groupby("Year")["close"].mean()

plt.figure(figsize=(9, 4))
year_avg.plot(kind="bar")
plt.title("Average Closing Price by Year")
plt.xlabel("Year")
plt.ylabel("Average Closing Price (USD)")
plt.grid(axis="y")
plt.show()

# Calculate correlation between closing price and trading volume
correlation = df[["close", "volume"]].corr()
print("Correlation matrix:")
display(correlation)

# Plot the relationship
plt.figure(figsize=(8, 5))
plt.scatter(df["volume"], df["close"], alpha=0.4)
plt.title("Closing Price vs Trading Volume")
plt.xlabel("Trading Volume")
plt.ylabel("Closing Price (USD)")
plt.grid(True)
plt.show()

# Install/import ARIMA from statsmodels
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Use the closing price series
close_prices = df["close"].astype(float)

# Split data in chronological order (80% train, 20% test)
split_point = int(len(close_prices) * 0.8)
train = close_prices.iloc[:split_point]
test = close_prices.iloc[split_point:]

# Fit ARIMA model
model = ARIMA(train, order=(5, 1, 0))
model_fit = model.fit()

# Forecast the test period
test_forecast = model_fit.forecast(steps=len(test))
test_forecast.index = test.index

# Evaluate forecast accuracy
mae = mean_absolute_error(test, test_forecast)
rmse = np.sqrt(mean_squared_error(test, test_forecast))

print("Mean Absolute Error (MAE):", round(mae, 3))
print("Root Mean Squared Error (RMSE):", round(rmse, 3))

# Plot actual and predicted prices
plt.figure(figsize=(12, 5))
plt.plot(train.index, train, label="Training Data")
plt.plot(test.index, test, label="Actual Test Prices")
plt.plot(test_forecast.index, test_forecast, label="ARIMA Forecast")
plt.title("ARIMA Forecast vs Actual Closing Prices")
plt.xlabel("Date")
plt.ylabel("Closing Price (USD)")
plt.legend()
plt.grid(True)
plt.show()

# Fit ARIMA on the complete series and forecast the next 30 observations
final_model = ARIMA(close_prices, order=(5, 1, 0))
final_model_fit = final_model.fit()

future_values = final_model_fit.forecast(steps=30)

# Create future business-day dates (approximate trading dates)
future_dates = pd.bdate_range(
    start=close_prices.index[-1] + pd.offsets.BDay(1),
    periods=30
)
future_forecast = pd.Series(future_values.values, index=future_dates)

# Display forecast values
display(future_forecast.to_frame(name="Forecast Closing Price"))

# Plot historical prices and future forecast
plt.figure(figsize=(12, 5))
plt.plot(close_prices.index, close_prices, label="Historical Closing Price")
plt.plot(future_forecast.index, future_forecast, label="30-day Forecast", color="red")
plt.title("Apple Stock Price: 30 Trading-Day ARIMA Forecast")
plt.xlabel("Date")
plt.ylabel("Closing Price (USD)")
plt.legend()
plt.grid(True)
plt.show()

# --- END EXTRACTED CODE ---
