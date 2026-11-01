DMV Experiment 4 - Short Version

Files:
- DMV_Experiment_4_Web_API_Weather_Simple.ipynb: clean notebook
- DMV_Experiment_4_Web_API_Weather_Simple_Executed.ipynb: notebook with outputs
- OpenWeatherMap_Pune_API_Response.json: supplied API response used for the practical
- api_key.txt: placeholder; replace with your own key for a live API request
- Pune_Weather_Cleaned.csv and Pune_Daily_Weather_Summary.csv: generated outputs

To use a live API key:
1. Open api_key.txt.
2. Replace PASTE_YOUR_OWN_API_KEY_HERE with your actual OpenWeatherMap key.
3. Save the file.
4. Run the optional API cell in the notebook.

Keep api_key.txt private. Do not upload or share a ZIP containing your real key.
For offline execution, leave the placeholder unchanged; the notebook automatically uses the supplied JSON response.

Install packages if needed:
pip install pandas matplotlib requests folium

# --- BEGIN EXTRACTED CODE ---

import pandas as pd
import matplotlib.pyplot as plt
import json

# Optional live API request
import requests

try:
    key = open("api_key.txt").read().strip().split("=", 1)[1]
    if key != "PASTE_YOUR_OWN_API_KEY_HERE":
        url = "https://api.openweathermap.org/data/2.5/forecast"
        params = {"q": "Pune,IN", "appid": key, "units": "metric"}
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        weather = response.json()
        print("Fresh data downloaded from OpenWeatherMap.")
    else:
        raise ValueError("No API key entered")
except Exception as e:
    print("Using the supplied API response instead.")
    with open("OpenWeatherMap_Pune_API_Response.json") as f:
        weather = json.load(f)

rows = []
for item in weather["list"]:
    rows.append({
        "datetime": item["dt_txt"],
        "temperature": item["main"]["temp"],
        "humidity": item["main"]["humidity"],
        "wind_speed": item["wind"]["speed"],
        "rainfall": item.get("rain", {}).get("3h", 0)
    })

df = pd.DataFrame(rows)
df.head()

print("Shape:", df.shape)
print(df.info())
print("\\nMissing values before cleaning:")
print(df.isnull().sum())

df["datetime"] = pd.to_datetime(df["datetime"])
for col in ["temperature", "humidity", "wind_speed", "rainfall"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(df[col].median())

df = df.drop_duplicates().sort_values("datetime").reset_index(drop=True)
print("\\nMissing values after cleaning:", df.isnull().sum().sum())

print("Average temperature:", round(df["temperature"].mean(), 2), "°C")
print("Maximum temperature:", round(df["temperature"].max(), 2), "°C")
print("Minimum temperature:", round(df["temperature"].min(), 2), "°C")
print("Average humidity:", round(df["humidity"].mean(), 2), "%")
print("Average wind speed:", round(df["wind_speed"].mean(), 2), "m/s")
print("Total forecast rainfall:", round(df["rainfall"].sum(), 2), "mm")


import matplotlib.dates as mdates

df["datetime"] = pd.to_datetime(df["datetime"])

plt.figure(figsize=(12, 5))

plt.bar(
    df["datetime"],
    df["rainfall"],
    width=0.08
)

plt.title("Rainfall Forecast", fontsize=14)
plt.xlabel("Date and Time")
plt.ylabel("Rainfall (mm per 3 hours)")

# Show labels at 12-hour intervals
plt.gca().xaxis.set_major_locator(
    mdates.HourLocator(interval=12)
)

plt.gca().xaxis.set_major_formatter(
    mdates.DateFormatter("%d %b\n%H:%M")
)

plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.show()

df["date"] = df["datetime"].dt.date
daily = df.groupby("date").agg(
    avg_temperature=("temperature", "mean"),
    max_temperature=("temperature", "max"),
    avg_humidity=("humidity", "mean"),
    total_rainfall=("rainfall", "sum")
).reset_index()

display(daily.round(2))
daily.plot(x="date", y="avg_temperature", kind="bar",
           title="Daily Average Temperature", legend=False, figsize=(8, 4))
plt.ylabel("Temperature (°C)")
plt.tight_layout()
plt.show()

# Coordinates are supplied by the API response.
print("City:", weather["city"]["name"])
print("Latitude:", weather["city"]["coord"]["lat"])
print("Longitude:", weather["city"]["coord"]["lon"])

# Optional map (install folium with: pip install folium)
import folium
m = folium.Map(
    location=[weather["city"]["coord"]["lat"], weather["city"]["coord"]["lon"]],
    zoom_start=10
)
folium.Marker(
    [weather["city"]["coord"]["lat"], weather["city"]["coord"]["lon"]],
    popup="Pune Weather"
).add_to(m)
m

corr = df[["temperature", "humidity", "wind_speed", "rainfall"]].corr()
plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Correlation")
plt.xticks(range(4), corr.columns, rotation=30)
plt.yticks(range(4), corr.columns)
plt.title("Weather Attribute Correlation")
for i in range(4):
    for j in range(4):
        plt.text(j, i, round(corr.iloc[i, j], 2), ha="center", va="center")
plt.tight_layout()
plt.show()

df.to_csv("Pune_Weather_Cleaned.csv", index=False)
daily.to_csv("Pune_Daily_Weather_Summary.csv", index=False)
print("Cleaned and daily summary files saved.")

# --- END EXTRACTED CODE ---
