from dotenv import load_dotenv
load_dotenv()
import os, requests, pandas as pd, joblib
from datetime import datetime, timedelta

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
LAT, LON = 24.8607, 67.0011
FEATURE_COLS = ["pm25","pm10","no2","so2","co","o3","temperature","humidity"]

model = joblib.load("best_aqi_model.pkl")

# Fetch pollution forecast from OpenWeather
url  = f"http://api.openweathermap.org/data/2.5/air_pollution/forecast?lat={LAT}&lon={LON}&appid={OPENWEATHER_API_KEY}"
poll = requests.get(url).json()

# Fetch weather forecast from Open-Meteo (free, no API key)
today     = datetime.utcnow().date()
end_date  = today + timedelta(days=4)
url3 = (f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={LAT}&longitude={LON}"
        f"&hourly=temperature_2m,relative_humidity_2m"
        f"&start_date={today}&end_date={end_date}"
        f"&timezone=UTC")
meteo = requests.get(url3).json()

# Build weather lookup by date (noon value)
weather_by_date = {}
for i, time_str in enumerate(meteo["hourly"]["time"]):
    dt   = datetime.strptime(time_str, "%Y-%m-%dT%H:%M")
    if dt.hour == 12:
        weather_by_date[dt.date()] = {
            "temperature": float(meteo["hourly"]["temperature_2m"][i]),
            "humidity":    float(meteo["hourly"]["relative_humidity_2m"][i]),
        }

seen_dates = []

print(f"{'Date':<12} {'PM2.5':>6} {'PM10':>6} {'NO2':>5} {'CO':>6} {'O3':>6} {'Temp':>6} {'Hum':>5} {'Model AQI':>10}")
print("-"*70)

for item in poll["list"]:
    dt   = datetime.utcfromtimestamp(item["dt"])
    date = dt.date()
    if date <= today: continue
    if date in seen_dates: continue
    seen_dates.append(date)

    comp    = item["components"]
    weather = weather_by_date.get(date, {"temperature": 35.0, "humidity": 55.0})

    row = {
        "pm25":        float(comp.get("pm2_5", 0.0)),
        "pm10":        float(comp.get("pm10",  0.0)),
        "no2":         float(comp.get("no2",   0.0)),
        "so2":         float(comp.get("so2",   0.0)),
        "co":          float(comp.get("co",    0.0)),
        "o3":          float(comp.get("o3",    0.0)),
        "temperature": weather["temperature"],
        "humidity":    weather["humidity"],
    }

    df_row    = pd.DataFrame([row])[FEATURE_COLS]
    predicted = float(model.predict(df_row)[0])
    predicted = max(1, min(5, round(predicted)))

    print(f"{str(date):<12} {row['pm25']:>6.2f} {row['pm10']:>6.2f} {row['no2']:>5.2f} {row['co']:>6.2f} {row['o3']:>6.2f} {row['temperature']:>6.1f} {row['humidity']:>5.0f} {predicted:>10}")

    if len(seen_dates) == 3:
        break