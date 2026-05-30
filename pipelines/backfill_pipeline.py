import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
import hopsworks
import time

load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
HOPSWORKS_API_KEY   = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT   = os.getenv("HOPSWORKS_PROJECT")
HOPSWORKS_HOST      = os.getenv("HOPSWORKS_HOST")

LAT, LON = 24.8607, 67.0011


def fetch_historical_pollution(dt: datetime):
    start = int(dt.timestamp())
    end   = start + 3600
    url   = (f"http://api.openweathermap.org/data/2.5/air_pollution/history"
             f"?lat={LAT}&lon={LON}&start={start}&end={end}"
             f"&appid={OPENWEATHER_API_KEY}")
    data  = requests.get(url).json()
    if "list" not in data or len(data["list"]) == 0:
        raise ValueError(f"No data for {dt}")
    comp = data["list"][0]["components"]
    aqi  = int(data["list"][0]["main"]["aqi"])
    return {
        "aqi":  aqi,
        "pm25": float(comp.get("pm2_5", 0.0)),
        "pm10": float(comp.get("pm10",  0.0)),
        "no2":  float(comp.get("no2",   0.0)),
        "so2":  float(comp.get("so2",   0.0)),
        "co":   float(comp.get("co",    0.0)),
        "o3":   float(comp.get("o3",    0.0)),
    }


def fetch_historical_weather(dt: datetime):
    """Fetch real historical weather from Open-Meteo (completely free, no API key)."""
    date_str = dt.strftime("%Y-%m-%d")
    url = (f"https://archive-api.open-meteo.com/v1/archive"
           f"?latitude={LAT}&longitude={LON}"
           f"&start_date={date_str}&end_date={date_str}"
           f"&hourly=temperature_2m,relative_humidity_2m"
           f"&timezone=UTC")
    data = requests.get(url).json()

    if "hourly" not in data:
        raise ValueError(f"No weather data for {dt}")

    # Get noon value (hour 12)
    temps     = data["hourly"]["temperature_2m"]
    humidity  = data["hourly"]["relative_humidity_2m"]

    return {
        "temperature": float(temps[12]),
        "humidity":    float(humidity[12]),
    }


def build_features(poll, weather, dt):
    return {
        "date":        dt.strftime("%Y-%m-%d %H:%M:%S"),
        "aqi":         poll["aqi"],
        "pm25":        poll["pm25"],
        "pm10":        poll["pm10"],
        "no2":         poll["no2"],
        "so2":         poll["so2"],
        "co":          poll["co"],
        "o3":          poll["o3"],
        "temperature": weather["temperature"],
        "humidity":    weather["humidity"],
    }


def generate_backfill(days_back=120):
    rows  = []
    today = datetime.utcnow().replace(hour=12, minute=0, second=0, microsecond=0)
    print(f"Fetching {days_back} days of real historical data...\n")

    for i in range(days_back, 0, -1):
        dt = today - timedelta(days=i)
        try:
            print(f"  {dt.strftime('%Y-%m-%d')}...", end=" ")
            poll    = fetch_historical_pollution(dt)
            weather = fetch_historical_weather(dt)
            row     = build_features(poll, weather, dt)
            rows.append(row)
            print(f"AQI={row['aqi']}, PM2.5={row['pm25']:.1f} ✅")
            time.sleep(0.3)
        except Exception as e:
            print(f"❌ {e}")
            continue

    return pd.DataFrame(rows)


def store_backfill(df):
    print("\nConnecting to Hopsworks...")
    project = hopsworks.login(
        host=HOPSWORKS_HOST,
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT
    )
    fs = project.get_feature_store()
    df["date"] = pd.to_datetime(df["date"])

    fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=7,
        primary_key=["date"],
        description="Real AQI features for Karachi - 120 days",
        event_time="date",
    )
    fg.insert(df)
    print(f"✅ Backfill complete! {len(df)} rows stored.")


if __name__ == "__main__":
    df = generate_backfill(days_back=120)
    print(f"\nSample:")
    print(df[["date", "aqi", "pm25", "pm10", "temperature"]].head())
    store_backfill(df)