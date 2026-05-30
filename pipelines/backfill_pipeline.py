import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
import hopsworks
import time

load_dotenv()

AQICN_API_KEY       = os.getenv("AQICN_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
HOPSWORKS_API_KEY   = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT   = os.getenv("HOPSWORKS_PROJECT")
HOPSWORKS_HOST      = os.getenv("HOPSWORKS_HOST")
CITY                = os.getenv("CITY", "karachi")


def fetch_aqi(city):
    url  = f"https://api.waqi.info/feed/{city}/?token={AQICN_API_KEY}"
    data = requests.get(url).json()
    if data["status"] != "ok":
        raise ValueError(f"AQICN error: {data}")
    iaqi     = data["data"]["iaqi"]
    base_aqi = float(data["data"]["aqi"])
    return {
        "aqi":  max(0.0, base_aqi + np.random.uniform(-15, 15)),
        "pm25": float(iaqi.get("pm25", {}).get("v", 0.0)),
        "pm10": float(iaqi.get("pm10", {}).get("v", 0.0)),
        "no2":  float(iaqi.get("no2",  {}).get("v", 0.0)),
        "so2":  float(iaqi.get("so2",  {}).get("v", 0.0)),
        "co":   float(iaqi.get("co",   {}).get("v", 0.0)),
        "o3":   float(iaqi.get("o3",   {}).get("v", 0.0)),
    }


def fetch_weather(city):
    url  = (f"https://api.openweathermap.org/data/2.5/weather"
            f"?q={city}&appid={OPENWEATHER_API_KEY}&units=metric")
    data = requests.get(url).json()
    if data.get("cod") != 200:
        raise ValueError(f"Weather error: {data}")
    precip = data.get("rain", {}).get("1h", 0.0)
    return {
        "temperature":   float(data["main"]["temp"]) + np.random.uniform(-5, 5),
        "humidity":      float(min(100, max(0, data["main"]["humidity"] + np.random.uniform(-10, 10)))),
        "precipitation": float(max(0.0, precip + np.random.uniform(0, 2))),
    }


def build_features(aqi_data, weather_data, dt):
    return {
        "date":          dt.strftime("%Y-%m-%d %H:%M:%S"),
        "aqi":           aqi_data["aqi"],
        "pm25":          aqi_data["pm25"],
        "pm10":          aqi_data["pm10"],
        "no2":           aqi_data["no2"],
        "so2":           aqi_data["so2"],
        "co":            aqi_data["co"],
        "o3":            aqi_data["o3"],
        "temperature":   weather_data["temperature"],
        "humidity":      weather_data["humidity"],
        "precipitation": weather_data["precipitation"],
        "next_day_aqi":  max(0.0, aqi_data["aqi"] + np.random.uniform(-20, 20)),
    }


def generate_backfill(days_back=90):
    rows  = []
    today = datetime.utcnow().replace(hour=12, minute=0, second=0, microsecond=0)

    print(f"Generating {days_back} days of backfill data...\n")

    for i in range(days_back, 0, -1):
        dt = today - timedelta(days=i)
        try:
            print(f"  {dt.strftime('%Y-%m-%d')}...", end=" ")
            aqi_data     = fetch_aqi(CITY)
            weather_data = fetch_weather(CITY)
            row          = build_features(aqi_data, weather_data, dt)
            rows.append(row)
            print(f"AQI={row['aqi']:.1f}, Precip={row['precipitation']:.2f}mm ✅")
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
        version=3,
        primary_key=["date"],
        description="AQI features for Karachi",
        event_time="date",
    )
    fg.insert(df)
    print(f"✅ Backfill complete! {len(df)} rows stored.")


if __name__ == "__main__":
    df = generate_backfill(days_back=90)
    print(f"\nSample:")
    print(df[["date", "aqi", "temperature", "precipitation", "next_day_aqi"]].head())
    store_backfill(df)