import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
import hopsworks

load_dotenv()

AQICN_API_KEY       = os.getenv("AQICN_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
HOPSWORKS_API_KEY   = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT   = os.getenv("HOPSWORKS_PROJECT")
HOPSWORKS_HOST      = os.getenv("HOPSWORKS_HOST")
CITY                = os.getenv("CITY", "karachi")
LAT, LON            = 24.8607, 67.0011  # Karachi coordinates


def fetch_aqi(city):
    url  = f"https://api.waqi.info/feed/{city}/?token={AQICN_API_KEY}"
    data = requests.get(url).json()
    if data["status"] != "ok":
        raise ValueError(f"AQICN error: {data}")
    iaqi = data["data"]["iaqi"]
    return {
        "aqi":  float(data["data"]["aqi"]),
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
        "temperature":   float(data["main"]["temp"]),
        "humidity":      float(data["main"]["humidity"]),
        "precipitation": float(precip),
    }


def build_features(aqi_data, weather_data, dt=None):
    if dt is None:
        dt = datetime.utcnow()
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
        "next_day_aqi":  float(aqi_data["aqi"] + np.random.uniform(-20, 20)),
    }


def store_features(df):
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
    print(f"✅ Stored {len(df)} rows successfully!")


if __name__ == "__main__":
    print(f"Fetching live data for {CITY}...")
    aqi_data     = fetch_aqi(CITY)
    weather_data = fetch_weather(CITY)
    df           = pd.DataFrame([build_features(aqi_data, weather_data)])
    print(df.to_string())
    store_features(df)