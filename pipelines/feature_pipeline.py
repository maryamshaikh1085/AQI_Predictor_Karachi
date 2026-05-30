import os
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import hopsworks

load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
HOPSWORKS_API_KEY   = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT   = os.getenv("HOPSWORKS_PROJECT")
HOPSWORKS_HOST      = os.getenv("HOPSWORKS_HOST")

LAT, LON = 24.8607, 67.0011


def fetch_live_pollution():
    url  = (f"http://api.openweathermap.org/data/2.5/air_pollution"
            f"?lat={LAT}&lon={LON}&appid={OPENWEATHER_API_KEY}")
    data = requests.get(url).json()
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


def fetch_live_weather():
    url  = (f"https://api.openweathermap.org/data/2.5/weather"
            f"?lat={LAT}&lon={LON}&appid={OPENWEATHER_API_KEY}&units=metric")
    data = requests.get(url).json()
    return {
        "temperature": float(data["main"]["temp"]),
        "humidity":    float(data["main"]["humidity"]),
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
        version=7,
        primary_key=["date"],
        description="Real AQI features for Karachi - 120 days",
        event_time="date",
    )
    fg.insert(df)
    print(f"✅ Stored successfully!")


if __name__ == "__main__":
    print("Fetching live data for Karachi...")
    poll    = fetch_live_pollution()
    weather = fetch_live_weather()
    now     = datetime.utcnow()
    row     = {
        "date":        now.strftime("%Y-%m-%d %H:%M:%S"),
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
    df = pd.DataFrame([row])
    print(df.to_string())
    store_features(df)