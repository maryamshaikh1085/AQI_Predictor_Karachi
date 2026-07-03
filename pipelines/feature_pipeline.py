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


def get_previous_aqi(fs):
    """Fetch last stored AQI to compute change rate."""
    try:
        fg  = fs.get_feature_group(name="aqi_features", version=1)
        df  = fg.read()
        df  = df.sort_values("date", ascending=False)
        return float(df["aqi"].iloc[0])
    except:
        return None


def build_row(poll, weather, prev_aqi=None):
    now = datetime.utcnow()
    aqi = poll["aqi"]

    # AQI change rate
    if prev_aqi is not None:
        aqi_change_rate = float(aqi - prev_aqi)
    else:
        aqi_change_rate = 0.0

    return {
        "date":            now.strftime("%Y-%m-%d %H:%M:%S"),
        # Pollutants
        "aqi":             aqi,
        "pm25":            poll["pm25"],
        "pm10":            poll["pm10"],
        "no2":             poll["no2"],
        "so2":             poll["so2"],
        "co":              poll["co"],
        "o3":              poll["o3"],
        # Weather
        "temperature":     weather["temperature"],
        "humidity":        weather["humidity"],
        # Time-based features
        "hour":            int(now.hour),
        "day":             int(now.day),
        "month":           int(now.month),
        "day_of_week":     int(now.weekday()),
        # Derived feature
        "aqi_change_rate": aqi_change_rate,
    }


def store_features(df, fs):
    df["date"] = pd.to_datetime(df["date"])
    fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["date"],
        description="AQI features for Karachi with time features and change rate",
        event_time="date",
    )
    fg.insert(df, write_options={
        "start_offline_materialization": False
    })
    print(f"✅ Stored successfully!")

if __name__ == "__main__":
    print("Fetching live data for Karachi...")

    project = hopsworks.login(
        host=HOPSWORKS_HOST,
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT
    )
    fs = project.get_feature_store()

    poll    = fetch_live_pollution()
    weather = fetch_live_weather()
    prev_aqi = get_previous_aqi(fs)

    row = build_row(poll, weather, prev_aqi)
    df  = pd.DataFrame([row])
    print(df.to_string())
    store_features(df, fs)