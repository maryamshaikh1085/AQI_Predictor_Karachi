import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dotenv import load_dotenv
import hopsworks
import requests

load_dotenv()

HOPSWORKS_API_KEY   = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT   = os.getenv("HOPSWORKS_PROJECT")
HOPSWORKS_HOST      = os.getenv("HOPSWORKS_HOST")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
LAT, LON            = 24.8607, 67.0011

FEATURE_COLS = ["pm25", "pm10", "no2", "so2", "co", "o3", "temperature", "humidity"]


def aqi_info(aqi):
    aqi = max(1, min(5, round(float(aqi))))
    if aqi == 1: return 1, "Good",      "#00e400", "✅ Air quality is good. Enjoy outdoor activities!"
    if aqi == 2: return 2, "Fair",      "#ffff00", "🟡 Acceptable. Sensitive people should limit prolonged outdoor exertion."
    if aqi == 3: return 3, "Moderate",  "#ff7e00", "🟠 Sensitive groups should reduce outdoor activity."
    if aqi == 4: return 4, "Poor",      "#ff0000", "🔴 Everyone should reduce outdoor exertion. Wear a mask."
    return           5, "Very Poor", "#8f3f97", "🚨 Very unhealthy! Avoid outdoor activities. Stay indoors."


@st.cache_resource
def connect_hopsworks():
    return hopsworks.login(
        host=HOPSWORKS_HOST,
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT
    )


@st.cache_data(ttl=3600)
def load_features():
    project = connect_hopsworks()
    fs  = project.get_feature_store()
    fg  = fs.get_feature_group(name="aqi_features", version=1)
    df  = fg.read()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_model():
    return joblib.load("best_aqi_model.pkl")


def fetch_live():
    # Pollution
    url  = (f"http://api.openweathermap.org/data/2.5/air_pollution"
            f"?lat={LAT}&lon={LON}&appid={OPENWEATHER_API_KEY}")
    data = requests.get(url).json()
    comp = data["list"][0]["components"]
    aqi  = int(data["list"][0]["main"]["aqi"])

    # Weather
    url2  = (f"https://api.openweathermap.org/data/2.5/weather"
             f"?lat={LAT}&lon={LON}&appid={OPENWEATHER_API_KEY}&units=metric")
    data2 = requests.get(url2).json()

    return {
        "aqi":         aqi,
        "pm25":        float(comp.get("pm2_5", 0.0)),
        "pm10":        float(comp.get("pm10",  0.0)),
        "no2":         float(comp.get("no2",   0.0)),
        "so2":         float(comp.get("so2",   0.0)),
        "co":          float(comp.get("co",    0.0)),
        "o3":          float(comp.get("o3",    0.0)),
        "temperature": float(data2["main"]["temp"]),
        "humidity":    float(data2["main"]["humidity"]),
    }


def forecast_3_days(model):
    # Get future pollution forecast from OpenWeather
    url  = (f"http://api.openweathermap.org/data/2.5/air_pollution/forecast"
            f"?lat={LAT}&lon={LON}&appid={OPENWEATHER_API_KEY}")
    data = requests.get(url).json()

    # Get future weather forecast from OpenWeather
    url2  = (f"https://api.openweathermap.org/data/2.5/forecast"
             f"?lat={LAT}&lon={LON}&appid={OPENWEATHER_API_KEY}&units=metric")
    data2 = requests.get(url2).json()

    # Build weather lookup by date
    weather_by_date = {}
    for w in data2["list"]:
        wdate = datetime.utcfromtimestamp(w["dt"]).date()
        if wdate not in weather_by_date:
            weather_by_date[wdate] = {
                "temperature": float(w["main"]["temp"]),
                "humidity":    float(w["main"]["humidity"]),
            }

    forecasts  = []
    seen_dates = []
    today      = datetime.utcnow().date()

    for item in data["list"]:
        dt   = datetime.utcfromtimestamp(item["dt"])
        date = dt.date()

        if date <= today:
            continue
        if date in seen_dates:
            continue

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
        forecasts.append({
            "day": dt.strftime("%b %d"),
            "aqi": predicted
        })

        if len(forecasts) == 3:
            break

    return forecasts


def main():
    st.set_page_config(
        page_title="Karachi AQI Predictor",
        page_icon="🌫️",
        layout="wide"
    )

    st.title("🌫️ Karachi Air Quality Index Predictor")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC | Source: OpenWeatherMap")
    st.divider()

    with st.spinner("Loading data..."):
        try:
            df    = load_features()
            model = load_model()
            live  = fetch_live()
        except Exception as e:
            st.error(f"Error: {e}")
            return

    aqi_val, category, color, advice = aqi_info(live["aqi"])

    # Alert banner
    if aqi_val >= 4:
        st.error(f"⚠️ AIR QUALITY ALERT — {category} (AQI: {aqi_val}/5) | {advice}")
    elif aqi_val == 3:
        st.warning(f"⚠️ {category} (AQI: {aqi_val}/5) | {advice}")
    else:
        st.success(f"✅ {category} (AQI: {aqi_val}/5) | {advice}")

    # Current conditions
    st.subheader("📍 Current Conditions in Karachi")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("🌫️ AQI (1-5)",   f"{aqi_val} — {category}")
    c2.metric("🌡️ Temperature", f"{live['temperature']}°C")
    c3.metric("💧 Humidity",    f"{live['humidity']}%")
    c4.metric("💨 PM2.5",       f"{live['pm25']} µg/m³")
    c5.metric("💨 PM10",        f"{live['pm10']} µg/m³")
    c6.metric("🌿 O3",          f"{live['o3']} µg/m³")

    st.divider()

    # Forecast + Pollutants
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("📅 3-Day AQI Forecast (Model Predicted)")
        forecasts = forecast_3_days(model)
        for f in forecasts:
            _, cat, col, adv = aqi_info(f["aqi"])
            st.markdown(
                f"""<div style='background:{col};padding:12px;border-radius:10px;
                    margin-bottom:8px;color:#000;font-weight:bold;'>
                    📅 {f['day']} &nbsp;&nbsp; AQI: {f['aqi']}/5 &nbsp;&nbsp; {cat}
                </div>""",
                unsafe_allow_html=True
            )
            st.caption(adv)

    with col_right:
        st.subheader("🧪 Current Pollutant Levels")
        pollutants = {
            "PM2.5":  live["pm25"],
            "PM10":   live["pm10"],
            "NO2":    live["no2"],
            "SO2":    live["so2"],
            "CO":     live["co"],
            "O3":     live["o3"],
        }
        fig, ax = plt.subplots(figsize=(6, 3))
        bars = ax.barh(
            list(pollutants.keys()),
            list(pollutants.values()),
            color=["#ff7e00","#ff0000","#8f3f97","#ffff00","#00e400","#1e90ff"]
        )
        ax.set_xlabel("Concentration (µg/m³)")
        ax.set_title("Pollutant Breakdown")
        ax.bar_label(bars, fmt="%.2f", padding=3)
        plt.tight_layout()
        st.pyplot(fig)

    st.divider()

    # Historical AQI trend
    st.subheader("📈 Historical AQI Trend (Last 120 Days)")
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(df["date"], df["aqi"], color="#ff7e00", linewidth=1.5, label="AQI")
    ax2.axhline(y=4, color="red",    linestyle="--", linewidth=1, label="Poor (4)")
    ax2.axhline(y=3, color="orange", linestyle="--", linewidth=1, label="Moderate (3)")
    ax2.axhline(y=2, color="green",  linestyle="--", linewidth=1, label="Fair (2)")
    ax2.fill_between(df["date"], df["aqi"], alpha=0.1, color="#ff7e00")
    ax2.set_yticks([1, 2, 3, 4, 5])
    ax2.set_yticklabels(["1-Good","2-Fair","3-Moderate","4-Poor","5-Very Poor"])
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig2)

    st.divider()

    # Weather trends
    st.subheader("🌤️ Weather Trends")
    w1, w2 = st.columns(2)

    with w1:
        fig3, ax3 = plt.subplots(figsize=(5, 3))
        ax3.plot(df["date"], df["temperature"], color="#ff4500", linewidth=1.5)
        ax3.set_title("Temperature (°C)")
        ax3.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig3)

    with w2:
        fig4, ax4 = plt.subplots(figsize=(5, 3))
        ax4.plot(df["date"], df["humidity"], color="#1e90ff", linewidth=1.5)
        ax4.set_title("Humidity (%)")
        ax4.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig4)

    st.divider()

    # Raw data
    st.subheader("🗃️ Raw Feature Data")
    st.dataframe(
        df.sort_values("date", ascending=False).head(30).reset_index(drop=True),
        use_container_width=True
    )

    st.caption("Built with ❤️ using Hopsworks + OpenWeatherMap + Streamlit | Karachi AQI Predictor")


if __name__ == "__main__":
    main()