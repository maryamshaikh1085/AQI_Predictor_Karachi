import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime, timedelta
from dotenv import load_dotenv
import hopsworks
import requests
import shap

load_dotenv()

HOPSWORKS_API_KEY   = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT   = os.getenv("HOPSWORKS_PROJECT")
HOPSWORKS_HOST      = os.getenv("HOPSWORKS_HOST")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
LAT, LON            = 24.8607, 67.0011

FEATURE_COLS = ["pm25", "pm10", "no2", "so2", "co", "o3", "temperature", "humidity"]

matplotlib.rcParams.update({
    "figure.facecolor":  "#ffffff",
    "axes.facecolor":    "#ffffff",
    "axes.edgecolor":    "#d6e4f0",
    "axes.labelcolor":   "#1a2b4a",
    "xtick.color":       "#6b8cae",
    "ytick.color":       "#6b8cae",
    "text.color":        "#1a2b4a",
    "grid.color":        "#e8f0f7",
    "legend.facecolor":  "#ffffff",
    "legend.edgecolor":  "#d6e4f0",
})


def aqi_info(aqi):
    aqi = max(1, min(5, round(float(aqi))))
    if aqi == 1: return 1, "Good",      "#00e676", "Air quality is good. Enjoy outdoor activities!"
    if aqi == 2: return 2, "Fair",      "#ffee58", "Acceptable. Sensitive people should limit prolonged outdoor exertion."
    if aqi == 3: return 3, "Moderate",  "#ffa726", "Sensitive groups should reduce outdoor activity."
    if aqi == 4: return 4, "Poor",      "#ef5350", "Everyone should reduce outdoor exertion. Wear a mask."
    return           5, "Very Poor", "#ab47bc", "Very unhealthy! Avoid outdoor activities. Stay indoors."


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
    url  = (f"http://api.openweathermap.org/data/2.5/air_pollution"
            f"?lat={LAT}&lon={LON}&appid={OPENWEATHER_API_KEY}")
    data = requests.get(url).json()
    comp = data["list"][0]["components"]
    aqi  = int(data["list"][0]["main"]["aqi"])

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
    url  = (f"http://api.openweathermap.org/data/2.5/air_pollution/forecast"
            f"?lat={LAT}&lon={LON}&appid={OPENWEATHER_API_KEY}")
    data = requests.get(url).json()

    url2  = (f"https://api.openweathermap.org/data/2.5/forecast"
             f"?lat={LAT}&lon={LON}&appid={OPENWEATHER_API_KEY}&units=metric")
    data2 = requests.get(url2).json()

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
        forecasts.append({"day": dt.strftime("%A, %b %d"), "aqi": predicted})

        if len(forecasts) == 3:
            break

    return forecasts


# ── SVG icon helpers ──────────────────────────────────────────────────────────
def icon_pin():
    return '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#0066cc" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:6px"><path d="M21 10c0 7-9 13-9 13S3 17 3 10a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/></svg>'

def icon_calendar():
    return '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#0066cc" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:6px"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></svg>'

def icon_flask():
    return '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#0066cc" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:6px"><path d="M9 3h6v11l4 7H5l4-7V3z"/><line x1="9" y1="9" x2="15" y2="9"/></svg>'

def icon_search():
    return '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#0066cc" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:6px"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>'

def icon_trending():
    return '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#0066cc" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:6px"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>'

def icon_cloud():
    return '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#0066cc" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:6px"><path d="M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z"/></svg>'

def icon_table():
    return '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#0066cc" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:6px"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="3" y1="15" x2="21" y2="15"/><line x1="9" y1="3" x2="9" y2="21"/><line x1="15" y1="3" x2="15" y2="21"/></svg>'


def main():
    st.set_page_config(
        page_title="Karachi AQI Predictor",
        page_icon="🌫️",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

   

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #f0f4f8;
        color: #1a2b4a;
    }

    h1, h2, h3 { font-family: 'Poppins', sans-serif; }

    .hero-box {
        background: linear-gradient(135deg, #003366 0%, #0066cc 60%, #0099ff 100%);
        border-radius: 20px;
        padding: 36px 44px;
        margin-bottom: 28px;
        box-shadow: 0 8px 32px rgba(0,102,204,0.25);
    }

    .hero-title {
        font-family: 'Poppins', sans-serif;
        font-size: 2.4rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .hero-subtitle {
        color: #a8d4ff;
        font-size: 0.9rem;
        margin-top: 6px;
        letter-spacing: 2px;
        text-transform: uppercase;
        font-weight: 500;
    }

    .hero-meta {
        color: #cce5ff;
        margin-top: 10px;
        font-size: 0.88rem;
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid #d6e4f0;
        border-top: 4px solid #0066cc;
        border-radius: 14px;
        padding: 20px 16px;
        text-align: center;
        box-shadow: 0 2px 12px rgba(0,102,204,0.08);
        transition: transform 0.2s, box-shadow 0.2s;
    }

    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0,102,204,0.15);
    }

    .metric-label {
        font-size: 0.72rem;
        color: #6b8cae;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 10px;
        font-weight: 600;
    }

    .metric-value {
        font-family: 'Poppins', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #003366;
    }

    .metric-unit {
        font-size: 0.72rem;
        color: #6b8cae;
        margin-top: 4px;
        font-weight: 500;
    }

    .forecast-card {
        border-radius: 14px;
        padding: 18px 24px;
        margin-bottom: 12px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border: 1px solid rgba(255,255,255,0.6);
    }

    .forecast-day {
        font-family: 'Poppins', sans-serif;
        font-size: 0.95rem;
        font-weight: 700;
        color: #003366;
    }

    .forecast-aqi {
        font-family: 'Poppins', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #003366;
        text-align: right;
    }

    .forecast-cat {
        font-size: 0.8rem;
        color: #003366;
        font-weight: 600;
        text-align: right;
    }

    .forecast-advice {
        font-size: 0.78rem;
        color: #1a2b4a;
        margin-top: 4px;
        opacity: 0.85;
    }

    .section-header {
        font-family: 'Poppins', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        color: #003366;
        margin-bottom: 16px;
        padding-bottom: 10px;
        border-bottom: 2px solid #0066cc;
        display: flex;
        align-items: center;
    }

    .alert-banner {
        border-radius: 12px;
        padding: 16px 22px;
        margin-bottom: 24px;
        font-weight: 600;
        font-size: 0.92rem;
        border-left: 5px solid;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    }

    .shap-info {
        background: linear-gradient(135deg, #e8f4ff, #f0f8ff);
        border: 1px solid #0066cc;
        border-radius: 10px;
        padding: 14px 20px;
        color: #003366;
        font-size: 0.88rem;
        margin-top: 10px;
        font-weight: 500;
    }

    .footer {
        text-align: center;
        padding: 28px 0 10px;
        color: #6b8cae;
        font-size: 0.8rem;
        border-top: 1px solid #d6e4f0;
        margin-top: 30px;
    }

    .stDataFrame { border-radius: 12px; overflow: hidden; }
    footer { display: none; }
    #MainMenu { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)

    # ── HERO ──
    st.markdown(f"""
    <div class="hero-box">
        <p class="hero-subtitle">Real-time Air Quality Intelligence &mdash; Karachi, Pakistan</p>
        <h1 style="
        color:#ffffff;
        font-family:'Poppins',sans-serif;
        font-size:2.4rem;
        font-weight:700;
        margin:0;
        letter-spacing:-0.5px;
    ">
        Karachi AQI Predictor
    </h1>
        <p class="hero-meta">
             &nbsp;&middot;&nbsp;
            Updated {datetime.now().strftime('%d %b %Y, %H:%M')} UTC
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Fetching live data..."):
        try:
            df    = load_features()
            model = load_model()
            live  = fetch_live()
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return

    aqi_val, category, color, advice = aqi_info(live["aqi"])

    # ── ALERT ──
    warn_icon = '<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:6px"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>'
    ok_icon   = '<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:6px"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>'

    if aqi_val >= 4:
        st.markdown(f"""
        <div class="alert-banner" style="background:#fff5f5; border-color:#ef5350; color:#c62828;">
            {warn_icon}<strong>AIR QUALITY ALERT</strong> — {category} (AQI {aqi_val}/5) &nbsp;|&nbsp; {advice}
        </div>""", unsafe_allow_html=True)
    elif aqi_val == 3:
        st.markdown(f"""
        <div class="alert-banner" style="background:#fff8f0; border-color:#ffa726; color:#e65100;">
            {warn_icon}<strong>{category}</strong> (AQI {aqi_val}/5) &nbsp;|&nbsp; {advice}
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="alert-banner" style="background:#f0fff6; border-color:#00c853; color:#1b5e20;">
            {ok_icon}<strong>{category}</strong> (AQI {aqi_val}/5) &nbsp;|&nbsp; {advice}
        </div>""", unsafe_allow_html=True)

    # ── CURRENT CONDITIONS ──
    st.markdown(f'<p class="section-header">{icon_pin()} Current Conditions in Karachi</p>',
                unsafe_allow_html=True)

    metrics = [
        ("AQI (1–5)",    f"{aqi_val} — {category}", ""),
        ("Temperature",  f"{live['temperature']}",   "°C"),
        ("Humidity",     f"{live['humidity']}",       "%"),
        ("PM2.5",        f"{live['pm25']}",           "µg/m³"),
        ("PM10",         f"{live['pm10']}",           "µg/m³"),
        ("O3",           f"{live['o3']}",             "µg/m³"),
    ]

    cols = st.columns(6)
    for col, (label, value, unit) in zip(cols, metrics):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-unit">{unit}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── FORECAST + POLLUTANTS ──
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown(f'<p class="section-header">{icon_calendar()} 3-Day AQI Forecast</p>',
                    unsafe_allow_html=True)
        forecasts = forecast_3_days(model)
        for f in forecasts:
            _, cat, col, adv = aqi_info(f["aqi"])
            st.markdown(f"""
            <div class="forecast-card" style="background:{col}20; border-left: 5px solid {col};">
                <div>
                    <div class="forecast-day">{f['day']}</div>
                    <div class="forecast-advice">{adv}</div>
                </div>
                <div>
                    <div class="forecast-aqi">{f['aqi']}/5</div>
                    <div class="forecast-cat">{cat}</div>
                </div>
            </div>""", unsafe_allow_html=True)

    with col_right:
        st.markdown(f'<p class="section-header">{icon_flask()} Current Pollutant Levels</p>',
                    unsafe_allow_html=True)
        pollutants = {
            "PM2.5": live["pm25"], "PM10": live["pm10"],
            "NO2":   live["no2"],  "SO2":  live["so2"],
            "CO":    live["co"],   "O3":   live["o3"],
        }
        # ── Background added to pollutant chart ──
        BG = "#e8f3ff"
        fig, ax = plt.subplots(figsize=(6, 3.5))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG)
        bar_colors = ["#0066cc","#0099ff","#33b5ff","#66ccff","#99ddff","#cceeff"]
        bars = ax.barh(
            list(pollutants.keys()),
            list(pollutants.values()),
            color=bar_colors, edgecolor="none", height=0.6
        )
        ax.set_xlabel("Concentration (µg/m³)", fontsize=9)
        ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=8, color="#1a2b4a")
        ax.spines[["top","right","left"]].set_visible(False)
        ax.tick_params(left=False)
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── SHAP ──
    st.markdown(f'<p class="section-header">{icon_search()} Feature Importance (SHAP)</p>',
                unsafe_allow_html=True)
    try:
        df_shap     = load_features()[FEATURE_COLS].dropna()
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df_shap)
        mean_shap   = pd.Series(
            np.abs(shap_values).mean(axis=0), index=FEATURE_COLS
        ).sort_values(ascending=False)

        # ── Background added to SHAP chart ──
        BG = "#e8f3ff"
        fig_s, ax_s = plt.subplots(figsize=(10, 3))
        fig_s.patch.set_facecolor(BG)
        ax_s.set_facecolor(BG)
        bar_c = ["#003366" if i == 0 else "#0066cc" if i == 1 else "#0099ff"
                 for i in range(len(mean_shap))]
        mean_shap.plot(kind="barh", ax=ax_s, color=bar_c, edgecolor="none")
        ax_s.set_xlabel("Mean |SHAP Value|", fontsize=9)
        ax_s.set_title("Which features drive AQI predictions the most?",
                       fontsize=10, pad=10, color="#003366")
        ax_s.invert_yaxis()
        ax_s.spines[["top","right","left"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_s)

        top_feature = mean_shap.index[0]
        st.markdown(f"""
        <div class="shap-info">
            <strong>Most influential feature: {top_feature.upper()}</strong> —
            strongest driver of AQI predictions in Karachi
        </div>""", unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"SHAP could not be computed: {e}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── HISTORICAL TREND ──
    st.markdown(f'<p class="section-header">{icon_trending()} Historical AQI Trend</p>',
                unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(df["date"], df["aqi"], color="#0066cc", linewidth=2, label="AQI")
    ax2.fill_between(df["date"], df["aqi"], alpha=0.1, color="#0066cc")
    ax2.axhline(y=4, color="#ef5350", linestyle="--", linewidth=0.8, label="Poor (4)")
    ax2.axhline(y=3, color="#ffa726", linestyle="--", linewidth=0.8, label="Moderate (3)")
    ax2.axhline(y=2, color="#00c853", linestyle="--", linewidth=0.8, label="Fair (2)")
    ax2.set_yticks([1, 2, 3, 4, 5])
    ax2.set_yticklabels(["Good","Fair","Moderate","Poor","Very Poor"], fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.spines[["top","right"]].set_visible(False)
    ax2.grid(True, alpha=0.3)
    plt.xticks(rotation=45, fontsize=8)
    plt.tight_layout()
    st.pyplot(fig2)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── WEATHER TRENDS ──
    st.markdown(f'<p class="section-header">{icon_cloud()} Weather Trends</p>',
                unsafe_allow_html=True)
    w1, w2 = st.columns(2, gap="large")

    with w1:
        fig3, ax3 = plt.subplots(figsize=(6, 3))
        ax3.plot(df["date"], df["temperature"], color="#0066cc", linewidth=1.5)
        ax3.fill_between(df["date"], df["temperature"], alpha=0.08, color="#0066cc")
        ax3.set_title("Temperature (°C)", fontsize=10, color="#003366")
        ax3.spines[["top","right"]].set_visible(False)
        ax3.grid(True, alpha=0.2)
        plt.xticks(rotation=45, fontsize=7)
        plt.tight_layout()
        st.pyplot(fig3)

    with w2:
        fig4, ax4 = plt.subplots(figsize=(6, 3))
        ax4.plot(df["date"], df["humidity"], color="#0099ff", linewidth=1.5)
        ax4.fill_between(df["date"], df["humidity"], alpha=0.08, color="#0099ff")
        ax4.set_title("Humidity (%)", fontsize=10, color="#003366")
        ax4.spines[["top","right"]].set_visible(False)
        ax4.grid(True, alpha=0.2)
        plt.xticks(rotation=45, fontsize=7)
        plt.tight_layout()
        st.pyplot(fig4)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── RAW DATA ──
    st.markdown(f'<p class="section-header">{icon_table()} Recent Feature Data</p>',
                unsafe_allow_html=True)
    st.dataframe(
        df.sort_values("date", ascending=False).head(30).reset_index(drop=True),
        use_container_width=True
    )

    # ── FOOTER ──
    st.markdown("""
    <div class="footer">
        <strong style="color:#0066cc;">Hopsworks</strong> &middot;
        <strong style="color:#0066cc;">OpenWeatherMap</strong> &middot;
        <strong style="color:#0066cc;">Streamlit</strong>
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()