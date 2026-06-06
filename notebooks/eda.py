import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import hopsworks

load_dotenv()

# ─────────────────────────────────────────
# 1. LOAD DATA FROM HOPSWORKS
# ─────────────────────────────────────────

project = hopsworks.login(
    host=os.getenv("HOPSWORKS_HOST"),
    api_key_value=os.getenv("HOPSWORKS_API_KEY"),
    project=os.getenv("HOPSWORKS_PROJECT")
)
fs = project.get_feature_store()
fg = fs.get_feature_group(name="aqi_features", version=1)
df = fg.read()
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

print(f"Dataset shape: {df.shape}")
print(df.describe())

# ─────────────────────────────────────────
# 2. AQI DISTRIBUTION
# ─────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Count plot
aqi_counts = df["aqi"].value_counts().sort_index()
colors = ["#00e400", "#ffff00", "#ff7e00", "#ff0000", "#8f3f97"]
axes[0].bar(
    ["1-Good", "2-Fair", "3-Moderate", "4-Poor", "5-Very Poor"],
    [aqi_counts.get(i, 0) for i in range(1, 6)],
    color=colors
)
axes[0].set_title("AQI Category Distribution")
axes[0].set_xlabel("AQI Category")
axes[0].set_ylabel("Count")
for i, v in enumerate([aqi_counts.get(i, 0) for i in range(1, 6)]):
    axes[0].text(i, v + 0.5, str(v), ha="center", fontweight="bold")

# Pie chart
axes[1].pie(
    [aqi_counts.get(i, 0) for i in range(1, 6)],
    labels=["Good", "Fair", "Moderate", "Poor", "Very Poor"],
    colors=colors,
    autopct="%1.1f%%",
    startangle=90
)
axes[1].set_title("AQI Distribution (Pie)")

plt.suptitle("Karachi AQI Distribution", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("notebooks/aqi_distribution.png", dpi=150)
plt.show()
print("✅ Saved aqi_distribution.png")

# ─────────────────────────────────────────
# 3. AQI TREND OVER TIME
# ─────────────────────────────────────────

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df["date"], df["aqi"], color="#ff7e00", linewidth=1.5, alpha=0.8)
ax.fill_between(df["date"], df["aqi"], alpha=0.1, color="#ff7e00")
ax.axhline(y=4, color="red",    linestyle="--", linewidth=1, label="Poor (4)")
ax.axhline(y=3, color="orange", linestyle="--", linewidth=1, label="Moderate (3)")
ax.axhline(y=2, color="green",  linestyle="--", linewidth=1, label="Fair (2)")
ax.set_yticks([1, 2, 3, 4, 5])
ax.set_yticklabels(["1-Good", "2-Fair", "3-Moderate", "4-Poor", "5-Very Poor"])
ax.set_title("Karachi AQI Trend Over Time", fontsize=14, fontweight="bold")
ax.set_xlabel("Date")
ax.legend()
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("notebooks/aqi_trend.png", dpi=150)
plt.show()
print("✅ Saved aqi_trend.png")

# ─────────────────────────────────────────
# 4. POLLUTANT DISTRIBUTIONS
# ─────────────────────────────────────────

pollutants = ["pm25", "pm10", "no2", "so2", "co", "o3"]
fig, axes  = plt.subplots(2, 3, figsize=(15, 8))
axes       = axes.flatten()

for i, col in enumerate(pollutants):
    axes[i].hist(df[col].dropna(), bins=30, color="#1e90ff", edgecolor="white", alpha=0.8)
    axes[i].axvline(df[col].mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean: {df[col].mean():.2f}")
    axes[i].set_title(f"{col.upper()} Distribution")
    axes[i].set_xlabel("µg/m³")
    axes[i].set_ylabel("Frequency")
    axes[i].legend()

plt.suptitle("Pollutant Distributions — Karachi", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("notebooks/pollutant_distributions.png", dpi=150)
plt.show()
print("✅ Saved pollutant_distributions.png")

# ─────────────────────────────────────────
# 5. CORRELATION HEATMAP
# ─────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 8))
cols    = ["aqi", "pm25", "pm10", "no2", "so2", "co", "o3", "temperature", "humidity"]
corr    = df[cols].corr()

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="RdYlGn",
    center=0,
    ax=ax,
    square=True,
    linewidths=0.5
)
ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("notebooks/correlation_heatmap.png", dpi=150)
plt.show()
print("✅ Saved correlation_heatmap.png")

# ─────────────────────────────────────────
# 6. AQI BY MONTH
# ─────────────────────────────────────────

df["month_name"] = df["date"].dt.strftime("%b")
df["month_num"]  = df["date"].dt.month
monthly_aqi      = df.groupby(["month_num", "month_name"])["aqi"].mean().reset_index()
monthly_aqi      = monthly_aqi.sort_values("month_num")

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(
    monthly_aqi["month_name"],
    monthly_aqi["aqi"],
    color="#ff7e00",
    edgecolor="white"
)
ax.set_title("Average AQI by Month — Karachi", fontsize=14, fontweight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("Average AQI (1-5)")
ax.set_ylim(0, 5)
ax.axhline(y=3, color="orange", linestyle="--", linewidth=1, label="Moderate threshold")
ax.bar_label(bars, fmt="%.2f", padding=3)
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("notebooks/aqi_by_month.png", dpi=150)
plt.show()
print("✅ Saved aqi_by_month.png")

# ─────────────────────────────────────────
# 7. AQI BY HOUR OF DAY
# ─────────────────────────────────────────

hourly_aqi = df.groupby("hour")["aqi"].mean()

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(hourly_aqi.index, hourly_aqi.values, color="#8f3f97", linewidth=2, marker="o")
ax.fill_between(hourly_aqi.index, hourly_aqi.values, alpha=0.1, color="#8f3f97")
ax.set_title("Average AQI by Hour of Day — Karachi", fontsize=14, fontweight="bold")
ax.set_xlabel("Hour (UTC)")
ax.set_ylabel("Average AQI (1-5)")
ax.set_xticks(range(0, 24))
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("notebooks/aqi_by_hour.png", dpi=150)
plt.show()
print("✅ Saved aqi_by_hour.png")

# ─────────────────────────────────────────
# 8. POLLUTANT TRENDS OVER TIME
# ─────────────────────────────────────────

fig, axes = plt.subplots(3, 2, figsize=(15, 12))
axes      = axes.flatten()
colors    = ["#ff7e00", "#ff0000", "#8f3f97", "#ffff00", "#00e400", "#1e90ff"]

for i, (col, color) in enumerate(zip(pollutants, colors)):
    axes[i].plot(df["date"], df[col], color=color, linewidth=1, alpha=0.8)
    axes[i].set_title(f"{col.upper()} Over Time")
    axes[i].set_xlabel("Date")
    axes[i].set_ylabel("µg/m³")
    axes[i].grid(True, alpha=0.3)
    plt.setp(axes[i].xaxis.get_majorticklabels(), rotation=45)

plt.suptitle("Pollutant Trends Over Time — Karachi", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("notebooks/pollutant_trends.png", dpi=150)
plt.show()
print("✅ Saved pollutant_trends.png")

print("\n✅ EDA Complete! All plots saved in notebooks/ folder.")