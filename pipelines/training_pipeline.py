import os
import pandas as pd
import numpy as np
import joblib
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import hopsworks

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")
HOPSWORKS_HOST    = os.getenv("HOPSWORKS_HOST")

FEATURE_COLS = [
    "aqi", "pm25", "pm10", "no2", "so2", "co", "o3",
    "temperature", "humidity", "precipitation"
]
TARGET_COL = "next_day_aqi"


# ─────────────────────────────────────────
# 1. FETCH DATA FROM HOPSWORKS
# ─────────────────────────────────────────

def fetch_training_data():
    print("Connecting to Hopsworks...")
    project = hopsworks.login(
        host=HOPSWORKS_HOST,
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT
    )
    fs = project.get_feature_store()

    fg = fs.get_feature_group(name="aqi_features", version=3)
    df = fg.read()

    print(f"✅ Fetched {len(df)} rows from Feature Store")
    print(df[["date", "aqi", "next_day_aqi"]].head())
    return df, project


# ─────────────────────────────────────────
# 2. TRAIN & EVALUATE MODELS
# ─────────────────────────────────────────

def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    mae   = mean_absolute_error(y_test, preds)
    r2    = r2_score(y_test, preds)
    print(f"\n{name}:")
    print(f"  RMSE : {rmse:.2f}")
    print(f"  MAE  : {mae:.2f}")
    print(f"  R²   : {r2:.4f}")
    return {"name": name, "model": model, "rmse": rmse, "mae": mae, "r2": r2}


def train_models(df):
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\nTraining on {len(X_train)} rows, testing on {len(X_test)} rows")

    models = []

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    models.append(evaluate("Random Forest", rf, X_test, y_test))

    # Ridge Regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    models.append(evaluate("Ridge Regression", ridge, X_test, y_test))

    # XGBoost
    xgb = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    xgb.fit(X_train, y_train)
    models.append(evaluate("XGBoost", xgb, X_test, y_test))

    # Pick best model by RMSE
    best = min(models, key=lambda x: x["rmse"])
    print(f"\n🏆 Best model: {best['name']} (RMSE={best['rmse']:.2f})")

    return best


# ─────────────────────────────────────────
# 3. SAVE MODEL TO HOPSWORKS MODEL REGISTRY
# ─────────────────────────────────────────

def save_model(best, project):
    model_path = "best_aqi_model.pkl"
    joblib.dump(best["model"], model_path)
    print(f"\nSaved model locally as {model_path}")

    mr = project.get_model_registry()

    aqi_model = mr.sklearn.create_model(
        name="aqi_predictor",
        metrics={
            "rmse": round(best["rmse"], 2),
            "mae":  round(best["mae"],  2),
            "r2":   round(best["r2"],   4),
        },
        description=f"Best model: {best['name']} trained on Karachi AQI data"
    )
    aqi_model.save(model_path)
    print(f"✅ Model saved to Hopsworks Model Registry!")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

if __name__ == "__main__":
    df, project = fetch_training_data()
    best        = train_models(df)
    save_model(best, project)