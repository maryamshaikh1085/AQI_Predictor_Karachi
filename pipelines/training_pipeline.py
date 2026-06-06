import os
import pandas as pd
import numpy as np
import joblib
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import hopsworks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
import pickle
from tensorflow.keras.layers import Input


load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")
HOPSWORKS_HOST    = os.getenv("HOPSWORKS_HOST")

FEATURE_COLS = ["pm25", "pm10", "no2", "so2", "co", "o3", "temperature", "humidity"]
TARGET_COL   = "aqi"


def fetch_training_data():
    print("Connecting to Hopsworks...")
    project = hopsworks.login(
        host=HOPSWORKS_HOST,
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT
    )
    fs  = project.get_feature_store()
    fg  = fs.get_feature_group(name="aqi_features", version=1)
    df  = fg.read()
    df["date"] = pd.to_datetime(df["date"])
    df  = df.sort_values("date").reset_index(drop=True)
    print(f"✅ Fetched {len(df)} rows")
    print(df[["date", "aqi", "pm25", "pm10"]].head())
    return df, project


def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    mae   = mean_absolute_error(y_test, preds)
    r2    = r2_score(y_test, preds)
    print(f"\n{name}:")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  R²   : {r2:.4f}")
    return {"name": name, "model": model, "rmse": rmse, "mae": mae, "r2": r2}


def train_models(df):
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    X  = df[FEATURE_COLS]
    y  = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTraining on {len(X_train)} rows, testing on {len(X_test)} rows")

    results = []

    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    results.append(evaluate("Random Forest", rf, X_test, y_test))

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    results.append(evaluate("Ridge Regression", ridge, X_test, y_test))

    xgb = XGBRegressor(n_estimators=200, random_state=42, verbosity=0)
    xgb.fit(X_train, y_train)
    results.append(evaluate("XGBoost", xgb, X_test, y_test))

    gb = GradientBoostingRegressor(n_estimators=200, random_state=42)
    gb.fit(X_train, y_train)
    results.append(evaluate("Gradient Boosting", gb, X_test, y_test))

    # Neural Network
    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    nn = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    nn.compile(optimizer="adam", loss="mse")
    nn.fit(X_train_scaled, y_train, epochs=50, verbose=0)

    nn_preds = nn.predict(X_test_scaled).flatten()
    nn_rmse  = np.sqrt(mean_squared_error(y_test, nn_preds))
    nn_mae   = mean_absolute_error(y_test, nn_preds)
    nn_r2    = r2_score(y_test, nn_preds)
    print(f"\nNeural Network:")
    print(f"  RMSE : {nn_rmse:.4f}")
    print(f"  MAE  : {nn_mae:.4f}")
    print(f"  R²   : {nn_r2:.4f}")
    results.append({
        "name":  "Neural Network",
        "model": nn,
        "rmse":  nn_rmse,
        "mae":   nn_mae,
        "r2":    nn_r2
    })

    best = min(results, key=lambda x: x["rmse"])
    print(f"\n🏆 Best model: {best['name']} (RMSE={best['rmse']:.4f}, R²={best['r2']:.4f})")
    return best


def save_model(best, project):
    if best["name"] == "Neural Network":
        path = "best_aqi_model.keras"
        best["model"].save(path)
    else:
        path = "best_aqi_model.pkl"
        joblib.dump(best["model"], path)

    mr    = project.get_model_registry()
    model = mr.sklearn.create_model(
        name="aqi_predictor",
        metrics={
            "rmse": round(best["rmse"], 4),
            "mae":  round(best["mae"],  4),
            "r2":   round(best["r2"],   4)
        },
        description=f"Best model: {best['name']} predicting AQI (1-5) for Karachi"
    )
    model.save(path)
    print(f"✅ Model saved! Best: {best['name']}")
    
if __name__ == "__main__":
    df, project = fetch_training_data()
    best        = train_models(df)
    save_model(best, project)