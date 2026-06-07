import os
import hopsworks
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

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

df.to_csv("aqi_data.csv", index=False)
print(f"✅ Saved! {len(df)} rows in aqi_data.csv")
print(df.head())