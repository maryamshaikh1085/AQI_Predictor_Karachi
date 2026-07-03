from dotenv import load_dotenv
load_dotenv()
import os, hopsworks, pandas as pd

project = hopsworks.login(
    host=os.getenv('HOPSWORKS_HOST'),
    api_key_value=os.getenv('HOPSWORKS_API_KEY'),
    project=os.getenv('HOPSWORKS_PROJECT')
)
fs = project.get_feature_store()
fg = fs.get_feature_group('aqi_features', version=1)
df = fg.read()
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date', ascending=False)
print(f'Total rows: {len(df)}')
print(df[['date','aqi','pm25']].head(10))