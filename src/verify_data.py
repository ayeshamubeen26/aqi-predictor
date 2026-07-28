from feature_store import get_feature_store

fs = get_feature_store()
fg = fs.get_feature_group("aqi_features", version=1)

df = fg.read()

print(f"Total rows: {len(df)}")
print(f"\nRows per city:\n{df['city'].value_counts()}")
print(f"\nDate range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"\nAny missing values?\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(f"\nDuplicate (city, timestamp) pairs: {df.duplicated(subset=['city', 'timestamp']).sum()}")
print(f"\nSample AQI-relevant stats:\n{df[['pm2_5', 'pm10', 'co']].describe()}")