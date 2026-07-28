import pandas as pd
from feature_store import get_feature_store

def add_rolling_and_lag_features(df):
    """
    Takes the full feature dataframe (all cities, all timestamps) and adds
    rolling averages and lag features, computed per city so one city's
    history never leaks into another's.
    """
    df = df.sort_values(["city", "timestamp"]).reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    grouped = df.groupby("city")

    df["pm2_5_lag_1h"] = grouped["pm2_5"].shift(1)
    df["pm2_5_lag_3h"] = grouped["pm2_5"].shift(3)
    df["pm2_5_lag_24h"] = grouped["pm2_5"].shift(24)

    df["pm2_5_roll_3h"] = grouped["pm2_5"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df["pm2_5_roll_6h"] = grouped["pm2_5"].transform(lambda x: x.rolling(6, min_periods=1).mean())
    df["pm2_5_roll_24h"] = grouped["pm2_5"].transform(lambda x: x.rolling(24, min_periods=1).mean())

    # Convert timestamp back to string, since Hopsworks primary keys work
    # more reliably as strings than as pandas datetime objects
    df["timestamp"] = df["timestamp"].astype(str)

    return df

if __name__ == "__main__":
    fs = get_feature_store()
    fg = fs.get_feature_group("aqi_features", version=1)

    print("Reading full dataset from Hopsworks...")
    df = fg.read()
    print(f"Loaded {len(df)} rows")

    df_with_features = add_rolling_and_lag_features(df)

    print(f"\nRows with any missing lag/rolling values: "
          f"{df_with_features[['pm2_5_lag_24h']].isnull().sum().sum()}")

    # Create (or connect to) the enriched feature group
    enriched_fg = fs.get_or_create_feature_group(
        name="aqi_features_enriched",
        version=1,
        primary_key=["city", "timestamp"],
        description="AQI features enriched with rolling averages and lag features, ready for training",
        online_enabled=False,
        time_travel_format="HUDI"
    )

    print("\nWriting enriched dataset to aqi_features_enriched...")
    enriched_fg.insert(df_with_features)
    print(f"Inserted {len(df_with_features)} rows into aqi_features_enriched.")