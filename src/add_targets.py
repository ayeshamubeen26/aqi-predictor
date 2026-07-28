import pandas as pd
from feature_store import get_feature_store
from calculate_aqi import calculate_aqi

def add_aqi_and_targets(df):
    """
    Adds a computed AQI column, then adds three target columns:
    the AQI value 24, 48, and 72 hours ahead, per city.
    """
    df = df.sort_values(["city", "timestamp"]).reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Compute AQI for every row from pm2.5 and pm10
    df["aqi"] = df.apply(lambda row: calculate_aqi(row["pm2_5"], row["pm10"]), axis=1)

    # For each city, look forward N rows (N hours, since data is hourly)
    # to find the AQI value at that future point, and use it as this
    # row's target.
    grouped = df.groupby("city")
    df["target_aqi_24h"] = grouped["aqi"].shift(-24)
    df["target_aqi_48h"] = grouped["aqi"].shift(-48)
    df["target_aqi_72h"] = grouped["aqi"].shift(-72)

    df["timestamp"] = df["timestamp"].astype(str)

    return df

if __name__ == "__main__":
    fs = get_feature_store()
    fg = fs.get_feature_group("aqi_features_enriched", version=1)

    print("Reading enriched dataset from Hopsworks...")
    df = fg.read()
    print(f"Loaded {len(df)} rows")

    df_with_targets = add_aqi_and_targets(df)

    print("\nSample row with AQI and targets:")
    sample_city = df_with_targets["city"].iloc[0]
    sample = df_with_targets[df_with_targets["city"] == sample_city].iloc[50]
    print(sample[["city", "timestamp", "pm2_5", "pm10", "aqi",
                   "target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]])

    print(f"\nRows missing target_aqi_72h (expected near the END of each "
          f"city's history, since there's no future data yet for those hours): "
          f"{df_with_targets['target_aqi_72h'].isnull().sum()}")

    print(f"\nOverall AQI distribution:\n{df_with_targets['aqi'].describe()}")

    print("\nMissing target_aqi_72h per city:")
    print(df_with_targets.groupby("city")["target_aqi_72h"].apply(lambda x: x.isnull().sum()))

    print("\nRow counts per city (for comparison):")
    print(df_with_targets["city"].value_counts())
    final_fg = fs.get_or_create_feature_group(
        name="aqi_features_final",
        version=1,
        primary_key=["city", "timestamp"],
        description="Final AQI features with computed AQI and 24/48/72h targets, ready for model training",
        online_enabled=False,
        time_travel_format="HUDI"
    )

    print("\nWriting final dataset to aqi_features_final...")
    final_fg.insert(df_with_targets)
    print(f"Inserted {len(df_with_targets)} rows into aqi_features_final.")