import os
import pandas as pd
from feature_store import get_feature_store


def load_final_dataset():
    """
    Loads the full aqi_features_final dataset from Hopsworks,
    same source used for training.
    """
    fs = get_feature_store()
    fg = fs.get_feature_group("aqi_features_final", version=1)

    print("Reading final dataset from Hopsworks...")
    df = fg.read()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["city", "timestamp"]).reset_index(drop=True)

    print(f"Loaded {len(df)} rows")
    return df


if __name__ == "__main__":
    aqi_features_final = load_final_dataset()

    # Always resolve to the project's data/ folder, regardless of
    # which directory this script is run from
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "data")
    os.makedirs(data_dir, exist_ok=True)

    # Combined file — all cities in one table
    combined_path = os.path.join(data_dir, "aqi_features_final.csv")
    aqi_features_final.to_csv(combined_path, index=False)
    print(f"Saved combined file: {len(aqi_features_final)} rows -> {combined_path}")

    # Per-city files
    for city, city_df in aqi_features_final.groupby("city"):
        filename = os.path.join(data_dir, f"aqi_features_final_{city.lower().replace(' ', '_')}.csv")
        city_df.to_csv(filename, index=False)
        print(f"Saved {city}: {len(city_df)} rows -> {filename}")