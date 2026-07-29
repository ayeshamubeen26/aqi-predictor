import os
from feature_store import get_feature_store

def export_feature_group_to_csv(feature_group_name, version=1, output_path=None):
    """
    Reads a feature group from Hopsworks and saves it as a local CSV file.
    Useful as a backup / for offline inspection, separate from the
    live feature store itself.
    """
    fs = get_feature_store()
    fg = fs.get_feature_group(feature_group_name, version=version)

    print(f"Reading {feature_group_name} from Hopsworks...")
    df = fg.read()
    print(f"Loaded {len(df)} rows")

    if output_path is None:
        output_path = f"../data/{feature_group_name}.csv"

    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    export_feature_group_to_csv("aqi_features_final", version=1)