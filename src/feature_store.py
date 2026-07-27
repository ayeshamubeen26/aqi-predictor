import os
import pandas as pd
import hopsworks
from dotenv import load_dotenv

load_dotenv()

def get_feature_store():
    """Connect to Hopsworks and return the project's feature store handle."""
    project = hopsworks.login(
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
        project=os.getenv("HOPSWORKS_PROJECT")
    )
    return project.get_feature_store()

def get_or_create_feature_group(fs):
    """
    Get the aqi_features feature group if it exists, or create it
    the first time this runs.
    """
    fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["city", "timestamp"],
        description="Hourly weather and pollutant features for AQI forecasting across Pakistani cities",
        online_enabled=False,
        time_travel_format="HUDI"
    )
    return fg

def insert_rows(rows):
    """
    Takes a list of feature dictionaries (like the ones engineer_features
    produces) and writes them into the Hopsworks feature group.
    """
    fs = get_feature_store()
    fg = get_or_create_feature_group(fs)
    df = pd.DataFrame(rows)
    fg.insert(df)
    print(f"Inserted {len(df)} rows into aqi_features.")