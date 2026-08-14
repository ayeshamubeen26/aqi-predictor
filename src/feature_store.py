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

def get_model_registry():
    """Connect to Hopsworks and return the project's model registry handle."""
    project = hopsworks.login(
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
        project=os.getenv("HOPSWORKS_PROJECT")
    )
    return project.get_model_registry()

def get_or_create_feature_group(fs):
    """
    Get the aqi_features feature group if it exists, or create it
    the first time this runs.

    statistics_config disabled: Hopsworks normally recomputes summary
    statistics (histograms, correlations, etc.) after every write, which
    means a full-table scan every single insert. On this project's
    free-tier cluster that step has been timing out and failing the
    entire job even after the actual data write already succeeded, this
    project doesn't use those auto-computed statistics for anything, so
    there's no reason to pay that cost or risk that failure every hour.

    The statistics_config passed to get_or_create_feature_group only
    takes effect the first time a feature group is created, since this
    group already exists from earlier runs, that argument alone would
    be silently ignored. update_statistics_config() is what actually
    changes it on a feature group that already exists.
    """
    fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["city", "timestamp"],
        description="Hourly weather and pollutant features for AQI forecasting across Pakistani cities",
        online_enabled=False,
        time_travel_format="HUDI",
        statistics_config={"enabled": False}
    )
    if fg.statistics_config.enabled:
        fg.statistics_config = {"enabled": False}
        fg.update_statistics_config()
    return fg

def insert_rows(rows):
    """
    Takes a list of feature dictionaries (like the ones engineer_features
    produces) and writes them into the Hopsworks feature group.

    wait=True blocks until Hopsworks' background materialization job
    actually finishes committing the write, rather than returning as
    soon as the job is merely launched. Without this, insert() returns
    almost immediately while the real write is still in progress, and
    anything reading the table moments later (like sync_final_features.py,
    which runs as the very next step in the same workflow) can find zero
    rows even though the insert reported success, since the data hadn't
    actually landed yet.
    """
    fs = get_feature_store()
    fg = get_or_create_feature_group(fs)
    df = pd.DataFrame(rows)
    fg.insert(df, wait=True)
    print(f"Inserted {len(df)} rows into aqi_features.")