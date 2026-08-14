"""
Keeps aqi_features_final up to date with what's actually landing in the
raw aqi_features table every hour.

The original pipeline (add_rolling_features.py -> add_targets.py) was
built as a one-time batch job: read the ENTIRE aqi_features table,
recompute rolling/lag features and targets for every row, and write the
whole thing to aqi_features_final. That's fine for a one-off backfill,
but running it every hour would mean re-reading and reprocessing two
years of history every single time, expensive and slow, and it isn't
what actually happened: data_fetch.py runs hourly, but nothing was ever
scheduled to promote those new rows into aqi_features_final at all.

This script does the same feature engineering (reusing the exact same
functions, not duplicating the logic) but only on a bounded recent
window, so it can run hourly cheaply while keeping aqi_features_final
genuinely current.

Why a 10-day window, not just "the last hour":
- Lag/rolling features for a brand-new row need up to 24h of prior
  history to compute correctly.
- Target columns (target_aqi_24h/48h/72h) can only be filled in once
  that much real time has actually passed, so a row from 3 days ago
  might only now be getting its target_aqi_72h filled in for the
  first time. Recomputing over a wider recent window is what lets that
  catch-up happen, a single-row insert never could.
- Hopsworks feature groups with a primary key upsert on insert, so
  re-inserting rows that already exist in aqi_features_final (to update
  their targets as they become available) safely overwrites them
  instead of creating duplicates.
"""
from datetime import datetime, timedelta
from feature_store import get_feature_store
from add_rolling_features import add_rolling_and_lag_features
from add_targets import add_aqi_and_targets

SYNC_WINDOW_DAYS = 10


def sync_final_features():
    fs = get_feature_store()
    fg = fs.get_feature_group("aqi_features", version=1)

    cutoff = (datetime.now() - timedelta(days=SYNC_WINDOW_DAYS)).strftime("%Y-%m-%d %H:%M:%S")
    print(f"Reading aqi_features rows since {cutoff}...")

    query = fg.select_all().filter(fg.timestamp >= cutoff)
    df = query.read()
    print(f"Loaded {len(df)} rows from the last {SYNC_WINDOW_DAYS} days")

    if df.empty:
        print("No recent rows to sync.")
        return

    df_enriched = add_rolling_and_lag_features(df)
    df_final = add_aqi_and_targets(df_enriched)

    final_fg = fs.get_or_create_feature_group(
        name="aqi_features_final",
        version=1,
        primary_key=["city", "timestamp"],
        description="Final AQI features with computed AQI and 24/48/72h targets, ready for model training",
        online_enabled=False,
        time_travel_format="HUDI",
        statistics_config={"enabled": False}
    )
    if final_fg.statistics_config.enabled:
        final_fg.statistics_config = {"enabled": False}
        final_fg.update_statistics_config()

    print(f"Syncing {len(df_final)} rows into aqi_features_final...")
    final_fg.insert(df_final, wait=True)
    print("Done.")


if __name__ == "__main__":
    sync_final_features()