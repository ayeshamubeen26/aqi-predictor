import os
import requests
import joblib
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from config import get_cities
from feature_engineering import engineer_features
from calculate_aqi import calculate_aqi
from feature_store import get_feature_store, get_model_registry

load_dotenv()

OPENWEATHER_KEY = os.getenv("OPENWEATHER_KEY")

HORIZONS = ["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]

# train.py only registers the winning model type per horizon, under a
# name like aqi_{model_type}_{target}. Since we don't know ahead of time
# which type won, load_model() tries each candidate name and uses
# whichever one actually exists in the registry, instead of a hardcoded
# guess that goes stale the next time the winner changes.
CANDIDATE_MODEL_TYPES = ["random_forest", "ridge", "neural_net"]

FEATURE_COLUMNS = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month",
    "temp", "humidity", "wind_speed", "wind_sin", "wind_cos",
    "co", "no2", "o3", "so2", "pm2_5", "pm10",
    "pm2_5_lag_1h", "pm2_5_lag_3h", "pm2_5_lag_24h",
    "pm2_5_roll_3h", "pm2_5_roll_6h", "pm2_5_roll_24h",
    "aqi",
]

_model_cache = {}


class NeuralNetModel:
    """
    Wraps a loaded Keras model together with the scaler it was trained
    with, so callers elsewhere (predict_city_with_features, explain.py)
    can call .predict(X) on raw feature values exactly like they do for
    the sklearn models, without needing to know scaling happened at all.
    """
    def __init__(self, keras_model, scaler):
        self.keras_model = keras_model
        self.scaler = scaler

    def predict(self, X):
        X_scaled = self.scaler.transform(np.asarray(X))
        return self.keras_model.predict(X_scaled, verbose=0).flatten()


def fetch_live_row(city):
    """
    Calls OpenWeather's current weather and air pollution endpoints for a
    city, then engineers the same raw feature row used during training,
    plus the calculated AQI.
    """
    lat, lon = city["lat"], city["lon"]

    weather_url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?lat={lat}&lon={lon}&appid={OPENWEATHER_KEY}&units=metric"
    )
    air_url = (
        f"https://api.openweathermap.org/data/2.5/air_pollution"
        f"?lat={lat}&lon={lon}&appid={OPENWEATHER_KEY}"
    )

    weather_json = requests.get(weather_url).json()
    air_json = requests.get(air_url).json()

    row = engineer_features(city["name"], weather_json, air_json)
    row["aqi"] = calculate_aqi(row["pm2_5"], row["pm10"])
    return row


def get_recent_history(fs, city_name, hours=25):
    """
    Pulls the most recent rows for a city from the feature store, enough
    to compute 24h lag and rolling features once the fresh row is added.

    Filters by city server-side via the Hopsworks query API, instead of
    reading the entire feature group (every city, entire training
    window) and filtering locally. Pulling the whole table on every
    single prediction is what strains the connection, especially over
    a home network rather than a CI runner's network path, and gets
    slower every day as the feature store keeps growing.
    """
    fg = fs.get_feature_group("aqi_features_final", version=1)
    query = fg.select_all().filter(fg.city == city_name)
    df = query.read()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    return df.tail(hours).reset_index(drop=True)


def add_lag_and_rolling(history_df, live_row):
    """
    Appends the fresh live row to recent history, then computes the same
    lag and rolling pm2_5 features used in training, taking the values
    for the newest (live) row.
    """
    combined = pd.concat(
        [history_df, pd.DataFrame([live_row])], ignore_index=True
    )
    combined["timestamp"] = pd.to_datetime(combined["timestamp"])
    combined = combined.sort_values("timestamp").reset_index(drop=True)

    combined["pm2_5_lag_1h"] = combined["pm2_5"].shift(1)
    combined["pm2_5_lag_3h"] = combined["pm2_5"].shift(3)
    combined["pm2_5_lag_24h"] = combined["pm2_5"].shift(24)

    combined["pm2_5_roll_3h"] = combined["pm2_5"].rolling(3).mean()
    combined["pm2_5_roll_6h"] = combined["pm2_5"].rolling(6).mean()
    combined["pm2_5_roll_24h"] = combined["pm2_5"].rolling(24).mean()

    return combined.iloc[-1]


def load_model(mr, target):
    """
    Downloads and caches the registered winning model for a given horizon
    from the Hopsworks Model Registry. `mr` is passed in rather than
    created here, so Streamlit can supply a single cached connection
    instead of this function logging in separately for every horizon.

    Tries each candidate model type in turn, since only the winning type
    for a given horizon is actually registered. This means predict.py
    keeps working automatically after a retrain changes the winner,
    instead of needing a manual update every time.
    """
    if target in _model_cache:
        return _model_cache[target]

    hw_model = None
    model_type = None
    for candidate_type in CANDIDATE_MODEL_TYPES:
        model_name = f"aqi_{candidate_type}_{target}"
        try:
            hw_model = mr.get_best_model(model_name, metric="rmse", direction="min")
            model_type = candidate_type
            break
        except Exception:
            continue

    if hw_model is None:
        raise RuntimeError(
            f"No registered model found for {target} under any of "
            f"{CANDIDATE_MODEL_TYPES}. Check that train.py registered "
            f"a winner for this horizon."
        )

    model_dir = hw_model.download()

    if model_type == "neural_net":
        # train.py saves the keras model and its scaler side by side in
        # one folder and registers that folder, so both come down
        # together in model_dir.
        import keras as keras_lib
        keras_path = os.path.join(model_dir, f"{model_type}_{target}.keras")
        scaler_path = os.path.join(model_dir, f"{model_type}_{target}_scaler.pkl")
        keras_model = keras_lib.models.load_model(keras_path)
        scaler = joblib.load(scaler_path)
        model = NeuralNetModel(keras_model, scaler)

        _model_cache[target] = model
        return model

    model_path = os.path.join(model_dir, f"{model_type}_{target}.pkl")
    model = joblib.load(model_path)

    _model_cache[target] = model
    return model


def predict_city(fs, mr, city):
    """
    Builds the current feature row for one city and returns a
    24h/48h/72h AQI forecast.
    """
    result = predict_city_with_features(fs, mr, city)
    if result is None:
        return None
    forecast, _, _ = result
    return forecast


def predict_city_with_features(fs, mr, city):
    """
    Same as predict_city, but also returns the built feature row and the
    recent history used as a SHAP background reference, so the app can
    explain the prediction as well as show it.
    """
    live_row = fetch_live_row(city)
    history_df = get_recent_history(fs, city["name"])
    full_row = add_lag_and_rolling(history_df, live_row)

    missing = [c for c in FEATURE_COLUMNS if pd.isna(full_row.get(c))]
    if missing:
        print(f"Skipping {city['name']}: not enough history for {missing}")
        return None

    X = pd.DataFrame([full_row[FEATURE_COLUMNS]])

    forecast = {"city": city["name"], "current_aqi": full_row["aqi"]}
    for target in HORIZONS:
        model = load_model(mr, target)
        forecast[target] = round(float(model.predict(X)[0]), 1)

    return forecast, X, history_df


if __name__ == "__main__":
    fs = get_feature_store()
    mr = get_model_registry()
    cities = get_cities()

    results = []
    for city in cities:
        print(f"Predicting for {city['name']}...")
        forecast = predict_city(fs, mr, city)
        if forecast:
            results.append(forecast)
            print(
                f"  Current: {forecast['current_aqi']}  "
                f"24h: {forecast['target_aqi_24h']}  "
                f"48h: {forecast['target_aqi_48h']}  "
                f"72h: {forecast['target_aqi_72h']}"
            )

    df_results = pd.DataFrame(results)
    print("\nAll forecasts:")
    print(df_results)