import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
import joblib
import os
import traceback
from feature_store import get_feature_store, get_model_registry

def load_clean_data():
    """
    Loads the final feature dataset from Hopsworks and drops any rows
    missing lag/rolling features or targets (unavoidable edge rows from
    the start/end of each city's history).
    """
    fs = get_feature_store()
    fg = fs.get_feature_group("aqi_features_final", version=1)

    print("Reading final dataset from Hopsworks...")
    df = fg.read()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["city", "timestamp"]).reset_index(drop=True)

    print(f"Loaded {len(df)} rows before cleaning")

    required_columns = [
        "pm2_5_lag_1h", "pm2_5_lag_3h", "pm2_5_lag_24h",
        "pm2_5_roll_3h", "pm2_5_roll_6h", "pm2_5_roll_24h",
        "target_aqi_24h", "target_aqi_48h", "target_aqi_72h"
    ]
    df_clean = df.dropna(subset=required_columns).reset_index(drop=True)

    print(f"Kept {len(df_clean)} rows after cleaning (dropped {len(df) - len(df_clean)})")
    return df_clean


def time_based_split(df, train_fraction=0.8):
    """
    Splits data by time, per city, so each city contributes its
    earliest 80% to training and most recent 20% to testing.
    Never shuffle time-series data randomly — that leaks future
    information into training.
    """
    train_parts, test_parts = [], []
    for city, city_df in df.groupby("city"):
        city_df = city_df.sort_values("timestamp").reset_index(drop=True)
        split_idx = int(len(city_df) * train_fraction)
        train_parts.append(city_df.iloc[:split_idx])
        test_parts.append(city_df.iloc[split_idx:])
    return (
        pd.concat(train_parts).reset_index(drop=True),
        pd.concat(test_parts).reset_index(drop=True),
    )


# The columns the models are allowed to learn from
FEATURE_COLUMNS = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month",
    "temp", "humidity", "wind_speed", "wind_sin", "wind_cos",
    "co", "no2", "o3", "so2", "pm2_5", "pm10",
    "pm2_5_lag_1h", "pm2_5_lag_3h", "pm2_5_lag_24h",
    "pm2_5_roll_3h", "pm2_5_roll_6h", "pm2_5_roll_24h",
    "aqi",
]


def evaluate_baseline(test_df, target_col):
    """
    Naive baseline: predicted future AQI = current AQI (no change).
    Any real model needs to beat this to prove it's learning something.
    """
    y_true = test_df[target_col]
    y_pred = test_df["aqi"]

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


def train_ridge(train_df, test_df, target_col):
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[target_col]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[target_col]

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model, rmse, mae, r2


def train_random_forest(train_df, test_df, target_col):
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[target_col]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[target_col]

    model = RandomForestRegressor(n_estimators=200, max_depth=6, min_samples_leaf=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model, rmse, mae, r2


def train_neural_network(train_df, test_df, target_col):
    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df[target_col].values
    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df[target_col].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = keras.Sequential([
        keras.layers.Input(shape=(X_train_scaled.shape[1],)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    model.fit(
        X_train_scaled, y_train,
        epochs=200, batch_size=32,
        validation_split=0.1, verbose=0,
        callbacks=[early_stop]
    )

    y_pred = model.predict(X_test_scaled, verbose=0).flatten()
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Neural net needs its scaler saved alongside it, since predict.py
    # will need to apply the same scaling at inference time
    return model, rmse, mae, r2, scaler


def register_model(model_path, model_type, target, rmse, mae, r2):
    mr = get_model_registry()

    model_name = f"aqi_{model_type}_{target}"
    metrics = {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}

    hw_model = mr.python.create_model(
        name=model_name,
        metrics=metrics,
        description=f"{model_type.replace('_', ' ').title()} model for {target}"
    )
    hw_model.save(model_path)
    print(f"Registered {model_name} to Hopsworks Model Registry (RMSE={rmse:.2f})")


if __name__ == "__main__":
    df = load_clean_data()
    train_df, test_df = time_based_split(df)

    print(f"\nTrain: {len(train_df)} rows, Test: {len(test_df)} rows\n")

    horizons = ["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]
    os.makedirs("models", exist_ok=True)

    for target in horizons:
        print(f"--- {target} ---")

        base_rmse, base_mae, base_r2 = evaluate_baseline(test_df, target)
        print(f"Baseline      -> RMSE: {base_rmse:.2f}  MAE: {base_mae:.2f}  R2: {base_r2:.3f}")

        ridge_model, ridge_rmse, ridge_mae, ridge_r2 = train_ridge(train_df, test_df, target)
        print(f"Ridge         -> RMSE: {ridge_rmse:.2f}  MAE: {ridge_mae:.2f}  R2: {ridge_r2:.3f}")

        rf_model, rf_rmse, rf_mae, rf_r2 = train_random_forest(train_df, test_df, target)
        print(f"Random Forest -> RMSE: {rf_rmse:.2f}  MAE: {rf_mae:.2f}  R2: {rf_r2:.3f}")

        nn_model, nn_rmse, nn_mae, nn_r2, nn_scaler = train_neural_network(train_df, test_df, target)
        print(f"Neural Net    -> RMSE: {nn_rmse:.2f}  MAE: {nn_mae:.2f}  R2: {nn_r2:.3f}")

        # Actually compare all three and pick the real winner by RMSE
        candidates = [
            ("ridge", ridge_model, ridge_rmse, ridge_mae, ridge_r2),
            ("random_forest", rf_model, rf_rmse, rf_mae, rf_r2),
            ("neural_net", nn_model, nn_rmse, nn_mae, nn_r2),
        ]
        winner_type, winner_model, winner_rmse, winner_mae, winner_r2 = min(
            candidates, key=lambda c: c[2]
        )

        print(f"Winner: {winner_type} (RMSE: {winner_rmse:.2f})")

        # Save the winning model to disk, handling Keras separately from sklearn,
        # since joblib doesn't reliably serialize Keras models
        if winner_type == "neural_net":
            model_path = f"models/{winner_type}_{target}.keras"
            winner_model.save(model_path)

            # Save the scaler too, predict.py will need it to transform
            # incoming features before calling this model
            scaler_path = f"models/{winner_type}_{target}_scaler.pkl"
            joblib.dump(nn_scaler, scaler_path)
            print(f"Saved scaler to {scaler_path}")
        else:
            model_path = f"models/{winner_type}_{target}.pkl"
            joblib.dump(winner_model, model_path)
            try:
                register_model(model_path, winner_type, target, winner_rmse, winner_mae, winner_r2)
            except Exception:
                print(f"!!! register_model FAILED for {target} !!!")
                traceback.print_exc()
                print()