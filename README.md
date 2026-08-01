# AQI Predictor

A serverless machine learning pipeline that forecasts Air Quality Index (AQI) 24, 48, and 72 hours ahead for five Pakistani cities. Weather and pollutant data is pulled hourly, engineered into a Hopsworks feature store, and used to train and compare multiple forecasting models per horizon. Live forecasts and SHAP explainability are served through a Streamlit dashboard.

Built for the Pearls AQI Predictor project brief: an end-to-end forecasting system with automated data collection, feature engineering, model training, and a real-time dashboard, using a 100% serverless stack.

## Cities covered

Karachi, Lahore, Islamabad, Faisalabad, Peshawar.

## Architecture

```
Weather & Pollutant APIs (OpenWeather)
        |
        v
Feature pipeline (src/data_fetch.py, feature_engineering.py)
        |
        v
Hopsworks Feature Store (aqi_features_final)
        |
        +---------------------------+
        v                           v
Training pipeline              Prediction (src/predict.py)
(src/train.py)                      |
        |                           v
        v                    Streamlit dashboard
Hopsworks Model Registry     (app/streamlit_app.py)
(Ridge, Random Forest,             |
 Neural Net, one winner            v
 registered per horizon)     SHAP explainability
                              (src/explain.py)
```

Both the feature pipeline and the training pipeline run on a schedule through GitHub Actions, not on any always-on server. The Streamlit app is the only component a person interacts with directly, and it reads everything it needs from the feature store and model registry at request time.

## Tech stack

- **Python** for the entire pipeline
- **OpenWeather API** for live weather and pollutant data
- **Hopsworks** for the feature store and model registry (free tier)
- **Scikit-learn** (Ridge, Random Forest) and **TensorFlow/Keras** (a small dense network with dropout and early stopping) for model training
- **GitHub Actions** for scheduling and running both pipelines
- **Streamlit** and **Plotly** for the dashboard
- **SHAP** for feature importance explanations

## Repository structure

```
aqi-predictor/
├── .github/workflows/
│   ├── feature_pipeline.yml     # runs hourly, fetches + stores new features
│   └── training_pipeline.yml    # runs daily, retrains + registers models
├── app/
│   └── streamlit_app.py         # dashboard: forecast chart + SHAP panel
├── data/
│   └── aqi_features_final*.csv  # exported snapshots of the feature store, for offline EDA
├── notebooks/
│   └── eda.ipynb                # exploratory analysis of the backfilled dataset
├── src/
│   ├── config.py                # city list, API keys, constants
│   ├── data_fetch.py            # pulls raw weather + pollutant data
│   ├── calculate_aqi.py         # EPA breakpoint AQI formula
│   ├── feature_engineering.py   # cyclical time features, per-city feature building
│   ├── add_rolling_features.py  # lag + rolling window features
│   ├── add_targets.py           # builds the 24h/48h/72h future AQI targets
│   ├── feature_store.py         # Hopsworks feature store + model registry helpers
│   ├── backfill.py              # runs the feature pipeline over a historical date range
│   ├── train.py                 # trains, evaluates, and registers the winning model per horizon
│   ├── predict.py               # loads the registered model, builds a live forecast
│   └── explain.py               # SHAP explanations, model-type aware
├── tests/
└── requirements.txt
```

## Setup

1. Clone the repo and install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and fill in:
   ```
   OPENWEATHER_KEY=
   HOPSWORKS_API_KEY=
   HOPSWORKS_PROJECT=
   CITIES=CityName:lat:lon,CityName2:lat2:lon2
   ```
3. For GitHub Actions automation, add the same values as repository secrets (Settings → Secrets and variables → Actions), so both workflows can authenticate without the `.env` file ever being committed.

## Pipeline details

### Feature pipeline
Runs hourly via `feature_pipeline.yml`. Fetches current weather and pollutant readings per city from OpenWeather, computes the AQI from PM2.5/PM10 using the EPA breakpoint formula, builds cyclical time features (hour, day of week, wind direction, encoded as sin/cos pairs rather than raw integers) and lag/rolling PM2.5 features (1h/3h/24h lags, 3h/6h/24h rolling means), and writes the result to the `aqi_features_final` feature group in Hopsworks.

### Backfill
`src/backfill.py` runs the same feature logic across a historical date range to build up training data. The current feature store holds roughly two years of hourly history across all five cities.

### Training pipeline
Runs daily via `training_pipeline.yml`. For each of the three horizons (24h, 48h, 72h):
- Loads the most recent 365 days of history from the feature store (configurable via `TRAIN_WINDOW_DAYS`), rather than the entire ever-growing history, so training time and Hopsworks read cost stay bounded as the feature store keeps growing.
- Splits by time, not randomly, using the earliest 80% of each city's rows for training and the most recent 20% for testing, to avoid leaking future data into training.
- Trains Ridge, Random Forest, and a small neural network, and evaluates all three against a persistence baseline (predict future AQI = current AQI) using RMSE, MAE, and R².
- Registers the winning model for that horizon to the Hopsworks Model Registry.

### Prediction and dashboard
`src/predict.py` pulls a city's recent history, fetches a live weather/pollutant reading, computes the same feature set used in training, and loads whichever model type actually won each horizon from the registry, rather than assuming a fixed model type. `app/streamlit_app.py` displays the resulting 3-day forecast as a line chart, flags hazardous AQI levels, and shows a SHAP feature-importance breakdown for each of the three horizons separately, since each horizon uses its own model and the driving features can differ.

## Model results

From the most recent training run, evaluated on the held-out (most recent 20%) test split:

| Horizon | Baseline RMSE | Winning model | Winner RMSE | Winner MAE | Winner R² |
|---|---|---|---|---|---|
| 24h | 28.45 | Random Forest | 24.90 | 17.31 | 0.555 |
| 48h | 33.09 | Random Forest | 28.34 | 21.28 | 0.417 |
| 72h | 35.60 | Random Forest | 30.79 | 23.66 | 0.310 |

Random Forest beat the persistence baseline at every horizon, which matters because the baseline is a genuinely hard bar to clear here: exploratory analysis (`notebooks/eda.ipynb`) shows current AQI correlates with AQI 72 hours out at r = 0.70, so "assume no change" is already a reasonably strong naive forecast in this domain. Error grows with horizon for every model, including the baseline, which is expected: air quality further out is inherently less predictable from current conditions alone.

SHAP explanations show the driving features shift with horizon. At 24h, the current PM2.5 reading dominates the prediction. At 72h, the 24-hour rolling average PM2.5 becomes co-dominant with the current reading, and single-moment weather features like wind speed matter less, the model leans more on smoothed recent history as the forecast window stretches out.

## Exploratory data analysis

See `notebooks/eda.ipynb` for the full analysis, run against the real backfilled dataset. Covers AQI distribution and category breakdown per city, seasonal and daily trends, correlation between weather/pollutants and AQI, and the persistence-correlation analysis referenced above.

## Known limitations and possible next steps

- The neural network never won any horizon in the runs so far, Random Forest consistently outperformed it. Worth revisiting with a different architecture or more training data before concluding tree-based models are simply the better fit for this problem.
- `tests/` currently has no real test coverage beyond a placeholder file.
- The repository has some leftover duplicate artifacts from earlier development (an older `src/models/` folder with early Ridge models, a duplicate `.env` file) that are safe to delete but haven't been cleaned up yet.
- Alerts for hazardous AQI levels are shown in the dashboard but aren't pushed anywhere (email, SMS), they're visible only to someone actively viewing the app.

## Security note

`.env` is gitignored and has never been committed to this repository. Automated runs authenticate through GitHub Actions repository secrets instead.
