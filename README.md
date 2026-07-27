# aqi-predictor
Serverless ML pipeline that forecasts 24, 48, and 72-hour Air Quality Index for a city. Pulls weather and pollutant data hourly, engineers features into a Hopsworks feature store, and trains Random Forest and TensorFlow models to predict AQI trends. Live forecasts and SHAP explainability are served through a Streamlit dashboard.
