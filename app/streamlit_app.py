import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from config import get_cities
from feature_store import get_feature_store
from predict import predict_city_with_features, load_model, FEATURE_COLUMNS, HORIZONS
from explain import explain_prediction

st.set_page_config(page_title="Pakistan AQI Forecast", page_icon="🌫️", layout="centered")

st.title("Pakistan AQI Forecast")
st.caption("3-day air quality forecast using live weather and pollutant data")


def aqi_color_and_label(aqi):
    if aqi <= 50:
        return "#00e400", "Good"
    elif aqi <= 100:
        return "#ffff00", "Moderate"
    elif aqi <= 150:
        return "#ff7e00", "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "#ff0000", "Unhealthy"
    elif aqi <= 300:
        return "#8f3f97", "Very Unhealthy"
    else:
        return "#7e0023", "Hazardous"


@st.cache_resource
def load_feature_store():
    return get_feature_store()


fs = load_feature_store()
cities = get_cities()
city_names = [c["name"] for c in cities]

selected_name = st.selectbox("Select a city", city_names)
selected_city = next(c for c in cities if c["name"] == selected_name)

with st.spinner(f"Fetching live data and forecasting for {selected_name}..."):
    result = predict_city_with_features(fs, selected_city)

if result is None:
    st.error(
        f"Not enough recent history for {selected_name} yet. "
        "The feature pipeline needs at least 24 hours of collected data "
        "before forecasts are possible."
    )
else:
    forecast, X_row, history_df = result

    current_color, current_label = aqi_color_and_label(forecast["current_aqi"])
    st.markdown(
        f"<div style='padding:14px;border-radius:8px;background-color:{current_color};"
        f"color:black;text-align:center;font-weight:bold;font-size:18px;'>"
        f"Current AQI: {forecast['current_aqi']} — {current_label}</div>",
        unsafe_allow_html=True,
    )

    max_forecast = max(forecast[h] for h in HORIZONS)
    if max_forecast > 150:
        st.warning(
            "Hazard alert: AQI is expected to reach unhealthy levels "
            "within the next 3 days."
        )

    horizon_labels = ["Now", "24h", "48h", "72h"]
    horizon_values = [
        forecast["current_aqi"],
        forecast["target_aqi_24h"],
        forecast["target_aqi_48h"],
        forecast["target_aqi_72h"],
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=horizon_labels,
            y=horizon_values,
            mode="lines+markers",
            line=dict(width=3),
            marker=dict(size=10),
        )
    )
    fig.update_layout(
        title=f"{selected_name} AQI Forecast",
        yaxis_title="AQI",
        xaxis_title="Time Horizon",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("What's driving the 24h prediction?")
    st.caption(
        "SHAP values show how much each feature pushed the 24h forecast "
        "up or down, relative to this city's recent typical conditions."
    )

    model_24h = load_model("target_aqi_24h")
    contributions = explain_prediction(model_24h, X_row, history_df, FEATURE_COLUMNS)

    top_contributions = contributions.head(10)
    shap_fig = go.Figure(
        go.Bar(
            x=top_contributions.values,
            y=top_contributions.index,
            orientation="h",
            marker_color=[
                "#ff0000" if v > 0 else "#00c04b" for v in top_contributions.values
            ],
        )
    )
    shap_fig.update_layout(
        title="Top feature contributions (24h forecast)",
        xaxis_title="Impact on predicted AQI",
        yaxis=dict(autorange="reversed"),
        height=400,
    )
    st.plotly_chart(shap_fig, use_container_width=True)

    st.caption(
        "Red bars pushed the forecast higher, green bars pulled it lower."
    )