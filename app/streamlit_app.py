import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from config import get_cities
from feature_store import get_feature_store, get_model_registry
from predict import predict_city_with_features, load_model, get_model_metrics, FEATURE_COLUMNS, HORIZONS
from explain import explain_prediction

st.set_page_config(page_title="Pakistan AQI Forecast", page_icon="🌫️", layout="wide")

# ---------------------------------------------------------------------------
# Styling. Streamlit's default look is functional but plain, this injects a
# dark, card-based theme via CSS instead, closer to a hand-built dashboard
# than the out-of-the-box widget styling.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }

    .hero-subtitle {
        color: #9aa4b2;
        font-size: 1rem;
        margin-bottom: 1.6rem;
    }

    .card {
        background-color: #151a23;
        border: 1px solid #232a36;
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
    }

    .stat-label {
        color: #9aa4b2;
        font-size: 0.78rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.3rem;
    }

    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
    }

    .aqi-hero {
        border-radius: 16px;
        padding: 1.8rem 2rem;
        text-align: center;
        margin-bottom: 1.2rem;
    }

    .aqi-hero-value {
        font-size: 3.2rem;
        font-weight: 800;
        line-height: 1;
        margin-bottom: 0.3rem;
    }

    .aqi-hero-label {
        font-size: 1.1rem;
        font-weight: 600;
        opacity: 0.9;
    }

    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin: 1.6rem 0 0.4rem 0;
    }

    .section-caption {
        color: #9aa4b2;
        font-size: 0.88rem;
        margin-bottom: 0.9rem;
    }

    div[data-testid="stSelectbox"] label {
        font-weight: 600;
        color: #e6e6e6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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


def stat_card(label, value, unit=""):
    st.markdown(
        f"""
        <div class="card">
            <div class="stat-label">{label}</div>
            <div class="stat-value">{value}{unit}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def styled_plotly(fig, height=400):
    """Applies the dark card theme to a Plotly figure so charts match the
    rest of the dashboard instead of defaulting to a white background."""
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#e6e6e6"),
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(gridcolor="#232a36", zerolinecolor="#232a36"),
        yaxis=dict(gridcolor="#232a36", zerolinecolor="#232a36"),
    )
    return fig


def health_guidance(aqi):
    if aqi <= 50:
        return "Air quality is good. No precautions needed for outdoor activity."
    elif aqi <= 100:
        return "Air quality is acceptable. Unusually sensitive individuals should consider limiting prolonged outdoor exertion."
    elif aqi <= 150:
        return "Sensitive groups (children, elderly, people with respiratory conditions) should limit prolonged outdoor exertion."
    elif aqi <= 200:
        return "Everyone may begin to experience health effects. Sensitive groups should avoid prolonged outdoor exertion."
    elif aqi <= 300:
        return "Health alert: everyone may experience more serious health effects. Avoid outdoor exertion."
    else:
        return "Health emergency: the entire population is likely to be affected. Avoid all outdoor activity."


@st.cache_resource
def load_feature_store():
    return get_feature_store()


@st.cache_resource
def load_model_registry():
    return get_model_registry()


fs = load_feature_store()
mr = load_model_registry()

st.markdown('<div class="hero-title">🌫️ Pakistan AQI Forecast</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">3-day air quality forecast using live weather and pollutant data</div>',
    unsafe_allow_html=True,
)

cities = get_cities()
city_names = [c["name"] for c in cities]
selected_name = st.selectbox("Select a city", city_names)
selected_city = next(c for c in cities if c["name"] == selected_name)

with st.spinner(f"Fetching live data and forecasting for {selected_name}..."):
    result = predict_city_with_features(fs, mr, selected_city)

if result is None:
    st.error(
        f"Not enough recent history for {selected_name} yet. "
        "The feature pipeline needs at least 24 hours of collected data "
        "before forecasts are possible."
    )
else:
    forecast, X_row, history_df = result
    current_color, current_label = aqi_color_and_label(forecast["current_aqi"])

    # --- Hero AQI card ---
    st.markdown(
        f"""
        <div class="aqi-hero" style="background-color:{current_color}22;
             border: 1px solid {current_color}55;">
            <div class="aqi-hero-value" style="color:{current_color};">
                {forecast['current_aqi']}
            </div>
            <div class="aqi-hero-label">{current_label} · Current AQI in {selected_name}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    max_forecast = max(forecast[h] for h in HORIZONS)
    if max_forecast > 150:
        st.warning(
            "⚠️ Hazard alert: AQI is expected to reach unhealthy levels "
            "within the next 3 days."
        )

    # --- Pollutant levels, all six raw pollutant readings, not just PM2.5/PM10 ---
    st.markdown('<div class="section-title">Current Pollutant Levels</div>', unsafe_allow_html=True)
    row = X_row.iloc[0]
    p1, p2, p3 = st.columns(3)
    with p1:
        stat_card("PM2.5", f"{row['pm2_5']:.0f}", " µg/m³")
        stat_card("CO", f"{row['co']:.0f}", " µg/m³")
    with p2:
        stat_card("PM10", f"{row['pm10']:.0f}", " µg/m³")
        stat_card("NO₂", f"{row['no2']:.1f}", " µg/m³")
    with p3:
        stat_card("O₃", f"{row['o3']:.1f}", " µg/m³")
        stat_card("SO₂", f"{row['so2']:.1f}", " µg/m³")

    # --- Current weather conditions used as model inputs ---
    st.markdown('<div class="section-title">Current Conditions</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        stat_card("Temperature", f"{row['temp']:.0f}", " °C")
    with c2:
        stat_card("Humidity", f"{row['humidity']:.0f}", "%")
    with c3:
        stat_card("Wind speed", f"{row['wind_speed']:.1f}", " m/s")

    # --- Health guidance, standard EPA-style AQI category guidance ---
    st.markdown(
        f"""
        <div class="card" style="border-left: 4px solid {current_color};">
            <div class="stat-label">Health Guidance</div>
            <div style="font-size:0.95rem; line-height:1.5;">{health_guidance(forecast['current_aqi'])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Model performance, real metrics stored by train.py at registration time ---
    st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Accuracy of the currently deployed model for each forecast horizon, measured on held-out historical data.</div>',
        unsafe_allow_html=True,
    )
    m1, m2, m3 = st.columns(3)
    for col, horizon, label in zip([m1, m2, m3], HORIZONS, ["24h", "48h", "72h"]):
        with col:
            metrics = get_model_metrics(mr, horizon)
            if metrics:
                st.markdown(
                    f"""
                    <div class="card">
                        <div class="stat-label">{label} forecast · {metrics['model_type'].replace('_', ' ').title()}</div>
                        <div class="stat-value">RMSE {metrics['rmse']:.2f}</div>
                        <div style="color:#9aa4b2; font-size:0.82rem; margin-top:0.2rem;">
                            MAE {metrics['mae']:.2f} · R² {metrics['r2']:.2f}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                stat_card(f"{label} forecast", "N/A")

    # --- Forecast chart ---
    st.markdown('<div class="section-title">3-Day Forecast</div>', unsafe_allow_html=True)
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
            line=dict(width=3, color="#00c04b"),
            marker=dict(size=10, color="#00c04b"),
            fill="tozeroy",
            fillcolor="rgba(0,192,75,0.08)",
        )
    )
    fig.update_layout(title=f"{selected_name} AQI Forecast", yaxis_title="AQI", xaxis_title="Time Horizon")
    st.plotly_chart(styled_plotly(fig), use_container_width=True)

    # --- SHAP explanations ---
    st.markdown('<div class="section-title">What\'s driving each forecast?</div>', unsafe_allow_html=True)
    st.markdown(
        """<div class="section-caption">SHAP values show how much each feature pushed a forecast up or
        down, relative to this city's recent typical conditions. Each horizon uses its own model,
        so the driving features can differ.</div>""",
        unsafe_allow_html=True,
    )

    horizon_tab_labels = {"target_aqi_24h": "24h", "target_aqi_48h": "48h", "target_aqi_72h": "72h"}
    tabs = st.tabs([horizon_tab_labels[h] for h in HORIZONS])

    for tab, horizon in zip(tabs, HORIZONS):
        with tab:
            model = load_model(mr, horizon)
            contributions = explain_prediction(model, X_row, history_df, FEATURE_COLUMNS)
            top_contributions = contributions.head(10)
            shap_fig = go.Figure(
                go.Bar(
                    x=top_contributions.values,
                    y=top_contributions.index,
                    orientation="h",
                    marker_color=[
                        "#ff4d4d" if v > 0 else "#00c04b" for v in top_contributions.values
                    ],
                )
            )
            shap_fig.update_layout(
                title=f"Top feature contributions ({horizon_tab_labels[horizon]} forecast)",
                xaxis_title="Impact on predicted AQI",
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(styled_plotly(shap_fig), use_container_width=True, key=f"shap_{horizon}")
            st.caption("Red bars pushed the forecast higher, green bars pulled it lower.")