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
# Styling: light, card-based theme with soft shadows instead of borders,
# closer to a hand-built product dashboard than Streamlit's plain default.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/icon?family=Material+Symbols+Outlined');

    .material-symbols-outlined {
        font-family: 'Material Symbols Outlined';
        font-size: 20px;
        vertical-align: middle;
        color: #6b7280;
    }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .block-container { padding-top: 1.5rem; padding-bottom: 3rem; padding-left: 2.5rem; padding-right: 2.5rem; max-width: 100%; }

    .hero-title { font-size: 2.1rem; font-weight: 800; margin-bottom: 0.1rem; letter-spacing: -0.02em; color: #1a1f29; }
    .hero-subtitle { color: #6b7280; font-size: 0.95rem; margin-bottom: 1.4rem; }

    .card {
        background-color: #ffffff;
        border: 1px solid #eef0f3;
        border-radius: 16px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(16, 24, 40, 0.05);
    }

    .group-card {
        background-color: #ffffff;
        border: 1px solid #eef0f3;
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 1px 3px rgba(16, 24, 40, 0.05);
    }

    .inner-stat {
        background-color: #f8f9fb;
        border-radius: 12px;
        padding: 0.9rem 1.1rem;
    }

    .stat-label { color: #6b7280; font-size: 0.74rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 0.3rem; }
    .stat-value { font-size: 1.4rem; font-weight: 700; color: #1a1f29; }
    .stat-sub { color: #9aa4b2; font-size: 0.8rem; margin-top: 0.15rem; }

    .section-title { font-size: 1.05rem; font-weight: 700; margin: 1.5rem 0 0.5rem 0; color: #1a1f29; }
    .section-caption { color: #6b7280; font-size: 0.85rem; margin-bottom: 0.8rem; }

    .badge {
        display: inline-block;
        padding: 2px 11px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 700;
    }

    div[data-testid="stSelectbox"] label { font-weight: 600; color: #1a1f29; }

    /* Restyle Streamlit's native tabs as rounded pill buttons */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #f1f2f5;
        padding: 4px;
        border-radius: 999px;
        width: fit-content;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 999px;
        padding: 4px 18px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        box-shadow: 0 1px 3px rgba(16, 24, 40, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def aqi_color_and_label(aqi):
    if aqi <= 50:
        return "#00b050", "Good"
    elif aqi <= 100:
        return "#d4a600", "Moderate"
    elif aqi <= 150:
        return "#e07a00", "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "#e02424", "Unhealthy"
    elif aqi <= 300:
        return "#8f3f97", "Very Unhealthy"
    else:
        return "#7e0023", "Hazardous"


FEATURE_LABELS = {
    "hour_sin": "Hour of Day (sin)",
    "hour_cos": "Hour of Day (cos)",
    "dow_sin": "Day of Week (sin)",
    "dow_cos": "Day of Week (cos)",
    "month": "Month",
    "temp": "Temperature",
    "humidity": "Humidity",
    "wind_speed": "Wind Speed",
    "wind_sin": "Wind Direction (sin)",
    "wind_cos": "Wind Direction (cos)",
    "co": "Carbon Monoxide (CO)",
    "no2": "Nitrogen Dioxide (NO₂)",
    "o3": "Ozone (O₃)",
    "so2": "Sulfur Dioxide (SO₂)",
    "pm2_5": "PM2.5",
    "pm10": "PM10",
    "pm2_5_lag_1h": "PM2.5 (1h ago)",
    "pm2_5_lag_3h": "PM2.5 (3h ago)",
    "pm2_5_lag_24h": "PM2.5 (24h ago)",
    "pm2_5_roll_3h": "PM2.5 (3h avg)",
    "pm2_5_roll_6h": "PM2.5 (6h avg)",
    "pm2_5_roll_24h": "PM2.5 (24h avg)",
    "aqi": "Current AQI",
}


def readable(col):
    return FEATURE_LABELS.get(col, col)


def hex_to_rgba(hex_color, alpha):
    """Converts a #RRGGBB string to an rgba() string Plotly will accept.
    Plotly's color properties (fillcolor, gauge steps, etc.) reject the
    8-digit hex-with-alpha shorthand that's valid in CSS, this is why the
    same trick works fine inside the raw HTML/CSS blocks in this file but
    throws a ValueError when passed directly to a go.Scatter or
    go.Indicator property instead.
    """
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def badge_html(label, color):
    return f'<span class="badge" style="background-color:{color}1a; color:{color};">{label}</span>'


def stat_card(label, value, unit="", sub=None, icon=None):
    sub_html = f'<div class="stat-sub">{sub}</div>' if sub else ""
    icon_html = f'<span class="material-symbols-outlined">{icon}</span> ' if icon else ""
    st.markdown(
        f"""<div class="card"><div class="stat-label">{icon_html}{label}</div>
        <div class="stat-value">{value}{unit}</div>{sub_html}</div>""",
        unsafe_allow_html=True,
    )


def styled_plotly(fig, height=380):
    """Applies the light card theme to a Plotly figure."""
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#1a1f29"),
        margin=dict(l=10, r=10, t=45, b=10),
        xaxis=dict(gridcolor="#eef0f3", zerolinecolor="#eef0f3"),
        yaxis=dict(gridcolor="#eef0f3", zerolinecolor="#eef0f3"),
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

st.markdown('<div class="hero-title">🌫️ AQI Prediction</div>', unsafe_allow_html=True)
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
    row = X_row.iloc[0]
    current_aqi = forecast["current_aqi"]
    current_color, current_label = aqi_color_and_label(current_aqi)

    # --- Top row: gauge + alert/health card ---
    top_left, top_right = st.columns([1, 1])

    with top_left:
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=current_aqi,
                number={"font": {"size": 42, "color": current_color}},
                gauge={
                    "axis": {"range": [0, 300], "tickwidth": 1, "tickcolor": "#c9cfd8"},
                    "bar": {"color": current_color, "thickness": 0.28},
                    "bgcolor": "white",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 50], "color": "rgba(0,176,80,0.13)"},
                        {"range": [50, 100], "color": "rgba(212,166,0,0.13)"},
                        {"range": [100, 150], "color": "rgba(224,122,0,0.13)"},
                        {"range": [150, 200], "color": "rgba(224,36,36,0.13)"},
                        {"range": [200, 300], "color": "rgba(143,63,151,0.13)"},
                    ],
                },
                title={"text": f"{selected_name} · Current AQI", "font": {"size": 14, "color": "#6b7280"}},
            )
        )
        st.plotly_chart(styled_plotly(gauge, height=260), use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            f'<div style="text-align:center; margin-top:-1rem;">{badge_html(current_label, current_color)}</div>',
            unsafe_allow_html=True,
        )

    with top_right:
        max_forecast = max(forecast[h] for h in HORIZONS)
        if max_forecast > 150:
            st.markdown(
                f"""<div class="card" style="border-left: 4px solid #e02424;">
                <div class="stat-label" style="color:#e02424;">⚠️ Air Quality Alert</div>
                <div style="font-size:0.9rem; color:#374151; margin-top:0.3rem;">
                Air quality is expected to reach unhealthy levels within the next 3 days.
                </div></div>""",
                unsafe_allow_html=True,
            )
        st.markdown(
            f"""<div class="card" style="border-left: 4px solid {current_color};">
            <div class="stat-label">Health Guidance</div>
            <div style="font-size:0.9rem; color:#374151; margin-top:0.3rem; line-height:1.5;">
            {health_guidance(current_aqi)}
            </div></div>""",
            unsafe_allow_html=True,
        )

    # --- Current pollutants ---
    st.markdown('<div class="section-title">Current Pollutants</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-caption">Live pollutant concentrations in {selected_name}</div>', unsafe_allow_html=True)
    pollutants = [
        ("PM2.5", row["pm2_5"], " µg/m³", 0, "blur_on"),
        ("PM10", row["pm10"], " µg/m³", 0, "grain"),
        ("O₃", row["o3"], " µg/m³", 1, "air"),
        ("NO₂", row["no2"], " µg/m³", 1, "science"),
        ("SO₂", row["so2"], " µg/m³", 1, "science"),
        ("CO", row["co"], " µg/m³", 0, "cloud"),
    ]
    pcols = st.columns(6)
    for col, (label, value, unit, dp, icon) in zip(pcols, pollutants):
        with col:
            stat_card(label, f"{value:.{dp}f}", unit, icon=icon)

    # --- 24h trend + current conditions ---
    trend_col, cond_col = st.columns([2, 1])

    with trend_col:
        st.markdown('<div class="section-title">24-Hour AQI Trend</div>', unsafe_allow_html=True)
        hist = history_df.copy()
        hist["timestamp"] = pd.to_datetime(hist["timestamp"])
        trend_fig = go.Figure(
            go.Scatter(
                x=hist["timestamp"],
                y=hist["aqi"],
                mode="lines",
                line=dict(color=current_color, width=2.5),
                fill="tozeroy",
                fillcolor=hex_to_rgba(current_color, 0.1),
            )
        )
        st.plotly_chart(styled_plotly(trend_fig, height=280), use_container_width=True, config={"displayModeBar": False})
        h1, h2, h3, h4 = st.columns(4)
        h1.markdown(f'<div class="stat-label">Current</div><div class="stat-value" style="font-size:1.1rem;">{hist["aqi"].iloc[-1]:.0f}</div>', unsafe_allow_html=True)
        h2.markdown(f'<div class="stat-label">Average</div><div class="stat-value" style="font-size:1.1rem;">{hist["aqi"].mean():.0f}</div>', unsafe_allow_html=True)
        h3.markdown(f'<div class="stat-label">Min</div><div class="stat-value" style="font-size:1.1rem;">{hist["aqi"].min():.0f}</div>', unsafe_allow_html=True)
        h4.markdown(f'<div class="stat-label">Max</div><div class="stat-value" style="font-size:1.1rem;">{hist["aqi"].max():.0f}</div>', unsafe_allow_html=True)

    with cond_col:
        st.markdown('<div class="section-title">Current Conditions</div>', unsafe_allow_html=True)
        stat_card("Temperature", f"{row['temp']:.1f}", " °C", icon="thermostat")
        stat_card("Humidity", f"{row['humidity']:.0f}", "%", icon="water_drop")
        stat_card("Wind speed", f"{row['wind_speed']:.1f}", " m/s", icon="air")

    # --- Forecast day cards, each with its own live RMSE badge ---
    st.markdown('<div class="section-title">AI Air Quality Forecast</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Predicted AQI for the next three days</div>', unsafe_allow_html=True)

    day_cols = st.columns(3)
    day_labels = [("24h", "Day 1"), ("48h", "Day 2"), ("72h", "Day 3")]
    for col, horizon, (h_label, d_label) in zip(day_cols, HORIZONS, day_labels):
        with col:
            value = forecast[horizon]
            f_color, f_label = aqi_color_and_label(value)
            metrics = get_model_metrics(mr, horizon)
            rmse_text = f"± {metrics['rmse']:.2f}" if metrics else "N/A"
            st.markdown(
                f"""<div class="card" style="border-top: 3px solid {f_color};">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span class="stat-label">{h_label} · {d_label}</span>
                    {badge_html(f_label, f_color)}
                </div>
                <div class="stat-value" style="font-size:1.7rem; color:{f_color}; margin-top:0.4rem;">{value:.1f}</div>
                <div class="stat-sub">predicted AQI</div>
                <div class="stat-sub" style="margin-top:0.5rem;">Model RMSE {rmse_text}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    # --- Forecast trend chart ---
    st.markdown('<div class="section-title">Predicted AQI Trend</div>', unsafe_allow_html=True)
    horizon_labels = ["Now", "24h", "48h", "72h"]
    horizon_values = [current_aqi, forecast["target_aqi_24h"], forecast["target_aqi_48h"], forecast["target_aqi_72h"]]
    trend2 = go.Figure(
        go.Scatter(
            x=horizon_labels, y=horizon_values, mode="lines+markers",
            line=dict(width=3, color="#2563eb"), marker=dict(size=9, color="#2563eb"),
            fill="tozeroy", fillcolor="rgba(37,99,235,0.08)",
        )
    )
    st.plotly_chart(styled_plotly(trend2, height=320), use_container_width=True, config={"displayModeBar": False})

    # --- Prediction system summary ---
    st.markdown('<div class="section-title">Prediction System</div>', unsafe_allow_html=True)
    sys_cols = st.columns(3)
    for col, horizon, label in zip(sys_cols, HORIZONS, ["24h", "48h", "72h"]):
        with col:
            metrics = get_model_metrics(mr, horizon)
            if metrics:
                st.markdown(
                    f"""<div class="card">
                    <div class="stat-label">{label} model</div>
                    <div class="stat-value" style="font-size:1.15rem;">{metrics['model_type'].replace('_', ' ').title()}</div>
                    <div class="stat-sub">RMSE {metrics['rmse']:.2f} · MAE {metrics['mae']:.2f} · R² {metrics['r2']:.2f}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            else:
                stat_card(f"{label} model", "N/A")

    # --- SHAP explanations, all wrapped in one grouped container ---
    shap_container = st.container(border=True)
    with shap_container:
        st.markdown('<div class="section-title" style="margin-top:0;">Why this prediction</div>', unsafe_allow_html=True)
        st.markdown(
            """<div class="section-caption">SHAP feature contributions for each forecast horizon, relative to this
            city's recent typical conditions.</div>""",
            unsafe_allow_html=True,
        )

        horizon_tab_labels = {"target_aqi_24h": "24h", "target_aqi_48h": "48h", "target_aqi_72h": "72h"}
        tabs = st.tabs([horizon_tab_labels[h] for h in HORIZONS])

        for tab, horizon in zip(tabs, HORIZONS):
            with tab:
                model = load_model(mr, horizon)
                contributions = explain_prediction(model, X_row, history_df, FEATURE_COLUMNS)
                top_increase = contributions[contributions > 0].head(1)
                top_decrease = contributions[contributions < 0].sort_values().head(1)

                s1, s2, s3 = st.columns(3)
                with s1:
                    st.markdown(
                        f"""<div class="inner-stat"><div class="stat-label">Predicted AQI</div>
                        <div class="stat-value">{forecast[horizon]:.1f}</div></div>""",
                        unsafe_allow_html=True,
                    )
                with s2:
                    if len(top_increase):
                        st.markdown(
                            f"""<div class="inner-stat"><div class="stat-label">Top increase</div>
                            <div class="stat-value" style="font-size:1.05rem;">{readable(top_increase.index[0])}</div>
                            <div class="stat-sub">+{top_increase.values[0]:.2f}</div></div>""",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown('<div class="inner-stat"><div class="stat-label">Top increase</div><div class="stat-value" style="font-size:1.05rem;">None</div></div>', unsafe_allow_html=True)
                with s3:
                    if len(top_decrease):
                        st.markdown(
                            f"""<div class="inner-stat"><div class="stat-label">Top decrease</div>
                            <div class="stat-value" style="font-size:1.05rem;">{readable(top_decrease.index[0])}</div>
                            <div class="stat-sub">{top_decrease.values[0]:.2f}</div></div>""",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown('<div class="inner-stat"><div class="stat-label">Top decrease</div><div class="stat-value" style="font-size:1.05rem;">None</div></div>', unsafe_allow_html=True)

                st.write("")
                top_contributions = contributions.head(10)
                readable_labels = [readable(c) for c in top_contributions.index]
                shap_fig = go.Figure(
                    go.Bar(
                        x=top_contributions.values,
                        y=readable_labels,
                        orientation="h",
                        marker_color=["#e02424" if v > 0 else "#00b050" for v in top_contributions.values],
                    )
                )
                shap_fig.update_layout(
                    title=f"Top feature contributions ({horizon_tab_labels[horizon]} forecast)",
                    xaxis_title="Impact on predicted AQI",
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(styled_plotly(shap_fig), use_container_width=True, key=f"shap_{horizon}", config={"displayModeBar": False})
                st.caption("Red bars pushed the forecast higher, green bars pulled it lower.")