import sys
import os
from datetime import datetime
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from config import get_cities
from feature_store import get_feature_store, get_model_registry
from predict import predict_city_with_features, load_model, get_model_metrics, get_recent_history, FEATURE_COLUMNS, HORIZONS
from explain import explain_prediction

st.set_page_config(page_title="Pakistan AQI Forecast", page_icon="🌫️", layout="wide")

# ---------------------------------------------------------------------------
# Icon system: small inline SVGs instead of an external icon font. A webfont
# ligature (e.g. "shield") that fails to load or render falls back to raw
# text in the browser's default font/size, which is what produced the
# oversized, wrongly-colored, overflowing glyph in the UI. Inline SVG has no
# network dependency and always renders at the exact size/color requested.
# ---------------------------------------------------------------------------
ICONS = {
    "shield": '<path d="M12 2 4 5v6c0 5 3.4 9 8 11 4.6-2 8-6 8-11V5l-8-3z"/>',
    "eco": '<path d="M5 21c9 0 14-5 14-14V4h-3C7 4 3 9 3 15v6z"/><path d="M5 21c4-4 8-8 14-14"/>',
    "location_on": '<path d="M12 21s7-6.6 7-12a7 7 0 1 0-14 0c0 5.4 7 12 7 12z"/><circle cx="12" cy="9" r="2.5"/>',
    "trending_down": '<path d="M4 7l7 7 4-4 5 5"/><path d="M15 15h5v-5"/>',
    "trending_up": '<path d="M4 17l7-7 4 4 5-5"/><path d="M15 9h5v5"/>',
    "trending_flat": '<path d="M4 12h16"/><path d="M15 8l4 4-4 4"/>',
    "thermostat": '<path d="M12 14.5V5a2 2 0 1 0-4 0v9.5a4 4 0 1 0 4 0z"/>',
    "water_drop": '<path d="M12 3s6 6.5 6 11a6 6 0 1 1-12 0c0-4.5 6-11 6-11z"/>',
    "air": '<path d="M4 8h11a2.5 2.5 0 1 0-2.5-2.5"/><path d="M4 12h14a2.5 2.5 0 1 1-2.5 2.5"/><path d="M4 16h8a2 2 0 1 1-2 2"/>',
    "blur_on": '<circle cx="6" cy="6" r="1.4"/><circle cx="12" cy="6" r="1.4"/><circle cx="18" cy="6" r="1.4"/><circle cx="6" cy="12" r="1.4"/><circle cx="12" cy="12" r="1.4"/><circle cx="18" cy="12" r="1.4"/><circle cx="6" cy="18" r="1.4"/><circle cx="12" cy="18" r="1.4"/><circle cx="18" cy="18" r="1.4"/>',
    "grain": '<circle cx="7" cy="7" r="1.3"/><circle cx="17" cy="7" r="1.3"/><circle cx="7" cy="17" r="1.3"/><circle cx="17" cy="17" r="1.3"/><circle cx="12" cy="12" r="1.3"/>',
    "science": '<path d="M9 3h6"/><path d="M10 3v6l-5.5 9.5A2 2 0 0 0 6.2 21h11.6a2 2 0 0 0 1.7-2.5L14 9V3"/>',
    "cloud": '<path d="M7 18a4 4 0 1 1 .7-7.9A5 5 0 0 1 17 11a3.5 3.5 0 0 1-.5 7H7z"/>',
    "check_circle": '<circle cx="12" cy="12" r="9"/><path d="M8 12l3 3 5-6"/>',
}


def icon_svg(name, color="currentColor", size=20, stroke_width=2):
    inner = ICONS.get(name, "")
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" '
        f'stroke-linejoin="round" style="vertical-align:middle;flex-shrink:0;">{inner}</svg>'
    )


# ---------------------------------------------------------------------------
# Styling: light, card-based theme with soft shadows instead of borders,
# closer to a hand-built product dashboard than Streamlit's plain default.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Page background: soft slate tint instead of stark white */
    .stApp { background-color: #eef1f6; }

    .block-container { padding-top: 3.5rem; padding-bottom: 3rem; padding-left: 2.5rem; padding-right: 2.5rem; max-width: 100%; }

    .hero-title { font-size: 2.1rem; font-weight: 800; margin-bottom: 0.1rem; letter-spacing: -0.02em; color: #1a1f29; }
    .hero-subtitle { color: #6b7280; font-size: 0.95rem; margin-bottom: 1.4rem; }

    .card {
        background-color: #ffffff;
        border: 1px solid #e4e8f0;
        border-radius: 16px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(30, 41, 59, 0.06);
    }

    .group-card {
        background-color: #ffffff;
        border: 1px solid #e4e8f0;
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 1px 3px rgba(30, 41, 59, 0.06);
    }

    .inner-stat {
        background-color: #e7edfb;
        border-radius: 12px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.2rem;
    }

    .stat-label { color: #6b7280; font-size: 0.74rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 0.3rem; display: flex; align-items: center; gap: 4px; }
    .stat-value { font-size: 1.4rem; font-weight: 700; color: #1a1f29; }
    .stat-sub { color: #9aa4b2; font-size: 0.8rem; margin-top: 0.15rem; }

    .section-title { font-size: 1.05rem; font-weight: 700; margin: 1.5rem 0 0.5rem 0; color: #1a1f29; }
    .section-caption { color: #6b7280; font-size: 0.85rem; margin-bottom: 0.8rem; }

    .badge {
        display: inline-flex;
        align-items: center;
        padding: 2px 11px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 700;
    }

    div[data-testid="stSelectbox"] label { font-weight: 600; color: #1a1f29; }

    .nav-bar { display: flex; align-items: center; justify-content: space-between; margin-bottom: 1.6rem; }
    .nav-left { display: flex; align-items: center; gap: 0.8rem; }
    .icon-badge {
        width: 42px; height: 42px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        flex-shrink: 0;
    }
    .nav-meta { color: #6b7280; font-size: 0.82rem; display: flex; align-items: center; gap: 1.1rem; }

    .eyebrow { color: #6b7280; font-size: 0.78rem; font-weight: 600; display: flex; align-items: center; gap: 4px; margin-bottom: 0.3rem; }
    .hero-card-title { font-size: 1.5rem; font-weight: 800; color: #1a1f29; margin-bottom: 0.1rem; margin-top: 0.35rem; }
    .hero-card-sub { color: #6b7280; font-size: 0.92rem; margin-bottom: 0.3rem; }
    .delta-pill {
        display: inline-flex; align-items: center; gap: 3px;
        padding: 2px 10px; border-radius: 999px; font-size: 0.78rem; font-weight: 700;
    }

    /* Restyle Streamlit's native tabs as rounded pill buttons */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #e9edf5;
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
        box-shadow: 0 1px 3px rgba(30, 41, 59, 0.1);
    }

    /* Tighten Streamlit's default column/element vertical gaps inside cards
       so icon rows and titles sit close together instead of leaving a
       visible band of empty space. */
    div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stVerticalBlock"] { gap: 0.35rem; }

    /* Give bordered Streamlit containers (the hero cards) real breathing
       room at the bottom instead of letting the last element (the
       health-guidance box) sit flush against the card's own edge. */
    div[data-testid="stVerticalBlockBorderWrapper"] > div { padding-bottom: 0.6rem; }

    /* Equal-height hero cards, scoped ONLY to the two-card row above (via
       the explicit key= given to each st.container, which Streamlit turns
       into an "st-key-..." class on that exact wrapper div). Scoping with
       :has() means this never touches any other st.container on the page.
       Within that scope only, every descendant div is forced to height:
       100%, which is a blunt instrument but a reliable one: it doesn't
       matter how many unlabeled wrapper divs Streamlit inserts between a
       column and the bordered container inside it (that's what silently
       broke the previous, narrower version of this rule and let the right
       card's content spill past its own shorter border), because every
       level in the chain now explicitly has height:100% rather than
       relying on a couple of levels I guessed at. */
    div[data-testid="stHorizontalBlock"]:has([class*="st-key-hero_current_card"]) {
        align-items: stretch;
    }
    div[data-testid="stHorizontalBlock"]:has([class*="st-key-hero_current_card"]) > div[data-testid="column"] {
        display: flex;
    }
    div[data-testid="stHorizontalBlock"]:has([class*="st-key-hero_current_card"]) > div[data-testid="column"] div {
        height: 100%;
    }
    [class*="st-key-hero_current_card"], [class*="st-key-hero_status_card"] {
        display: flex;
        flex-direction: column;
        overflow: hidden;
        border-radius: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


POLLUTANT_THRESHOLDS = {
    "pm2_5": (55, "PM2.5"),
    "pm10": (150, "PM10"),
    "o3": (100, "Ozone (O₃)"),
    "no2": (100, "Nitrogen Dioxide (NO₂)"),
    "so2": (75, "Sulfur Dioxide (SO₂)"),
    "co": (4000, "Carbon Monoxide (CO)"),
}


def primary_pollutant_driver(row):
    """
    Identifies which pollutant is proportionally furthest past its rough
    unhealthy threshold, so guidance can name the actual driver in this
    city right now instead of giving the same generic advice regardless
    of which pollutant is actually the problem.
    """
    best_label, best_ratio, best_value = None, 0, 0
    for col, (threshold, label) in POLLUTANT_THRESHOLDS.items():
        value = row.get(col, 0)
        ratio = value / threshold if threshold else 0
        if ratio > best_ratio:
            best_label, best_ratio, best_value = label, ratio, value
    return best_label, best_value


def aqi_color_and_label(aqi):
    if aqi <= 50:
        return "#0f766e", "Good"
    elif aqi <= 100:
        return "#b45309", "Moderate"
    elif aqi <= 150:
        return "#c2410c", "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "#b91c1c", "Unhealthy"
    elif aqi <= 300:
        return "#6d28d9", "Very Unhealthy"
    else:
        return "#7f1d1d", "Hazardous"


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
    icon_html = f'{icon_svg(icon, color="#6b7280", size=15)} ' if icon else ""
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
        xaxis=dict(gridcolor="#e4e8f0", zerolinecolor="#e4e8f0"),
        yaxis=dict(gridcolor="#e4e8f0", zerolinecolor="#e4e8f0"),
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


def aqi_headline(aqi):
    if aqi <= 50:
        return "Air Quality is Good", "Conditions are clean and safe for all outdoor activities."
    elif aqi <= 100:
        return "Air Quality is Acceptable", "Current air quality conditions are within a generally acceptable range for most people."
    elif aqi <= 150:
        return "Sensitive Groups May Be Affected", "Air quality may pose a moderate health concern for sensitive individuals."
    elif aqi <= 200:
        return "Air Quality is Unhealthy", "Air quality is at levels that may affect the general population, not just sensitive groups."
    elif aqi <= 300:
        return "Air Quality is Very Unhealthy", "Health alert: current conditions pose a serious risk to the general population."
    else:
        return "Health Emergency Conditions", "Air quality has reached hazardous levels affecting the entire population."


def safety_precautions(aqi):
    if aqi <= 50:
        return ["Enjoy outdoor activities as normal.", "No special precautions needed."]
    elif aqi <= 100:
        return [
            "Most people can continue normal outdoor activities.",
            "Unusually sensitive individuals should watch for symptoms like coughing or shortness of breath.",
        ]
    elif aqi <= 150:
        return [
            "Sensitive groups (children, elderly, people with asthma or heart conditions) should reduce prolonged outdoor exertion.",
            "Consider a well-fitted mask (N95/KN95) outdoors if you're in a sensitive group.",
            "Keep quick-relief medication on hand if you have asthma.",
        ]
    elif aqi <= 200:
        return [
            "Everyone should limit prolonged or heavy outdoor exertion.",
            "Wear a well-fitted N95/KN95 mask outdoors.",
            "Keep windows closed, run an air purifier indoors if you have one.",
            "Sensitive groups should stay indoors where possible.",
        ]
    elif aqi <= 300:
        return [
            "Avoid outdoor exertion entirely.",
            "Wear a properly fitted N95/KN95 mask if you must go outside.",
            "Keep windows and doors sealed, run an air purifier continuously if available.",
            "Sensitive groups should remain indoors at all times.",
        ]
    else:
        return [
            "Avoid all outdoor activity.",
            "Stay indoors with windows and doors sealed.",
            "Run an air purifier continuously if available.",
            "Seek medical attention if experiencing difficulty breathing.",
        ]


@st.cache_resource
def load_feature_store():
    return get_feature_store()


@st.cache_resource
def load_model_registry():
    return get_model_registry()


@st.cache_data(ttl=900, show_spinner=False)
def get_national_overview(_fs, _mr, cities):
    """
    Pulls current AQI for every monitored city in one pass, for the
    national comparison view. Cached for 15 minutes (Streamlit reruns
    the whole script on every interaction, so without caching, simply
    switching the city dropdown would trigger a fresh live fetch across
    all five cities every single time, five times the API load for no
    reason since national conditions don't meaningfully change minute
    to minute).
    """
    overview = []
    for city in cities:
        try:
            result = predict_city_with_features(_fs, _mr, city)
            if result is None:
                continue
            forecast, _, _ = result
            aqi = forecast["current_aqi"]
            color, label = aqi_color_and_label(aqi)
            overview.append({"name": city["name"], "aqi": aqi, "color": color, "label": label})
        except Exception:
            continue
    return overview


fs = load_feature_store()
mr = load_model_registry()

cities = get_cities()
city_names = [c["name"] for c in cities]

nav_left, nav_right = st.columns([2, 1.6])
with nav_left:
    st.markdown(
        f"""
        <div class="nav-bar">
            <div class="nav-left">
                <div class="icon-badge" style="background-color:#d6e9e6;">
                    {icon_svg('eco', color='#0f766e', size=22)}
                </div>
                <div>
                    <div style="font-size:1.3rem; font-weight:800; color:#1a1f29; line-height:1.1;">AQI Prediction</div>
                    <div style="color:#6b7280; font-size:0.82rem;">AI-powered air quality intelligence</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with nav_right:
    ctrl_col1, ctrl_col2 = st.columns([2, 1])
    with ctrl_col1:
        selected_name = st.selectbox("City", city_names, label_visibility="collapsed")
    with ctrl_col2:
        refresh_clicked = st.button("↻ Refresh", use_container_width=True)
    st.markdown(
        f'<div class="nav-meta" style="justify-content:flex-end; margin-top:0.3rem;">Updated {datetime.now().strftime("%I:%M %p")}</div>',
        unsafe_allow_html=True,
    )

if refresh_clicked:
    st.rerun()

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
    headline, headline_desc = aqi_headline(current_aqi)

    prev_aqi = history_df["aqi"].iloc[-1] if len(history_df) else current_aqi
    delta = current_aqi - prev_aqi
    if delta < 0:
        delta_color, delta_icon, delta_text = "#0f766e", "trending_down", f"{abs(delta):.0f} points · Improving vs previous reading"
    elif delta > 0:
        delta_color, delta_icon, delta_text = "#b91c1c", "trending_up", f"{abs(delta):.0f} points · Worsening vs previous reading"
    else:
        delta_color, delta_icon, delta_text = "#6b7280", "trending_flat", "No change vs previous reading"

    # --- Hero: two cards side by side ---
    top_left, top_right = st.columns([1, 1])

    with top_left:
        with st.container(border=True, key="hero_current_card"):
            info_col, gauge_col = st.columns([1.1, 1])
            with info_col:
                st.markdown(
                    f"""<div class="eyebrow">{icon_svg('location_on', size=15)} {selected_name.upper()}</div>
                    <div class="hero-card-title">Current Air Quality</div>
                    <div class="hero-card-sub">{current_label}</div>
                    <div class="delta-pill" style="background-color:{delta_color}1a; color:{delta_color};">
                        {icon_svg(delta_icon, color=delta_color, size=15)}
                        {delta_text}
                    </div>
                    <div style="color:#9aa4b2; font-size:0.8rem; margin-top:0.8rem;">
                        Updated at hour {datetime.now().strftime("%H:00")}
                    </div>""",
                    unsafe_allow_html=True,
                )
            with gauge_col:
                gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=current_aqi,
                        number={"font": {"size": 34, "color": current_color}},
                        gauge={
                            "axis": {"range": [0, 300], "tickwidth": 1, "tickcolor": "#c9cfd8"},
                            "bar": {"color": current_color, "thickness": 0.28},
                            "bgcolor": "white",
                            "borderwidth": 0,
                            "steps": [
                                {"range": [0, 50], "color": "rgba(15,118,110,0.13)"},
                                {"range": [50, 100], "color": "rgba(180,83,9,0.13)"},
                                {"range": [100, 150], "color": "rgba(194,65,12,0.13)"},
                                {"range": [150, 200], "color": "rgba(185,28,28,0.13)"},
                                {"range": [200, 300], "color": "rgba(109,40,217,0.13)"},
                            ],
                        },
                    )
                )
                st.plotly_chart(styled_plotly(gauge, height=180), use_container_width=True, config={"displayModeBar": False})

    with top_right:
        with st.container(border=True, key="hero_status_card"):
            badge_col, spacer, pill_col = st.columns([1, 2, 2])
            with badge_col:
                st.markdown(
                    f'<div class="icon-badge" style="background-color:{current_color}1a;">{icon_svg("shield", color=current_color, size=22)}</div>',
                    unsafe_allow_html=True,
                )
            with pill_col:
                st.markdown(
                    f'<div style="text-align:right; margin-top:0.5rem;">{badge_html(current_label, current_color)}</div>',
                    unsafe_allow_html=True,
                )
            st.markdown(
                f"""<div class="hero-card-title">{headline}</div>
                <div class="hero-card-sub">{headline_desc}</div>
                <div class="inner-stat" style="line-height:1.5; font-size:0.88rem; color:#374151; margin-top:0.3rem; margin-bottom:0.2rem;"><b>Health guidance:</b> {health_guidance(current_aqi)}</div>""",
                unsafe_allow_html=True,
            )
            max_forecast = max(forecast[h] for h in HORIZONS)
            if max_forecast > 150:
                st.markdown(
                    '<div style="margin-top:0.7rem; color:#b91c1c; font-size:0.85rem; font-weight:600;">⚠️ AQI is expected to reach unhealthy levels within the next 3 days.</div>',
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
                        marker_color=["#b91c1c" if v > 0 else "#0f766e" for v in top_contributions.values],
                    )
                )
                shap_fig.update_layout(
                    title=f"Top feature contributions ({horizon_tab_labels[horizon]} forecast)",
                    xaxis_title="Impact on predicted AQI",
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(styled_plotly(shap_fig), use_container_width=True, key=f"shap_{horizon}", config={"displayModeBar": False})
                st.caption("Red bars pushed the forecast higher, green bars pulled it lower.")

    # --- Safety precautions, keyed to the worst predicted conditions ahead, not just the current moment ---
    worst_horizon = max(HORIZONS, key=lambda h: forecast[h])
    worst_value = forecast[worst_horizon]
    worst_color, worst_label = aqi_color_and_label(worst_value)
    worst_horizon_display = {"target_aqi_24h": "24 hours", "target_aqi_48h": "48 hours", "target_aqi_72h": "72 hours"}[worst_horizon]
    driver_label, driver_value = primary_pollutant_driver(row)

    st.markdown('<div class="section-title">Safety Precautions</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-caption">Based on the worst air quality expected in {selected_name} over the next 3 days '
        f'({worst_label}, predicted around {worst_horizon_display} from now).</div>',
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        if driver_label:
            st.markdown(
                f'<div style="font-size:0.85rem; color:#6b7280; margin-bottom:0.8rem;">'
                f'<b>{driver_label}</b> is the main pollutant of concern right now ({driver_value:.1f} µg/m³).</div>',
                unsafe_allow_html=True,
            )
        for tip in safety_precautions(worst_value):
            st.markdown(
                f'<div style="display:flex; align-items:flex-start; gap:0.6rem; margin-bottom:0.6rem;">'
                f'{icon_svg("check_circle", color=worst_color, size=19)}'
                f'<span style="font-size:0.9rem; color:#374151; line-height:1.4;">{tip}</span></div>',
                unsafe_allow_html=True,
            )

    # --- Forecast accuracy backtest: real predicted-vs-actual over recent history ---
    st.markdown('<div class="section-title">Forecast Accuracy Over Time</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">The 24h model\'s predictions made at each past hour, compared to what AQI actually turned out to be.</div>',
        unsafe_allow_html=True,
    )
    extended_hist = get_recent_history(fs, selected_name, hours=120)
    X_hist = extended_hist[FEATURE_COLUMNS].dropna()
    if len(X_hist) >= 10:
        model_24h_bt = load_model(mr, "target_aqi_24h")
        preds = model_24h_bt.predict(X_hist)
        pred_times = extended_hist.loc[X_hist.index, "timestamp"] + pd.Timedelta(hours=24)

        bt_fig = go.Figure()
        bt_fig.add_trace(
            go.Scatter(
                x=extended_hist["timestamp"], y=extended_hist["aqi"],
                mode="lines", name="Actual AQI",
                line=dict(color="#2563eb", width=2.5),
            )
        )
        bt_fig.add_trace(
            go.Scatter(
                x=pred_times, y=preds,
                mode="lines", name="Predicted (24h ahead)",
                line=dict(color="#c2410c", width=2, dash="dot"),
            )
        )
        bt_fig.update_layout(yaxis_title="AQI", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(styled_plotly(bt_fig, height=340), use_container_width=True, config={"displayModeBar": False})
        st.caption("Dotted orange shows what the model predicted 24 hours in advance for each point in time, plotted against the solid blue actual reading at that same time.")
    else:
        st.info("Not enough historical data yet to show a backtest for this city.")

# --- National Overview: something a single-city dashboard can't offer, shown regardless of the selected city's own data availability ---
st.markdown('<div class="section-title">National Overview</div>', unsafe_allow_html=True)
st.markdown('<div class="section-caption">Live AQI across all five monitored cities, ranked worst to best</div>', unsafe_allow_html=True)

with st.container(border=True):
    with st.spinner("Loading national comparison..."):
        overview = get_national_overview(fs, mr, cities)

    if overview:
        overview_sorted = sorted(overview, key=lambda c: c["aqi"], reverse=True)
        bar_colors = [
            "#1a1f29" if c["name"] == selected_name else c["color"]
            for c in overview_sorted
        ]
        overview_fig = go.Figure(
            go.Bar(
                x=[c["aqi"] for c in overview_sorted],
                y=[c["name"] for c in overview_sorted],
                orientation="h",
                marker_color=bar_colors,
                text=[f"{c['aqi']:.0f} · {c['label']}" for c in overview_sorted],
                textposition="outside",
            )
        )
        overview_fig.update_layout(
            yaxis=dict(autorange="reversed"),
            xaxis_title="Current AQI",
            margin=dict(l=10, r=80, t=10, b=10),
        )
        st.plotly_chart(styled_plotly(overview_fig, height=260), use_container_width=True, config={"displayModeBar": False})
        st.caption(f"{selected_name} is highlighted in black. Use the dropdown near the top of the page to explore a different city.")
    else:
        st.info("National comparison isn't available yet, not enough recent history for other cities.")