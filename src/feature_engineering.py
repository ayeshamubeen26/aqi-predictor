import math
from datetime import datetime

def engineer_features(city_name, weather_json, air_json):
    dt = datetime.fromtimestamp(weather_json["dt"])

    hour = dt.hour
    day_of_week = dt.weekday()
    month = dt.month

    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)
    dow_sin = math.sin(2 * math.pi * day_of_week / 7)
    dow_cos = math.cos(2 * math.pi * day_of_week / 7)

    wind_deg = weather_json["wind"].get("deg", 0)
    wind_sin = math.sin(2 * math.pi * wind_deg / 360)
    wind_cos = math.cos(2 * math.pi * wind_deg / 360)

    components = air_json["list"][0]["components"]

    return {
        "city": city_name,
        "timestamp": dt.isoformat(),
        "hour": hour,
        "day_of_week": day_of_week,
        "month": month,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "dow_sin": dow_sin,
        "dow_cos": dow_cos,
        "temp": weather_json["main"]["temp"],
        "humidity": weather_json["main"]["humidity"],
        "wind_speed": weather_json["wind"]["speed"],
        "wind_sin": wind_sin,
        "wind_cos": wind_cos,
        "co": components["co"],
        "no2": components["no2"],
        "o3": components["o3"],
        "so2": components["so2"],
        "pm2_5": components["pm2_5"],
        "pm10": components["pm10"],
    }