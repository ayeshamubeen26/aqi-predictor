import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
from config import get_cities
from feature_engineering import engineer_features
from feature_store import insert_rows

load_dotenv()
OPENWEATHER_KEY = os.getenv("OPENWEATHER_KEY")

def get_date_range(days_back, weather_buffer_days=5):
    """
    Returns the start and end of a historical window.
    - Unix timestamps (for OpenWeather pollution) go right up to now.
    - Date strings (for Open-Meteo weather) end a few days earlier,
      since Open-Meteo's archive has a ~2-5 day processing delay.
    """
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days_back)
    weather_end_dt = end_dt - timedelta(days=weather_buffer_days)

    return {
        "start_unix": int(start_dt.timestamp()),
        "end_unix": int(end_dt.timestamp()),
        "start_date": start_dt.strftime("%Y-%m-%d"),
        "end_date": weather_end_dt.strftime("%Y-%m-%d"),
    }

def fetch_pollution_history(lat, lon, start_unix, end_unix):
    """
    Fetches historical pollutant data from OpenWeather for one city.
    Returns a dict keyed by hour (as a datetime).
    """
    url = (
        f"https://api.openweathermap.org/data/2.5/air_pollution/history"
        f"?lat={lat}&lon={lon}&start={start_unix}&end={end_unix}&appid={OPENWEATHER_KEY}"
    )
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    data = response.json()

    pollution_by_hour = {}
    for entry in data["list"]:
        hour_dt = datetime.fromtimestamp(entry["dt"])
        pollution_by_hour[hour_dt] = entry["components"]

    return pollution_by_hour

def fetch_weather_history(lat, lon, start_date, end_date):
    """
    Fetches historical weather data from Open-Meteo for one city.
    Returns a dict keyed by hour (as a datetime).
    """
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}"
        f"&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m"
    )
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    data = response.json()

    hourly = data["hourly"]
    weather_by_hour = {}

    for i, time_str in enumerate(hourly["time"]):
        hour_dt = datetime.fromisoformat(time_str)
        weather_by_hour[hour_dt] = {
            "temp": hourly["temperature_2m"][i],
            "humidity": hourly["relative_humidity_2m"][i],
            "wind_speed": hourly["wind_speed_10m"][i],
            "wind_deg": hourly["wind_direction_10m"][i],
        }

    return weather_by_hour

def combine_history(city_name, pollution_by_hour, weather_by_hour):
    """
    Matches pollution and weather records by hour, and runs each
    matched hour through feature engineering. Hours missing from
    either side are skipped safely.
    """
    combined_rows = []
    skipped = 0

    for hour, components in pollution_by_hour.items():
        if hour not in weather_by_hour:
            skipped += 1
            continue

        weather_values = weather_by_hour[hour]

        # Build fake "weather_json" and "air_json" shapes so we can
        # reuse engineer_features() exactly as it already works
        fake_weather_json = {
            "dt": int(hour.timestamp()),
            "main": {"temp": weather_values["temp"], "humidity": weather_values["humidity"]},
            "wind": {"speed": weather_values["wind_speed"], "deg": weather_values["wind_deg"]},
        }
        fake_air_json = {
            "list": [{"components": components}]
        }

        row = engineer_features(city_name, fake_weather_json, fake_air_json)
        combined_rows.append(row)

    print(f"  {city_name}: matched {len(combined_rows)} hours, skipped {skipped} (no weather match)")
    return combined_rows

if __name__ == "__main__":
    DAYS_BACK = 90  # adjust this later once you're confident everything works

    dates = get_date_range(DAYS_BACK)
    print("Date range:", dates)

    all_rows = []

    for city in get_cities():
        name, lat, lon = city["name"], city["lat"], city["lon"]
        print(f"\nFetching history for {name}...")

        try:
            pollution = fetch_pollution_history(lat, lon, dates["start_unix"], dates["end_unix"])
            weather = fetch_weather_history(lat, lon, dates["start_date"], dates["end_date"])
        except requests.RequestException as e:
            print(f"  Failed to fetch history for {name}: {e}")
            continue

        rows = combine_history(name, pollution, weather)
        all_rows.extend(rows)

    print(f"\nTotal combined rows across all cities: {len(all_rows)}")

    if all_rows:
        insert_rows(all_rows)
    else:
        print("No rows to insert.")