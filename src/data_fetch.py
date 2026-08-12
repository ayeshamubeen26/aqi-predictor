import os
import sys
import time
import requests
from dotenv import load_dotenv
from config import get_cities
from feature_engineering import engineer_features
from feature_store import insert_rows

load_dotenv()
KEY = os.getenv("OPENWEATHER_KEY")

def fetch_with_retry(url, retries=3, delay=2):
    last_error = None
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            last_error = e
            print(f"  Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
    if last_error is not None and getattr(last_error, "response", None) is not None \
            and last_error.response.status_code == 401:
        print("  ^ 401 Unauthorized: OPENWEATHER_KEY is invalid, expired, or not yet active.")
    return None

all_rows = []
failed_cities = []

for city in get_cities():
    name, lat, lon = city["name"], city["lat"], city["lon"]
    print(f"\nFetching {name}...")

    weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={KEY}&units=metric"
    air_url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={KEY}"

    weather = fetch_with_retry(weather_url)
    air = fetch_with_retry(air_url)

    if weather is None or air is None:
        print(f"  Skipping {name} — failed after retries.")
        failed_cities.append(name)
        continue

    row = engineer_features(name, weather, air)
    all_rows.append(row)
    print(" ", row)

if all_rows:
    insert_rows(all_rows)
    if failed_cities:
        # Partial failure: some cities got data, some didn't. Still exit
        # non-zero so this shows up as a failed/flagged run instead of a
        # silent partial success, but only after successfully inserting
        # whatever did come through.
        print(f"\nFailed to fetch: {', '.join(failed_cities)}")
        sys.exit(1)
else:
    print("No rows fetched from any city, nothing was inserted.")
    print("This usually means OPENWEATHER_KEY is invalid/expired, or OpenWeather is down.")
    sys.exit(1)