import os
import time
import requests
from dotenv import load_dotenv
from config import get_cities
from feature_engineering import engineer_features

load_dotenv()
KEY = os.getenv("OPENWEATHER_KEY")

def fetch_with_retry(url, retries=3, delay=2):
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"  Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
    return None

for city in get_cities():
    name, lat, lon = city["name"], city["lat"], city["lon"]
    print(f"\nFetching {name}...")

    weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={KEY}&units=metric"
    air_url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={KEY}"

    weather = fetch_with_retry(weather_url)
    air = fetch_with_retry(air_url)

    if weather is None or air is None:
        print(f"  Skipping {name} — failed after retries.")
        continue

    row = engineer_features(name, weather, air)
    print(" ", row)