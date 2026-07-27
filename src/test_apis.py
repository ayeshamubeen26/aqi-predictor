import os
import requests
from dotenv import load_dotenv
from config import get_cities
from feature_engineering import engineer_features

load_dotenv()
KEY = os.getenv("OPENWEATHER_KEY")

for city in get_cities():
    name, lat, lon = city["name"], city["lat"], city["lon"]

    weather = requests.get(
        f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={KEY}&units=metric"
    ).json()
    air = requests.get(
        f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={KEY}"
    ).json()

    row = engineer_features(name, weather, air)
    print(row)