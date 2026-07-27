import os
import requests
from dotenv import load_dotenv
from config import get_cities

load_dotenv()
KEY = os.getenv("OPENWEATHER_KEY")

for city in get_cities():
    name, lat, lon = city["name"], city["lat"], city["lon"]
    print(f"\n=== {name} ===")

    weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={KEY}&units=metric"
    weather = requests.get(weather_url).json()
    print("Weather:", weather)

    air_url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={KEY}"
    air = requests.get(air_url).json()
    print("Air pollution:", air)