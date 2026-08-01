import os
from dotenv import load_dotenv

load_dotenv()

def get_cities():
    raw = os.getenv("CITIES", "")
    cities = []
    for entry in raw.split(","):
        name, lat, lon = entry.split(":")
        cities.append({"name": name, "lat": float(lat), "lon": float(lon)})
    return cities

    