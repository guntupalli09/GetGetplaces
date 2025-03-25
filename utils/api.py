# utils/api.py
import requests
import os
from dotenv import load_dotenv
from geopy.distance import geodesic

load_dotenv()

def get_coordinates(destination):
    api_key = os.getenv("GOOGLE_GEOCODING_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_GEOCODING_API_KEY not set")
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={destination}&key={api_key}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("status") == "OK":
            location = data["results"][0]["geometry"]["location"]
            return location["lat"], location["lng"]
        raise ValueError("No coordinates found")
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Geocoding API Error: {e}")

def get_airport_code(destination):
    # Placeholder for airport code lookup (e.g., using a static map or API)
    airport_codes = {"Tampa": "TPA", "Orlando": "MCO", "Miami": "MIA"}
    return airport_codes.get(destination, None)

def haversine_distance(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).km