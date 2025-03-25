# utils/weather.py
import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)

def fetch_weather(lat, lon, start_date, end_date):
    api_key = os.getenv("OPENWEATHERMAP_API_KEY")
    if not api_key:
        logger.warning("OPENWEATHERMAP_API_KEY not set, skipping weather fetch")
        return {}

    url = "https://api.openweathermap.org/data/3.0/onecall"
    params = {
        "lat": lat,
        "lon": lon,
        "exclude": "current,minutely,hourly,alerts",
        "units": "metric",
        "appid": api_key
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        daily_forecasts = data.get("daily", [])

        weather_by_date = {}
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            for day in daily_forecasts:
                forecast_date = datetime.fromtimestamp(day["dt"]).strftime("%Y-%m-%d")
                if forecast_date == date_str:
                    weather = day.get("weather", [{}])[0].get("main", "Clear")
                    weather_by_date[date_str] = weather
                    break
            else:
                weather_by_date[date_str] = "Clear"
            current_date += timedelta(days=1)
        return weather_by_date
    except requests.exceptions.RequestException as e:
        logger.error(f"OpenWeatherMap API Error: {e}")
        return {}