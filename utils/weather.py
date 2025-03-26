# utils/weather.py
import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)

def fetch_weather(lat, lon, start_date, end_date):
    """
    Fetch weather data for the given coordinates between start_date and end_date.
    
    Args:
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.
        start_date (datetime): Start date for weather data.
        end_date (datetime): End date for weather data.
    
    Returns:
        dict: A dictionary mapping dates (YYYY-MM-DD) to weather conditions (e.g., {"2025-03-26": "Clear"}).
    """
    api_key = os.getenv("OPENWEATHERMAP_API_KEY")
    if not api_key:
        logger.warning("OPENWEATHERMAP_API_KEY not set, skipping weather fetch")
        return {}

    # Check if the dates are too far in the future (OpenWeatherMap free tier only supports 7-day forecasts)
    current_date = datetime.now()
    max_forecast_date = current_date + timedelta(days=7)
    if start_date > max_forecast_date:
        logger.warning(f"Weather data unavailable for {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} (beyond 7-day forecast limit)")
        weather_by_date = {}
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            weather_by_date[date_str] = "Weather unavailable (future date)"
            current_date += timedelta(days=1)
        return weather_by_date

    url = "https://api.openweathermap.org/data/3.0/onecall"
    params = {
        "lat": lat,
        "lon": lon,
        "exclude": "current,minutely,hourly,alerts",
        "units": "metric",
        "appid": api_key
    }
    try:
        response = requests.get(url, params=params, timeout=10)
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
                logger.warning(f"No weather data found for {date_str}, defaulting to 'Clear'")
                weather_by_date[date_str] = "Clear"
            current_date += timedelta(days=1)
        return weather_by_date
    except requests.exceptions.RequestException as e:
        logger.error(f"OpenWeatherMap API Error: {e}")
        # Return a fallback for the entire date range
        weather_by_date = {}
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            weather_by_date[date_str] = "Weather unavailable (API error)"
            current_date += timedelta(days=1)
        return weather_by_date