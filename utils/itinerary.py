# utils/itinerary.py
from datetime import datetime, timedelta
from utils.distance import estimate_travel_time
from utils.api import get_coordinates
import logging

logger = logging.getLogger(__name__)

def generate_itinerary(destinations, pick_up_date, drop_off_date, hotels_by_city, cars, attractions_by_city, restaurants_by_city, weather_by_city, budget, pick_up_time="10:00", drop_off_time="10:00"):
    itinerary = []
    cost_summary = {"hotels": 0, "cars": 0, "food": 0, "total": 0}

    num_days = (drop_off_date - pick_up_date).days + 1
    num_cities = len(destinations)
    days_per_city = max(1, num_days // num_cities)
    remaining_days = num_days % num_cities

    current_date = pick_up_date
    city_days = {}
    for city in destinations:
        days = days_per_city + (1 if remaining_days > 0 else 0)
        city_days[city] = days
        remaining_days -= 1 if remaining_days > 0 else 0

    food_cost_per_day = 50
    cost_summary["food"] = food_cost_per_day * num_days

    car = cars[0] if cars else {"name": "No car recommended", "price": 0}
    cost_summary["cars"] = car["price"] if cars else 0

    while current_date <= drop_off_date:
        days_passed = (current_date - pick_up_date).days
        city_days_cumulative = 0
        current_city = destinations[0]
        for i, city in enumerate(destinations):
            city_days_cumulative += city_days[city]
            if days_passed < city_days_cumulative:
                current_city = city
                break

        hotels = hotels_by_city.get(current_city, [])
        attractions = attractions_by_city.get(current_city, [])
        restaurants = restaurants_by_city.get(current_city, [])
        hotel = hotels[0] if hotels else {"name": "Placeholder Hotel", "price": 100}
        weather = weather_by_city.get(current_city, {}).get(current_date.strftime("%Y-%m-%d"), "Clear")

        # Use placeholder cost if hotel price is 0
        cost_summary["hotels"] += hotel["price"] if hotel["price"] > 0 else 100

        is_rainy = "Rain" in weather
        if is_rainy:
            try:
                central_lat, central_lon = get_coordinates(current_city)
            except ValueError as e:
                logger.error(f"Failed to fetch coordinates for {current_city} during itinerary generation: {e}")
                central_lat, central_lon = (0, 0)
            hotel_coords = (hotels[0]["geometry"]["location"]["lat"] if hotels and "geometry" in hotels[0] else central_lat,
                           hotels[0]["geometry"]["location"]["lng"] if hotels and "geometry" in hotels[0] else central_lon) if hotels else (central_lat, central_lon)
            attractions = recommend_attractions(current_city, pick_up_date, drop_off_date, hotel_coords[0], hotel_coords[1], prefer_indoor=True)

        day_itinerary = f"**Day {current_date.strftime('%Y-%m-%d')} in {current_city}**\n"
        day_itinerary += f"- **Weather Forecast**: {weather}\n"
        day_itinerary += f"- **Stay at**: {hotel['name']} (${hotel['price']:.1f})\n"
        if car and car["name"] != "No car recommended":
            if current_date == pick_up_date:
                day_itinerary += f"- **Pick up car at {pick_up_time}**: {car['name']} from {car['company']} (${car['price']:.1f})\n"
            elif current_date == drop_off_date:
                day_itinerary += f"- **Drop off car at {drop_off_time}**: {car['name']} from {car['company']}\n"
            else:
                day_itinerary += f"- **Travel with**: {car['name']} from {car['company']} (${car['price']:.1f})\n"

        total_attractions = len(attractions)
        total_restaurants = len(restaurants)
        attractions_per_day = max(1, total_attractions // city_days[current_city]) if total_attractions > 0 else 1
        restaurants_per_day = max(1, total_restaurants // city_days[current_city]) if total_restaurants > 0 else 1

        days_in_city = (current_date - pick_up_date).days - sum(city_days[c] for c, d in city_days.items() if destinations.index(c) < destinations.index(current_city))
        attraction_start = days_in_city * attractions_per_day
        restaurant_start = days_in_city * restaurants_per_day

        # Ensure at least one attraction and restaurant per day
        day_attractions = attractions[attraction_start:attraction_start + attractions_per_day]
        if not day_attractions:
            day_attractions = [{"name": "Placeholder Attraction", "rating": 4.0, "distance": 1.0, "reviews": ["Default review"]}]

        day_itinerary += "- **Attractions to Visit:**\n"
        start_time = datetime.strptime(pick_up_time if current_date == pick_up_date else "09:00", "%H:%M")
        for attr in day_attractions:
            travel_time = estimate_travel_time(attr["distance"])
            time_str = start_time.strftime("%I:%M %p")
            day_itinerary += f"  - {time_str} - {attr['name']} (Rating: {attr['rating']}, Distance: {attr['distance']:.1f} km, Travel: ~{travel_time:.0f} min)\n"
            for review in attr['reviews']:
                day_itinerary += f"    - Review: {review[:50]}...\n"
            start_time += timedelta(hours=1.5) + timedelta(minutes=travel_time)

        day_restaurants = restaurants[restaurant_start:restaurant_start + restaurants_per_day]
        if not day_restaurants:
            day_restaurants = [{"name": "Placeholder Restaurant", "rating": 4.0, "distance": 1.0, "reviews": ["Default review"]}]

        day_itinerary += "- **Restaurants to Dine at:**\n"
        start_time = datetime.strptime("18:00", "%H:%M")
        for rest in day_restaurants:
            travel_time = estimate_travel_time(rest["distance"])
            time_str = start_time.strftime("%I:%M %p")
            day_itinerary += f"  - {time_str} - {rest['name']} (Rating: {rest['rating']}, Distance: {rest['distance']:.1f} km, Travel: ~{travel_time:.0f} min)\n"
            for review in rest['reviews']:
                day_itinerary += f"    - Review: {review[:50]}...\n"
            start_time += timedelta(hours=1.5) + timedelta(minutes=travel_time)

        itinerary.append(day_itinerary)
        current_date += timedelta(days=1)

    cost_summary["total"] = cost_summary["hotels"] + cost_summary["cars"] + cost_summary["food"]
    cost_summary_str = (
        "**Cost Summary:**\n"
        f"- Hotels: ${cost_summary['hotels']:.1f}\n"
        f"- Car Rental: ${cost_summary['cars']:.1f}\n"
        f"- Estimated Food: ${cost_summary['food']:.1f}\n"
        f"- **Total Estimated Cost**: ${cost_summary['total']:.1f}\n"
    )

    return "\n".join(itinerary), cost_summary_str