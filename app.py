# app.py
from flask import Flask, request, render_template
import logging
from datetime import datetime
from utils.api import get_coordinates, get_airport_code
from utils.database import Database
from utils.weather import fetch_weather
from utils.itinerary import generate_itinerary
from utils.distance import haversine_distance
from models.recommendation import RecommendationModel
from models.price_predictor import PricePredictor
from nlp.parser import parse_nlp_input
from chatbot.bot import Chatbot
from vision.image_scorer import score_image
import requests
import os
from dotenv import load_dotenv

# Set up logging for getgetplaces
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
db = Database()
recommendation_model = RecommendationModel()
price_predictor = PricePredictor()
chatbot = Chatbot()

def fetch_hotels(destination, budget, pick_up_date, drop_off_date):
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    api_key = os.getenv("GOOGLE_PLACES_API_KEY")
    logger.info(f"Fetching hotels for {destination} in getgetplaces with budget {budget}")
    
    fallback_hotels = [{"name": "Placeholder Hotel", "price": 100.0, "rating": 4.0, "distance": 1.0, "reviews": ["Default review"], "geometry": {"location": {"lat": 0, "lng": 0}}}]
    if not api_key:
        logger.warning("GOOGLE_PLACES_API_KEY not set, using fallback hotels")
        return fallback_hotels

    try:
        central_lat, central_lon = get_coordinates(destination)
    except ValueError as e:
        logger.error(f"Failed to fetch coordinates for {destination}: {e}")
        return fallback_hotels

    params = {"query": f"hotels in {destination}", "type": "lodging", "key": api_key}
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        logger.debug(f"Hotel API response for {destination}: {data}")

        if not isinstance(data, dict) or data.get("status") != "OK":
            logger.error(f"Google Places API Error (Hotels) for {destination}: {data.get('error_message', 'Unknown error')}")
            return fallback_hotels

        hotels = []
        for place in data.get("results", [])[:10]:
            place_id = place.get("place_id")
            name = place.get("name", "Unknown")
            rating = float(place.get("rating", 0))
            price_level = place.get("price_level", 2)
            estimated_price = price_predictor.predict_price(float(price_level * 50), pick_up_date)

            if estimated_price > budget * 1.2:
                logger.debug(f"Hotel {name} (${estimated_price}) exceeds budget {budget * 1.2}, skipping")
                continue

            geometry = place.get("geometry", {})
            location = geometry.get("location", {})
            place_lat = location.get("lat", central_lat)
            place_lon = location.get("lng", central_lon)
            distance = haversine_distance(central_lat, central_lon, place_lat, place_lon)

            details_url = "https://maps.googleapis.com/maps/api/place/details/json"
            details_params = {"place_id": place_id, "fields": "name,rating,reviews", "key": api_key}
            details_response = requests.get(details_url, params=details_params, timeout=10)
            details_response.raise_for_status()
            details_data = details_response.json()

            if details_data.get("status") != "OK":
                logger.error(f"Google Place Details Error (Hotels) for {name}: {details_data.get('error_message', 'Unknown error')}")
                continue

            reviews = details_data.get("result", {}).get("reviews", [])
            review_texts = [review.get("text", "") for review in reviews[:2]]

            hotel = {
                "name": name,
                "price": estimated_price,
                "rating": rating,
                "distance": distance,
                "reviews": review_texts,
                "geometry": geometry
            }
            hotels.append(hotel)
            db.insert_hotel(name, estimated_price, rating, distance, destination, place_lat, place_lon, review_texts)

        if not hotels:
            logger.warning(f"No hotels found for {destination} within budget {budget}, using fallback")
            return fallback_hotels

        logger.debug(f"Returning {len(hotels)} hotels for {destination}")
        return hotels

    except requests.exceptions.RequestException as e:
        logger.error(f"Google Places API Error (Hotels) for {destination}: {e}")
        return fallback_hotels

def fetch_cars(destination, budget, pick_up_date, drop_off_date, hotel_lat=0, hotel_lon=0, pick_up_time="10:00", drop_off_time="10:00"):
    url = "https://priceline-com2.p.rapidapi.com/cars/search"
    api_key = os.getenv("RAPIDAPI_KEY_PRICELINE")
    logger.info(f"Fetching cars for {destination} in getgetplaces with budget {budget}")
    
    fallback_cars = [{"name": "Placeholder Car", "price": 50.0, "rating": 4.0, "distance": 1.0, "company": "Placeholder Company", "reviews": ["Default review"]}]
    if not api_key:
        logger.warning("RAPIDAPI_KEY_PRICELINE not set, using fallback car")
        return fallback_cars

    headers = {"X-RapidAPI-Key": api_key, "X-RapidAPI-Host": "priceline-com2.p.rapidapi.com"}

    try:
        central_lat, central_lon = get_coordinates(destination)
    except ValueError as e:
        logger.error(f"Failed to fetch coordinates for {destination}: {e}")
        return fallback_cars

    airport_code = get_airport_code(destination)
    location = airport_code if airport_code else f"{central_lat},{central_lon}"

    params = {
        "pickUpLocation": location,
        "pickUpDate": pick_up_date.strftime("%Y-%m-%d"),
        "pickUpTime": pick_up_time,
        "dropOffLocation": location,
        "dropOffDate": drop_off_date.strftime("%Y-%m-%d"),
        "dropOffTime": drop_off_time
    }
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        json_data = response.json()
        logger.debug(f"Car API response for {destination}: {json_data}")

        if not isinstance(json_data, dict):
            logger.error(f"Priceline API Error: Response is not a dictionary - {json_data}")
            return fallback_cars

        vehicles = json_data.get("data", [])
        if not vehicles:
            logger.warning("Priceline API Warning: No vehicles found in response")
            return fallback_cars

        cars = []
        for vehicle in vehicles[:10]:
            price = float(vehicle.get("price", 0))
            if price > budget * 1.2:
                logger.debug(f"Car ${price} exceeds budget {budget * 1.2}, skipping")
                continue

            vehicle_lat = vehicle.get("pickUpLocation", {}).get("latitude", hotel_lat if hotel_lat else central_lat)
            vehicle_lon = vehicle.get("pickUpLocation", {}).get("longitude", hotel_lon if hotel_lon else central_lon)
            distance = haversine_distance(hotel_lat if hotel_lat else central_lat, hotel_lon if hotel_lon else central_lon, vehicle_lat, vehicle_lon)

            car = {
                "name": vehicle.get("vehicleName", "Unknown Car"),
                "price": price,
                "rating": float(vehicle.get("rating", 0)),
                "distance": distance,
                "company": vehicle.get("company", "Unknown"),
                "reviews": vehicle.get("reviews", []) if vehicle.get("reviews") else []
            }
            cars.append(car)
            db.insert_car(car["name"], car["price"], car["rating"], car["distance"], car["company"], destination, car["reviews"])

        if not cars:
            logger.warning(f"No cars found for {destination} within budget {budget}, using fallback")
            return fallback_cars

        logger.debug(f"Returning {len(cars)} cars for {destination}")
        return cars[:5]

    except requests.exceptions.RequestException as e:
        logger.error(f"Priceline API RequestException for {destination}: {e}")
        return fallback_cars

def fetch_attractions(destination, pick_up_date, drop_off_date, hotel_lat=0, hotel_lon=0):
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    api_key = os.getenv("GOOGLE_PLACES_API_KEY")
    logger.info(f"Fetching attractions for {destination} in getgetplaces")
    
    fallback_attractions = [{"name": "Placeholder Attraction", "rating": 4.0, "distance": 1.0, "reviews": ["Default review"], "is_indoor": False, "image_score": 0}]
    if not api_key:
        logger.warning("GOOGLE_PLACES_API_KEY not set, using fallback attractions")
        return fallback_attractions

    try:
        central_lat, central_lon = get_coordinates(destination)
    except ValueError as e:
        logger.error(f"Failed to fetch coordinates for {destination}: {e}")
        return fallback_attractions

    params = {"query": f"attractions in {destination}", "type": "tourist_attraction", "key": api_key}
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        logger.debug(f"Attraction API response for {destination}: {data}")

        if not isinstance(data, dict) or data.get("status") != "OK":
            logger.error(f"Google Places API Error (Attractions) for {destination}: {data.get('error_message', 'Unknown error')}")
            return fallback_attractions

        attractions = []
        for place in data.get("results", [])[:10]:
            place_id = place.get("place_id")
            name = place.get("name", "Unknown")
            rating = float(place.get("rating", 0))

            geometry = place.get("geometry", {})
            location = geometry.get("location", {})
            place_lat = location.get("lat", hotel_lat if hotel_lat else central_lat)
            place_lon = location.get("lng", hotel_lon if hotel_lon else central_lon)

            distance = haversine_distance(hotel_lat if hotel_lat else central_lat, hotel_lon if hotel_lon else central_lon, place_lat, place_lon)

            details_url = "https://maps.googleapis.com/maps/api/place/details/json"
            details_params = {"place_id": place_id, "fields": "name,rating,reviews,types,photos", "key": api_key}
            details_response = requests.get(details_url, params=details_params, timeout=10)
            details_response.raise_for_status()
            details_data = details_response.json()

            if details_data.get("status") != "OK":
                logger.error(f"Google Place Details Error (Attractions) for {name}: {details_data.get('error_message', 'Unknown error')}")
                continue

            reviews = details_data.get("result", {}).get("reviews", [])
            review_texts = [review.get("text", "") for review in reviews[:2]]
            types = details_data.get("result", {}).get("types", [])
            is_indoor = any(t in ["museum", "gallery", "indoor"] for t in types)

            image_score = 0
            if "photos" in details_data.get("result", {}) and details_data["result"]["photos"]:
                photo_ref = details_data["result"]["photos"][0]["photo_reference"]
                photo_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={photo_ref}&key={api_key}"
                image_score = score_image(photo_url)

            attraction = {
                "name": name,
                "rating": rating,
                "distance": distance,
                "reviews": review_texts,
                "is_indoor": is_indoor,
                "image_score": image_score
            }
            attractions.append(attraction)
            details = {
                "reviews": review_texts,
                "is_indoor": is_indoor,
                "image_score": image_score,
                "coordinates": {"lat": place_lat, "lng": place_lon}
            }
            db.insert_attraction(name, rating, distance, destination, place_lat, place_lon, review_texts, is_indoor, image_score)

        if not attractions:
            logger.warning(f"No attractions found for {destination}, using fallback")
            return fallback_attractions

        logger.debug(f"Returning {len(attractions)} attractions for {destination}")
        return attractions

    except requests.exceptions.RequestException as e:
        logger.error(f"Google Places API Error (Attractions) for {destination}: {e}")
        return fallback_attractions

def fetch_restaurants(destination, pick_up_date, drop_off_date, hotel_lat=0, hotel_lon=0):
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    api_key = os.getenv("GOOGLE_PLACES_API_KEY")
    logger.info(f"Fetching restaurants for {destination} in getgetplaces")
    
    fallback_restaurants = [{"name": "Placeholder Restaurant", "rating": 4.0, "distance": 1.0, "reviews": ["Default review"]}]
    if not api_key:
        logger.warning("GOOGLE_PLACES_API_KEY not set, using fallback restaurants")
        return fallback_restaurants

    try:
        central_lat, central_lon = get_coordinates(destination)
    except ValueError as e:
        logger.error(f"Failed to fetch coordinates for {destination}: {e}")
        return fallback_restaurants

    params = {"query": f"restaurants in {destination}", "type": "restaurant", "key": api_key}
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        logger.debug(f"Restaurant API response for {destination}: {data}")

        if not isinstance(data, dict) or data.get("status") != "OK":
            logger.error(f"Google Places API Error (Restaurants) for {destination}: {data.get('error_message', 'Unknown error')}")
            return fallback_restaurants

        restaurants = []
        for place in data.get("results", [])[:10]:
            place_id = place.get("place_id")
            name = place.get("name", "Unknown")
            rating = float(place.get("rating", 0))

            geometry = place.get("geometry", {})
            location = geometry.get("location", {})
            place_lat = location.get("lat", hotel_lat if hotel_lat else central_lat)
            place_lon = location.get("lng", hotel_lon if hotel_lon else central_lon)

            distance = haversine_distance(hotel_lat if hotel_lat else central_lat, hotel_lon if hotel_lon else central_lon, place_lat, place_lon)

            details_url = "https://maps.googleapis.com/maps/api/place/details/json"
            details_params = {"place_id": place_id, "fields": "name,rating,reviews", "key": api_key}
            details_response = requests.get(details_url, params=details_params, timeout=10)
            details_response.raise_for_status()
            details_data = details_response.json()

            if details_data.get("status") != "OK":
                logger.error(f"Google Place Details Error (Restaurants) for {name}: {details_data.get('error_message', 'Unknown error')}")
                continue

            reviews = details_data.get("result", {}).get("reviews", [])
            review_texts = [review.get("text", "") for review in reviews[:2]]

            restaurant = {
                "name": name,
                "rating": rating,
                "distance": distance,
                "reviews": review_texts
            }
            restaurants.append(restaurant)
            details = {
                "reviews": review_texts,
                "coordinates": {"lat": place_lat, "lng": place_lon}
            }
            db.insert_restaurant(name, rating, distance, destination, place_lat, place_lon, review_texts)

        if not restaurants:
            logger.warning(f"No restaurants found for {destination}, using fallback")
            return fallback_restaurants

        logger.debug(f"Returning {len(restaurants)} restaurants for {destination}")
        return restaurants

    except requests.exceptions.RequestException as e:
        logger.error(f"Google Places API Error (Restaurants) for {destination}: {e}")
        return fallback_restaurants

def fetch_weather(lat, lon, pick_up_date, drop_off_date):
    api_key = os.getenv("OPENWEATHERMAP_API_KEY")
    logger.info(f"Fetching weather for lat={lat}, lon={lon} in getgetplaces")
    
    fallback_weather = {"forecast": "Clear", "temperature": 25, "humidity": 60}
    if not api_key:
        logger.warning("OPENWEATHERMAP_API_KEY not set, using fallback weather")
        return fallback_weather

    url = f"http://api.openweathermap.org/data/2.5/forecast"
    params = {"lat": lat, "lon": lon, "appid": api_key, "units": "metric"}
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        logger.debug(f"Weather API response: {data}")

        if data.get("cod") != "200":
            logger.error(f"OpenWeatherMap API Error: {data.get('message', 'Unknown error')}")
            return fallback_weather

        weather_data = {}
        for entry in data.get("list", []):
            dt = datetime.fromtimestamp(entry["dt"])
            if pick_up_date <= dt <= drop_off_date:
                weather_data[dt.strftime("%Y-%m-%d")] = {
                    "forecast": entry["weather"][0]["main"],
                    "temperature": entry["main"]["temp"],
                    "humidity": entry["main"]["humidity"]
                }
                db.insert_weather("", dt.strftime("%Y-%m-%d"), entry["weather"][0]["main"], entry["main"]["temp"], entry["main"]["humidity"])
        return weather_data

    except requests.exceptions.RequestException as e:
        logger.error(f"OpenWeatherMap API Error: {e}")
        return fallback_weather

def recommend_hotels(destination, budget, pick_up_date, drop_off_date, user_id=None):
    hotels = fetch_hotels(destination, budget, pick_up_date, drop_off_date)
    logger.debug(f"Hotels received for {destination}: {len(hotels)}")
    for h in hotels:
        predicted_rating = recommendation_model.predict_preference(h["price"], h["distance"])
        h["score"] = 0.4 * predicted_rating + 0.3 * (budget - h["price"]) / budget + 0.3 * (5 - h["distance"]) / 5
    top_hotels = sorted(hotels, key=lambda x: x["score"], reverse=True)[:2]
    logger.debug(f"Top hotels for {destination}: {top_hotels}")
    try:
        central_lat, central_lon = get_coordinates(destination)
    except ValueError as e:
        logger.error(f"Failed to fetch coordinates for {destination} in recommend_hotels: {e}")
        central_lat, central_lon = (0, 0)
    hotel_coords = (top_hotels[0]["geometry"]["location"]["lat"] if top_hotels and "geometry" in top_hotels[0] else central_lat,
                    top_hotels[0]["geometry"]["location"]["lng"] if top_hotels and "geometry" in top_hotels[0] else central_lon) if top_hotels else (central_lat, central_lon)
    return top_hotels, hotel_coords

def recommend_cars(destination, budget, pick_up_date, drop_off_date, hotel_lat=0, hotel_lon=0, pick_up_time="10:00", drop_off_time="10:00"):
    cars = fetch_cars(destination, budget, pick_up_date, drop_off_date, hotel_lat, hotel_lon, pick_up_time, drop_off_time)
    logger.debug(f"Cars received for {destination}: {len(cars)}")
    for c in cars:
        predicted_rating = recommendation_model.predict_preference(c["price"], c["distance"])
        c["score"] = 0.4 * predicted_rating + 0.3 * (budget - c["price"]) / budget + 0.3 * (5 - c["distance"]) / 5
    top_cars = sorted(cars, key=lambda x: x["score"], reverse=True)[:1]
    logger.debug(f"Top car for {destination}: {top_cars}")
    return top_cars

def recommend_attractions(destination, pick_up_date, drop_off_date, hotel_lat=0, hotel_lon=0, prefer_indoor=False):
    attractions = fetch_attractions(destination, pick_up_date, drop_off_date, hotel_lat, hotel_lon)
    logger.debug(f"Attractions received for {destination}: {len(attractions)}")
    if prefer_indoor:
        attractions = [a for a in attractions if a.get("is_indoor", False)]
        if not attractions:
            logger.warning(f"No indoor attractions found for {destination}, falling back to all attractions")
            attractions = fetch_attractions(destination, pick_up_date, drop_off_date, hotel_lat, hotel_lon)

    for a in attractions:
        a["score"] = 0.5 * a["rating"] + 0.5 * (5 - a["distance"]) / 5 + 0.2 * a["image_score"]
    top_attractions = sorted(attractions, key=lambda x: x["score"], reverse=True)[:3]
    logger.debug(f"Top attractions for {destination}: {top_attractions}")
    return top_attractions

def recommend_restaurants(destination, pick_up_date, drop_off_date, hotel_lat=0, hotel_lon=0):
    restaurants = fetch_restaurants(destination, pick_up_date, drop_off_date, hotel_lat, hotel_lon)
    logger.debug(f"Restaurants received for {destination}: {len(restaurants)}")
    for r in restaurants:
        r["score"] = 0.5 * r["rating"] + 0.5 * (5 - r["distance"]) / 5
    top_restaurants = sorted(restaurants, key=lambda x: x["score"], reverse=True)[:3]
    logger.debug(f"Top restaurants for {destination}: {top_restaurants}")
    return top_restaurants

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            text_input = request.form.get("text_input", "").strip()
            if text_input:
                destination, budget, preferences = parse_nlp_input(text_input)
                if not destination or not budget:
                    return "Error: Could not parse destination or budget from input.", 400
                destinations = [destination]
                pick_up_date = datetime.strptime("2025-03-10", "%Y-%m-%d")
                drop_off_date = datetime.strptime("2025-03-14", "%Y-%m-%d")
                pick_up_time = "10:00"
                drop_off_time = "10:00"
            else:
                destinations_input = request.form.get("destinations", "").strip()
                if not destinations_input:
                    return "Error: Destinations field is required.", 400

                destinations = [d.strip().capitalize() for d in destinations_input.split(",") if d.strip()]
                if not destinations:
                    return "Error: At least one valid destination is required.", 400

                budget_str = request.form.get("budget", "").strip()
                if not budget_str:
                    return "Error: Budget field is required.", 400
                budget = float(budget_str) or 500.0  # Default to $500 if not provided
                if budget <= 0:
                    return "Error: Budget must be a positive number.", 400

                pick_up_date_str = request.form.get("pickUpDate", "").strip()
                drop_off_date_str = request.form.get("dropOffDate", "").strip()
                pick_up_time = request.form.get("pickUpTime", "10:00").strip()
                drop_off_time = request.form.get("dropOffTime", "10:00").strip()
                if not pick_up_date_str or not drop_off_date_str:
                    return "Error: Pick-up and drop-off dates are required.", 400

                pick_up_date = datetime.strptime(pick_up_date_str, "%Y-%m-%d")
                drop_off_date = datetime.strptime(drop_off_date_str, "%Y-%m-%d")

            hotels_by_city = {}
            attractions_by_city = {}
            restaurants_by_city = {}
            weather_by_city = {}

            first_city = destinations[0]
            hotels, hotel_coords = recommend_hotels(first_city, budget, pick_up_date, drop_off_date)
            hotels_by_city[first_city] = hotels
            cars = recommend_cars(first_city, budget, pick_up_date, drop_off_date, hotel_coords[0], hotel_coords[1], pick_up_time, drop_off_time)

            for city in destinations:
                hotels, hotel_coords = recommend_hotels(city, budget, pick_up_date, drop_off_date)
                hotels_by_city[city] = hotels if hotels else [{"name": "Placeholder Hotel", "price": 100.0, "rating": 4.0, "distance": 1.0, "reviews": ["Default review"], "geometry": {"location": {"lat": 0, "lng": 0}}}]
                weather = fetch_weather(hotel_coords[0], hotel_coords[1], pick_up_date, drop_off_date)
                weather_by_city[city] = weather
                attractions = recommend_attractions(city, pick_up_date, drop_off_date, hotel_coords[0], hotel_coords[1])
                attractions_by_city[city] = attractions if attractions else [{"name": "Placeholder Attraction", "rating": 4.0, "distance": 1.0, "reviews": ["Default review"], "is_indoor": False, "image_score": 0}]
                restaurants = recommend_restaurants(city, pick_up_date, drop_off_date, hotel_coords[0], hotel_coords[1])
                restaurants_by_city[city] = restaurants if restaurants else [{"name": "Placeholder Restaurant", "rating": 4.0, "distance": 1.0, "reviews": ["Default review"]}]

            itinerary, cost_summary = generate_itinerary(destinations, pick_up_date, drop_off_date, hotels_by_city, cars, attractions_by_city, restaurants_by_city, weather_by_city, budget, pick_up_time, drop_off_time)

            result = ""
            for city in destinations:
                result += f"Top Hotels in {city}:<br>"
                for h in hotels_by_city[city]:
                    result += f"{h['name']} (${h['price']:.1f}) - Rating: {h['rating']} - Distance: {h['distance']:.1f} km<br>"
                    result += "Reviews:<br>"
                    for review in h['reviews']:
                        result += f"- {review}<br>"
                    result += "<br>"

            result += "Top Car Rental (Picked up in First City):<br>"
            for c in cars:
                result += f"{c['name']} (${c['price']:.1f}) - Company: {c['company']} - Distance: {c['distance']:.1f} km<br>"
                result += "<br>"

            for city in destinations:
                result += f"Top Attractions in {city}:<br>"
                for a in attractions_by_city[city]:
                    result += f"{a['name']} - Rating: {a['rating']} - Distance: {a['distance']:.1f} km<br>"
                    result += "Reviews:<br>"
                    for review in a['reviews']:
                        result += f"- {review}<br>"
                    result += "<br>"

            for city in destinations:
                result += f"Top Restaurants in {city}:<br>"
                for r in restaurants_by_city[city]:
                    result += f"{r['name']} - Rating: {r['rating']} - Distance: {r['distance']:.1f} km<br>"
                    result += "Reviews:<br>"
                    for review in r['reviews']:
                        result += f"- {review}<br>"
                    result += "<br>"

            result += "Travel Itinerary:<br><pre>" + itinerary + "\n\n" + cost_summary + "</pre><br>"
            return result

        except Exception as e:
            logger.error(f"Error processing request in getgetplaces: {e}")
            return f"Error: {str(e)}", 400

    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    message = request.json.get("message")
    response = chatbot.handle_message(message)
    return {"response": response}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)