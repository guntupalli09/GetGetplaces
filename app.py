from flask import Flask, request, render_template
import logging
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import requests
import json
import pandas as pd

from utils.api import get_coordinates, get_airport_code
from utils.database import Database
from utils.weather import fetch_weather
from utils.distance import haversine_distance
from models.recommendation import RecommendationModel
from models.price_predictor import PricePredictor
from nlp.parser import parse_nlp_input
from chatbot.bot import Chatbot
from vision.image_scorer import score_image

# Set up logging with more granularity
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Output to console
    ]
)
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
    logger.debug(f"Loaded GOOGLE_PLACES_API_KEY: {api_key}")
    logger.info(f"Fetching hotels for {destination} with budget {budget}")
    
    if not api_key:
        logger.error("GOOGLE_PLACES_API_KEY not set")
        raise ValueError("API key missing for hotel search")

    params = {"query": f"hotels in {destination}", "type": "lodging", "key": api_key}
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        logger.debug(f"Hotel API response for {destination}: {data}")

        if data.get("status") != "OK":
            logger.error(f"API error response: {data.get('error_message')}")
            raise ValueError(f"Error fetching hotels: {data.get('error_message')}")

        return process_hotel_data(data, destination, budget, pick_up_date)
    except requests.HTTPError as http_err:
        logger.error(f"HTTP error occurred: {http_err}")
        raise
    except requests.RequestException as req_err:
        logger.error(f"Request exception occurred: {req_err}")
        raise
    except Exception as e:
        logger.error(f"An error occurred in fetch_hotels: {e}", exc_info=True)
        raise

def process_hotel_data(data, destination, budget, pick_up_date):
    logger.info(f"Processing hotel data for {destination} with budget {budget}")
    hotels = []
    try:
        central_lat, central_lon = get_coordinates(destination)
        logger.debug(f"Coordinates for {destination}: lat={central_lat}, lon={central_lon}")
        if isinstance(central_lat, pd.Series) or isinstance(central_lon, pd.Series):
            logger.error(f"Coordinates for {destination} returned as Series: lat={central_lat}, lon={central_lon}")
            raise ValueError("Coordinates must be scalar values")
        central_lat, central_lon = float(central_lat), float(central_lon)
    except ValueError as e:
        logger.error(f"Failed to fetch coordinates for {destination}: {e}")
        raise

    # Ensure pick_up_date is a datetime object
    if not isinstance(pick_up_date, datetime):
        logger.error(f"pick_up_date is not a datetime object: {type(pick_up_date)}")
        raise ValueError(f"pick_up_date must be a datetime object, got {type(pick_up_date)}")

    for place in data.get("results", [])[:10]:
        name = place.get("name", "Unknown")
        rating = float(place.get("rating", 0))
        price_level = int(place.get("price_level", 2))
        logger.debug(f"Processing hotel {name}: price_level={price_level}, rating={rating}")

        # Get the estimated price
        try:
            estimated_price = price_predictor.predict_price(price_level * 50, pick_up_date)
            logger.debug(f"Estimated price for {name}: {estimated_price}, type: {type(estimated_price)}")
        except Exception as e:
            logger.error(f"Error predicting price for {name}: {e}")
            estimated_price = price_level * 50  # Fallback to base price
            logger.warning(f"Using base price {estimated_price} for {name} due to prediction error")

        # Handle different types of estimated_price (just in case)
        if isinstance(estimated_price, pd.Series):
            logger.warning(f"Estimated price for {name} is a pandas Series: {estimated_price}")
            if estimated_price.empty:
                logger.error(f"Estimated price Series for {name} is empty, skipping hotel")
                continue
            elif len(estimated_price) > 1:
                logger.warning(f"Estimated price Series for {name} has multiple values: {estimated_price}")
                estimated_price = estimated_price.iloc[0]
                logger.debug(f"Using first value from Series: {estimated_price}")
            else:
                estimated_price = estimated_price.item()
                logger.debug(f"Converted estimated price to scalar: {estimated_price}")

            # Ensure the value is numeric
            try:
                estimated_price = float(estimated_price)
            except (ValueError, TypeError) as e:
                logger.error(f"Estimated price for {name} is not a valid number: {estimated_price}, error: {e}")
                continue
        elif isinstance(estimated_price, (int, float)):
            logger.debug(f"Estimated price for {name} is already a scalar: {estimated_price}")
        else:
            logger.error(f"Unexpected type for estimated price for {name}: {type(estimated_price)}")
            continue

        if estimated_price <= budget:
            logger.debug(f"Hotel {name} price {estimated_price} is within budget {budget}")
            geometry = place.get("geometry", {}).get("location", {})
            place_lat = geometry.get("lat", central_lat)
            place_lon = geometry.get("lng", central_lon)
            logger.debug(f"Hotel {name} coordinates: lat={place_lat}, lon={place_lon}")

            try:
                distance = haversine_distance(central_lat, central_lon, place_lat, place_lon)
                logger.debug(f"Calculated distance for {name}: {distance} km")
            except Exception as e:
                logger.error(f"Error calculating distance for {name}: {e}")
                continue

            reviews = place.get("reviews", [])
            review_texts = [review.get("text", "") for review in reviews[:2]]
            try:
                json_reviews = json.dumps(review_texts)
                logger.debug(f"Reviews for {name}: {json_reviews} (size: {len(json_reviews)} characters)")
            except ValueError as e:
                logger.error(f"Invalid JSON for reviews in hotel {name}: {e}")
                review_texts = ["Invalid review data"]
                json_reviews = json.dumps(review_texts)

            try:
                db.insert_hotel(name, estimated_price, rating, distance, destination, place_lat, place_lon, review_texts)
                logger.debug(f"Successfully inserted hotel {name} into database")
            except Exception as e:
                logger.error(f"Failed to insert hotel {name} into database: {e}", exc_info=True)
                logger.warning(f"Continuing with in-memory data for {name} despite DB failure")

            hotels.append({
                "name": name,
                "price": estimated_price,
                "rating": rating,
                "distance": distance,
                "city": destination,
                "lat": place_lat,
                "long": place_lon,
                "reviews": review_texts
            })
        else:
            logger.debug(f"Hotel {name} price {estimated_price} exceeds budget {budget}, skipping")

    logger.info(f"{len(hotels)} hotels found within budget for {destination}")
    if not hotels:
        logger.warning(f"No hotels found within budget {budget} for {destination}")
    return hotels

def fetch_cars(destination, budget, pick_up_date, drop_off_date, hotel_lat=0, hotel_lon=0, pick_up_time="10:00", drop_off_time="10:00"):
    url = "https://priceline-com2.p.rapidapi.com/cars/search"
    api_key = os.getenv("RAPIDAPI_KEY_PRICELINE")
    logger.debug(f"Loaded RAPIDAPI_KEY_PRICELINE: {api_key}")
    logger.info(f"Fetching cars for {destination} with budget {budget}")
    
    if not api_key:
        logger.error("RAPIDAPI_KEY_PRICELINE not set")
        return []

    headers = {"X-RapidAPI-Key": api_key, "X-RapidAPI-Host": "priceline-com2.p.rapidapi.com"}

    try:
        central_lat, central_lon = get_coordinates(destination)
        logger.debug(f"Coordinates for {destination}: lat={central_lat}, lon={central_lon}")
        if isinstance(central_lat, pd.Series) or isinstance(central_lon, pd.Series):
            logger.error(f"Coordinates for {destination} returned as Series: lat={central_lat}, lon={central_lon}")
            return []
        central_lat, central_lon = float(central_lat), float(central_lon)
    except ValueError as e:
        logger.error(f"Failed to fetch coordinates for {destination}: {e}")
        return []

    airport_code = get_airport_code(destination)
    location = airport_code if airport_code else f"{central_lat},{central_lon}"
    logger.debug(f"Using location for car search: {location}")

    params = {
        "pickUpLocation": location,
        "pickUpDate": pick_up_date.strftime("%Y-%m-%d"),
        "pickUpTime": pick_up_time,
        "dropOffLocation": location,
        "dropOffDate": drop_off_date.strftime("%Y-%m-%d"),
        "dropOffTime": drop_off_time
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        response.raise_for_status()
        json_data = response.json()
        logger.debug(f"Car API response for {destination}: {json_data}")

        if not isinstance(json_data, dict):
            logger.error(f"Priceline API Error: Response is not a dictionary - {json_data}")
            return []

        vehicles = json_data.get("data", [])
        if not vehicles:
            logger.warning(f"No cars found for {destination} on {pick_up_date.strftime('%Y-%m-%d')} to {drop_off_date.strftime('%Y-%m-%d')}")
            return []

        cars = []
        for vehicle in vehicles[:10]:
            price = vehicle.get("price", 0)
            logger.debug(f"Car price: {price}, type: {type(price)}")

            if isinstance(price, pd.Series):
                logger.warning(f"Car price is a pandas Series: {price}")
                if price.empty:
                    logger.error("Car price Series is empty, skipping vehicle")
                    continue
                elif len(price) > 1:
                    logger.warning(f"Car price Series has multiple values: {price}, using first value")
                    price = price.iloc[0]
                else:
                    price = price.item()
            price = float(price)

            logger.debug(f"Car price after conversion: ${price}, budget: ${budget * 1.2}")
            if price > budget * 1.2:
                logger.debug(f"Car ${price} exceeds budget {budget * 1.2}, skipping")
                continue

            vehicle_lat = vehicle.get("pickUpLocation", {}).get("latitude", hotel_lat if hotel_lat else central_lat)
            vehicle_lon = vehicle.get("pickUpLocation", {}).get("longitude", hotel_lon if hotel_lon else central_lon)
            logger.debug(f"Car coordinates: lat={vehicle_lat}, lon={vehicle_lon}")

            try:
                distance = haversine_distance(hotel_lat if hotel_lat else central_lat, hotel_lon if hotel_lon else central_lon, vehicle_lat, vehicle_lon)
                logger.debug(f"Calculated distance for car: {distance} km")
            except Exception as e:
                logger.error(f"Error calculating distance for car: {e}")
                continue

            car = {
                "name": vehicle.get("vehicleName", "Unknown Car"),
                "price": price,
                "rating": float(vehicle.get("rating", 0)),
                "distance": distance,
                "company": vehicle.get("company", "Unknown"),
                "reviews": vehicle.get("reviews", []) if vehicle.get("reviews") else [],
                "lat": vehicle_lat,
                "long": vehicle_lon
            }
            cars.append(car)

            try:
                db.insert_car(car["name"], car["price"], car["rating"], car["distance"], car["company"], destination, car["reviews"])
                logger.debug(f"Successfully inserted car {car['name']} into database")
            except Exception as e:
                logger.error(f"Failed to insert car {car['name']} into database: {e}", exc_info=True)
                logger.warning(f"Continuing with in-memory data for car {car['name']} despite DB failure")

        logger.debug(f"Returning {len(cars)} cars for {destination}")
        if not cars:
            logger.warning(f"No cars found within budget {budget * 1.2} for {destination}")
        return cars

    except requests.HTTPError as http_err:
        logger.error(f"HTTP error occurred: {http_err}")
        return []
    except requests.RequestException as req_err:
        logger.error(f"Request exception occurred: {req_err}")
        return []
    except Exception as e:
        logger.error(f"An error occurred in fetch_cars: {e}", exc_info=True)
        return []

def fetch_attractions(destination, pick_up_date, drop_off_date, hotel_lat=0, hotel_lon=0):
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    api_key = os.getenv("GOOGLE_PLACES_API_KEY")
    logger.debug(f"Loaded GOOGLE_PLACES_API_KEY: {api_key}")
    logger.info(f"Fetching attractions for {destination}")
    
    if not api_key:
        logger.error("GOOGLE_PLACES_API_KEY not set")
        raise ValueError("API key missing for attraction search")

    try:
        central_lat, central_lon = get_coordinates(destination)
        logger.debug(f"Coordinates for {destination}: lat={central_lat}, lon={central_lon}")
        if isinstance(central_lat, pd.Series) or isinstance(central_lon, pd.Series):
            logger.error(f"Coordinates for {destination} returned as Series: lat={central_lat}, lon={central_lon}")
            raise ValueError("Coordinates must be scalar values")
        central_lat, central_lon = float(central_lat), float(central_lon)
    except ValueError as e:
        logger.error(f"Failed to fetch coordinates for {destination}: {e}")
        raise

    params = {"query": f"attractions in {destination}", "type": "tourist_attraction", "key": api_key}
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        logger.debug(f"Attraction API response for {destination}: {data}")

        if not isinstance(data, dict) or data.get("status") != "OK":
            logger.error(f"Google Places API Error (Attractions) for {destination}: {data.get('error_message', 'Unknown error')}")
            raise ValueError("Attraction search failed")

        attractions = []
        for place in data.get("results", [])[:10]:
            place_id = place.get("place_id")
            name = place.get("name", "Unknown")
            rating = float(place.get("rating", 0))
            geometry = place.get("geometry", {})
            location = geometry.get("location", {})
            place_lat = location.get("lat", hotel_lat if hotel_lat else central_lat)
            place_lon = location.get("lng", hotel_lon if hotel_lon else central_lon)
            logger.debug(f"Attraction {name} coordinates: lat={place_lat}, lon={place_lon}")

            try:
                distance = haversine_distance(hotel_lat if hotel_lat else central_lat, hotel_lon if hotel_lon else central_lon, place_lat, place_lon)
                logger.debug(f"Calculated distance for {name}: {distance} km")
            except Exception as e:
                logger.error(f"Error calculating distance for {name}: {e}")
                continue

            details_url = "https://maps.googleapis.com/maps/api/place/details/json"
            details_params = {"place_id": place_id, "fields": "name,rating,reviews,types,photos", "key": api_key}
            details_response = requests.get(details_url, params=details_params, timeout=15)
            details_response.raise_for_status()
            details_data = details_response.json()
            logger.debug(f"Attraction details response for {name}: {details_data}")

            if details_data.get("status") != "OK":
                logger.error(f"Google Place Details Error (Attractions) for {name}: {details_data.get('error_message', 'Unknown error')}")
                continue

            reviews = details_data.get("result", {}).get("reviews", [])
            review_texts = [review.get("text", "") for review in reviews[:2]]
            types = details_data.get("result", {}).get("types", [])
            is_indoor = any(t in ["museum", "gallery", "indoor"] for t in types)
            logger.debug(f"Attraction {name} is_indoor: {is_indoor}")

            image_score = 0
            if "photos" in details_data.get("result", {}) and details_data["result"]["photos"]:
                photo_ref = details_data["result"]["photos"][0]["photo_reference"]
                photo_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={photo_ref}&key={api_key}"
                try:
                    image_score = score_image(photo_url)
                    logger.debug(f"Image score for {name}: {image_score}, type: {type(image_score)}")
                    if isinstance(image_score, pd.Series):
                        logger.warning(f"Image score for {name} is a pandas Series: {image_score}")
                        if image_score.empty:
                            logger.error(f"Image score Series for {name} is empty")
                            image_score = 0
                        elif len(image_score) > 1:
                            logger.warning(f"Image score Series for {name} has multiple values: {image_score}, using first value")
                            image_score = image_score.iloc[0]
                        else:
                            image_score = image_score.item()
                    image_score = float(image_score)
                except Exception as e:
                    logger.error(f"Error scoring image for {name}: {e}")
                    image_score = 0

            attraction = {
                "name": name,
                "rating": rating,
                "distance": distance,
                "reviews": review_texts,
                "is_indoor": is_indoor,
                "image_score": image_score,
                "lat": place_lat,
                "long": place_lon
            }
            attractions.append(attraction)

            try:
                db.insert_attraction(name, rating, distance, destination, place_lat, place_lon, review_texts, is_indoor, image_score)
                logger.debug(f"Successfully inserted attraction {name} into database")
            except Exception as e:
                logger.error(f"Failed to insert attraction {name} into database: {e}", exc_info=True)
                logger.warning(f"Continuing with in-memory data for attraction {name} despite DB failure")

        logger.debug(f"Returning {len(attractions)} attractions for {destination}")
        if not attractions:
            logger.warning(f"No attractions found for {destination}")
        return attractions

    except requests.HTTPError as http_err:
        logger.error(f"HTTP error occurred: {http_err}")
        raise
    except requests.RequestException as req_err:
        logger.error(f"Request exception occurred: {req_err}")
        raise
    except Exception as e:
        logger.error(f"An error occurred in fetch_attractions: {e}", exc_info=True)
        raise

def fetch_restaurants(destination, pick_up_date, drop_off_date, hotel_lat=0, hotel_lon=0):
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    api_key = os.getenv("GOOGLE_PLACES_API_KEY")
    logger.debug(f"Loaded GOOGLE_PLACES_API_KEY: {api_key}")
    logger.info(f"Fetching restaurants for {destination}")
    
    if not api_key:
        logger.error("GOOGLE_PLACES_API_KEY not set")
        raise ValueError("API key missing for restaurant search")

    try:
        central_lat, central_lon = get_coordinates(destination)
        logger.debug(f"Coordinates for {destination}: lat={central_lat}, lon={central_lon}")
        if isinstance(central_lat, pd.Series) or isinstance(central_lon, pd.Series):
            logger.error(f"Coordinates for {destination} returned as Series: lat={central_lat}, lon={central_lon}")
            raise ValueError("Coordinates must be scalar values")
        central_lat, central_lon = float(central_lat), float(central_lon)
    except ValueError as e:
        logger.error(f"Failed to fetch coordinates for {destination}: {e}")
        raise

    params = {"query": f"restaurants in {destination}", "type": "restaurant", "key": api_key}
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        logger.debug(f"Restaurant API response for {destination}: {data}")

        if not isinstance(data, dict) or data.get("status") != "OK":
            logger.error(f"Google Places API Error (Restaurants) for {destination}: {data.get('error_message', 'Unknown error')}")
            raise ValueError("Restaurant search failed")

        restaurants = []
        for place in data.get("results", [])[:10]:
            place_id = place.get("place_id")
            name = place.get("name", "Unknown")
            rating = float(place.get("rating", 0))
            geometry = place.get("geometry", {})
            location = geometry.get("location", {})
            place_lat = location.get("lat", hotel_lat if hotel_lat else central_lat)
            place_lon = location.get("lng", hotel_lon if hotel_lon else central_lon)
            logger.debug(f"Restaurant {name} coordinates: lat={place_lat}, lon={place_lon}")

            try:
                distance = haversine_distance(hotel_lat if hotel_lat else central_lat, hotel_lon if hotel_lon else central_lon, place_lat, place_lon)
                logger.debug(f"Calculated distance for {name}: {distance} km")
            except Exception as e:
                logger.error(f"Error calculating distance for {name}: {e}")
                continue

            details_url = "https://maps.googleapis.com/maps/api/place/details/json"
            details_params = {"place_id": place_id, "fields": "name,rating,reviews", "key": api_key}
            details_response = requests.get(details_url, params=details_params, timeout=15)
            details_response.raise_for_status()
            details_data = details_response.json()
            logger.debug(f"Restaurant details response for {name}: {details_data}")

            if details_data.get("status") != "OK":
                logger.error(f"Google Place Details Error (Restaurants) for {name}: {details_data.get('error_message', 'Unknown error')}")
                continue

            reviews = details_data.get("result", {}).get("reviews", [])
            review_texts = [review.get("text", "") for review in reviews[:2]]

            restaurant = {
                "name": name,
                "rating": rating,
                "distance": distance,
                "reviews": review_texts,
                "lat": place_lat,
                "long": place_lon
            }
            restaurants.append(restaurant)

            try:
                db.insert_restaurant(name, rating, distance, destination, place_lat, place_lon, review_texts)
                logger.debug(f"Successfully inserted restaurant {name} into database")
            except Exception as e:
                logger.error(f"Failed to insert restaurant {name} into database: {e}", exc_info=True)
                logger.warning(f"Continuing with in-memory data for restaurant {name} despite DB failure")

        logger.debug(f"Returning {len(restaurants)} restaurants for {destination}")
        if not restaurants:
            logger.warning(f"No restaurants found for {destination}")
        return restaurants

    except requests.HTTPError as http_err:
        logger.error(f"HTTP error occurred: {http_err}")
        raise
    except requests.RequestException as req_err:
        logger.error(f"Request exception occurred: {req_err}")
        raise
    except Exception as e:
        logger.error(f"An error occurred in fetch_restaurants: {e}", exc_info=True)
        raise

def generate_plans(hotels, cars, attractions, restaurants, destination, pick_up_date, drop_off_date, budget, preferred_days=None):
    """
    Generate 2-3 daily plans for the given date range, ensuring total cost is within budget or up to 20% more.
    
    Args:
        hotels (list): List of hotel dictionaries.
        cars (list): List of car dictionaries.
        attractions (list): List of attraction dictionaries.
        restaurants (list): List of restaurant dictionaries.
        destination (str): The destination city.
        pick_up_date (datetime): Start date of the trip.
        drop_off_date (datetime): End date of the trip.
        budget (float): User's budget.
        preferred_days (int, optional): Number of days the user wants the plan for (e.g., 1 for a 1-day plan).
    
    Returns:
        list: List of plans, each containing a day-by-day schedule.
    """
    logger.info(f"Generating plans for {destination} from {pick_up_date} to {drop_off_date} with budget {budget}")

    # Get coordinates for the destination to fetch weather
    try:
        dest_lat, dest_lon = get_coordinates(destination)
        logger.debug(f"Coordinates for {destination}: lat={dest_lat}, lon={dest_lon}")
        if isinstance(dest_lat, pd.Series) or isinstance(dest_lon, pd.Series):
            logger.error(f"Coordinates for {destination} returned as Series: lat={dest_lat}, lon={dest_lon}")
            raise ValueError("Coordinates must be scalar values")
        dest_lat, dest_lon = float(dest_lat), float(dest_lon)
    except ValueError as e:
        logger.error(f"Failed to fetch coordinates for {destination}: {e}")
        dest_lat, dest_lon = 0, 0  # Fallback coordinates (weather fetch will fail gracefully)

    # Calculate the total number of days in the trip
    total_days = (drop_off_date - pick_up_date).days + 1
    logger.debug(f"Total days in trip: {total_days}")

    # If the user specified a preferred number of days, use that; otherwise, generate plans for 1, 2, and 3 days
    if preferred_days:
        if preferred_days > total_days:
            logger.warning(f"Preferred days ({preferred_days}) exceeds total trip days ({total_days}). Using total days.")
            preferred_days = total_days
        plan_lengths = [preferred_days]
    else:
        # Generate plans for 1, 2, and 3 days (or fewer if the trip is shorter)
        plan_lengths = [1, 2, 3]
        plan_lengths = [length for length in plan_lengths if length <= total_days]
        if not plan_lengths:
            logger.error("Trip duration is less than 1 day. Cannot generate plans.")
            return []

    # Calculate the maximum allowable budget (budget + 20%)
    max_budget = budget * 1.2
    logger.debug(f"Maximum allowable budget (budget + 20%): ${max_budget}")

    # Select a hotel (use the first one within budget)
    selected_hotel = None
    for hotel in hotels:
        if hotel["price"] <= budget:
            selected_hotel = hotel
            break
    if not selected_hotel:
        logger.warning("No hotels found within budget. Using the cheapest hotel.")
        selected_hotel = min(hotels, key=lambda x: x["price"], default=None)
        if not selected_hotel:
            logger.error("No hotels available to generate plans.")
            return []

    hotel_cost_per_night = selected_hotel["price"]
    hotel_lat = selected_hotel["lat"]
    hotel_lon = selected_hotel["long"]
    logger.debug(f"Selected hotel: {selected_hotel['name']}, cost per night: ${hotel_cost_per_night}")

    # Select a car (use the first one within budget)
    selected_car = None
    for car in cars:
        # Estimate car cost for the entire trip (assuming daily rate)
        car_daily_rate = car["price"]
        car_total_cost = car_daily_rate * total_days
        if car_total_cost <= budget * 0.3:  # Allocate 30% of budget to car
            selected_car = car
            break
    if not selected_car:
        logger.warning("No cars found within budget. Using the cheapest car.")
        selected_car = min(cars, key=lambda x: x["price"], default=None)
        if not selected_car:
            logger.warning("No cars available. Plans will not include car travel.")
            selected_car = None
        else:
            car_daily_rate = selected_car["price"]
            logger.debug(f"Selected car: {selected_car['name']}, daily rate: ${car_daily_rate}")

    # Sort attractions and restaurants by rating and distance
    attractions = sorted(attractions, key=lambda x: (-x["rating"], x["distance"]))
    restaurants = sorted(restaurants, key=lambda x: (-x["rating"], x["distance"]))

    # Generate plans for each plan length
    plans = []
    for plan_days in plan_lengths:
        logger.debug(f"Generating plan for {plan_days} days")

        # Calculate costs for this plan
        hotel_cost = hotel_cost_per_night * plan_days
        car_cost = selected_car["price"] * plan_days if selected_car else 0
        remaining_budget = budget - hotel_cost - car_cost
        if remaining_budget < 0:
            logger.warning(f"Hotel and car costs (${hotel_cost + car_cost}) exceed budget (${budget}) for {plan_days}-day plan. Checking if within max budget.")
            if (hotel_cost + car_cost) > max_budget:
                logger.warning(f"Hotel and car costs (${hotel_cost + car_cost}) exceed max budget (${max_budget}) for {plan_days}-day plan. Skipping.")
                continue
            remaining_budget = max_budget - hotel_cost - car_cost

        # Allocate remaining budget for meals and activities
        meal_cost_per_day = 50  # Estimate $50 per day for meals
        total_meal_cost = meal_cost_per_day * plan_days
        if total_meal_cost > remaining_budget:
            logger.warning(f"Meal costs (${total_meal_cost}) exceed remaining budget (${remaining_budget}) for {plan_days}-day plan. Adjusting.")
            total_meal_cost = remaining_budget * 0.5  # Allocate 50% of remaining budget to meals
            remaining_budget -= total_meal_cost
        else:
            remaining_budget -= total_meal_cost

        # Estimate attraction costs (e.g., $20 per attraction)
        attraction_cost_per_visit = 20
        max_attractions = int(remaining_budget // attraction_cost_per_visit)
        logger.debug(f"Remaining budget after hotel, car, and meals: ${remaining_budget}, can afford {max_attractions} attractions")

        # Generate the daily schedule
        plan = {
            "days": plan_days,
            "total_cost": 0,
            "schedule": [],
            "hotel": selected_hotel,
            "car": selected_car
        }
        current_date = pick_up_date
        used_attractions = []
        used_restaurants = []

        for day in range(plan_days):
            daily_schedule = []
            daily_cost = hotel_cost_per_night  # Hotel cost for this day
            if selected_car:
                daily_cost += selected_car["price"]  # Car cost for this day

            # Get the date string for fetching weather
            date_str = current_date.strftime("%Y-%m-%d")

            # 10:00 AM - Pick up car (if available)
            if selected_car:
                schedule_entry = {
                    "time": "10:00 AM",
                    "activity": "Pick up car",
                    "details": {
                        "name": selected_car["name"],
                        "company": selected_car["company"],
                        "distance": selected_car["distance"],
                        "rating": selected_car["rating"],
                        "location": f"Coordinates: ({selected_car['lat']}, {selected_car['long']})",
                        "cost": selected_car["price"]
                    }
                }
                daily_schedule.append(schedule_entry)

            # 10:30 AM - Visit first attraction
            attraction1 = None
            for attr in attractions:
                if attr not in used_attractions and len(used_attractions) < max_attractions:
                    attraction1 = attr
                    used_attractions.append(attr)
                    break
            if attraction1:
                # Fetch weather for the current day (start_date and end_date are the same)
                weather_data = fetch_weather(dest_lat, dest_lon, current_date, current_date)
                weather = weather_data.get(date_str, "Weather unavailable")
                schedule_entry = {
                    "time": "10:30 AM",
                    "activity": "Visit attraction",
                    "details": {
                        "name": attraction1["name"],
                        "distance": attraction1["distance"],
                        "rating": attraction1["rating"],
                        "reviews": attraction1["reviews"],
                        "location": f"Coordinates: ({attraction1['lat']}, {attraction1['long']})",
                        "weather": weather,
                        "cost": attraction_cost_per_visit
                    }
                }
                daily_schedule.append(schedule_entry)
                daily_cost += attraction_cost_per_visit

            # 1:00 PM - Lunch at a restaurant
            restaurant = None
            for rest in restaurants:
                if rest not in used_restaurants:
                    restaurant = rest
                    used_restaurants.append(rest)
                    break
            if restaurant:
                # Fetch weather for the current day (start_date and end_date are the same)
                weather_data = fetch_weather(dest_lat, dest_lon, current_date, current_date)
                weather = weather_data.get(date_str, "Weather unavailable")
                meal_cost = min(meal_cost_per_day, total_meal_cost / plan_days)  # Distribute meal cost evenly
                schedule_entry = {
                    "time": "1:00 PM",
                    "activity": "Lunch",
                    "details": {
                        "name": restaurant["name"],
                        "distance": restaurant["distance"],
                        "rating": restaurant["rating"],
                        "reviews": restaurant["reviews"],
                        "location": f"Coordinates: ({restaurant['lat']}, {restaurant['long']})",
                        "weather": weather,
                        "cost": meal_cost
                    }
                }
                daily_schedule.append(schedule_entry)
                daily_cost += meal_cost

            # 3:00 PM - Visit second attraction
            attraction2 = None
            for attr in attractions:
                if attr not in used_attractions and len(used_attractions) < max_attractions:
                    attraction2 = attr
                    used_attractions.append(attr)
                    break
            if attraction2:
                # Fetch weather for the current day (start_date and end_date are the same)
                weather_data = fetch_weather(dest_lat, dest_lon, current_date, current_date)
                weather = weather_data.get(date_str, "Weather unavailable")
                schedule_entry = {
                    "time": "3:00 PM",
                    "activity": "Visit attraction",
                    "details": {
                        "name": attraction2["name"],
                        "distance": attraction2["distance"],
                        "rating": attraction2["rating"],
                        "reviews": attraction2["reviews"],
                        "location": f"Coordinates: ({attraction2['lat']}, {attraction2['long']})",
                        "weather": weather,
                        "cost": attraction_cost_per_visit
                    }
                }
                daily_schedule.append(schedule_entry)
                daily_cost += attraction_cost_per_visit

            # 7:00 PM - Return to hotel for dinner and sleep
            schedule_entry = {
                "time": "7:00 PM",
                "activity": "Return to hotel",
                "details": {
                    "name": selected_hotel["name"],
                    "distance": 0,  # Already at hotel
                    "rating": selected_hotel["rating"],
                    "reviews": selected_hotel["reviews"],
                    "location": f"Coordinates: ({hotel_lat}, {hotel_lon})",
                    "cost": 0  # Dinner cost included in hotel cost
                }
            }
            daily_schedule.append(schedule_entry)

            # Add the daily schedule to the plan
            plan["schedule"].append({
                "date": current_date.strftime("%Y-%m-%d"),
                "day": day + 1,
                "activities": daily_schedule,
                "daily_cost": daily_cost
            })
            plan["total_cost"] += daily_cost
            current_date += timedelta(days=1)

        # Check if the plan is within the maximum allowable budget (budget + 20%)
        if plan["total_cost"] <= max_budget:
            plans.append(plan)
        else:
            logger.warning(f"Plan for {plan_days} days exceeds maximum allowable budget (${plan['total_cost']} > ${max_budget}). Skipping.")

    logger.info(f"Generated {len(plans)} plans for {destination}")
    return plans

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            logger.info("Received POST request to /")
            logger.debug(f"Raw form data: {request.form.to_dict()}")

            text_input = request.form.get("text_input", "").strip()
            destinations = request.form.get("destinations", "").strip()
            logger.debug(f"Text input: {text_input}, Destinations: {destinations}")

            destination = None
            if text_input:
                logger.debug(f"Parsing text_input: {text_input}")
                parsed_destination, budget_from_text, preferences = parse_nlp_input(text_input)
                logger.debug(f"Parsed NLP input: destination={parsed_destination}, budget={budget_from_text}, preferences={preferences}")
                if parsed_destination:
                    destination = parsed_destination
                else:
                    logger.warning("Failed to parse destination from text_input")
            elif destinations:
                destination_list = [d.strip() for d in destinations.split(",") if d.strip()]
                logger.debug(f"Parsed destinations list: {destination_list}")
                if destination_list:
                    destination = destination_list[0]
                    logger.debug(f"Selected first destination: {destination}")
                else:
                    logger.warning("Destinations field is empty after parsing")

            if not destination:
                logger.error("No valid destination provided in text_input or destinations")
                return "Error: Please provide a destination to plan your trip (either via free-text input or destinations).", 400

            budget = request.form.get("budget", 0)
            pick_up_date_str = request.form.get("pickUpDate")
            pick_up_time = request.form.get("pickUpTime", "10:00")
            preferred_days = request.form.get("preferredDays")
            logger.debug(f"Form data: destination={destination}, budget={budget}, pickUpDate={pick_up_date_str}, pickUpTime={pick_up_time}, preferredDays={preferred_days}")

            try:
                budget = float(budget)
                logger.debug(f"Converted budget to float: {budget}")
            except (ValueError, TypeError) as e:
                logger.error(f"Invalid budget value: {budget}, error: {e}")
                return "Error: Budget must be a valid number", 400

            if budget <= 0:
                logger.error(f"Budget must be positive, got: {budget}")
                return "Error: Budget must be a positive number", 400

            if not pick_up_date_str:
                logger.error("Pick-up date is missing")
                return "Error: Start date is required", 400

            try:
                pick_up_date = datetime.strptime(pick_up_date_str, "%Y-%m-%d")
                logger.debug(f"Parsed start date: pickUpDate={pick_up_date}")
            except ValueError as e:
                logger.error(f"Invalid date format: pickUpDate={pick_up_date_str}, error: {e}")
                return "Error: Start date must be in YYYY-MM-DD format", 400

            # Validate that pick-up date is not in the past
            current_date = datetime.strptime("2025-03-25", "%Y-%m-%d")  # Current date as per system
            if pick_up_date < current_date:
                logger.error(f"Pick-up date {pick_up_date} is in the past (current date: {current_date})")
                return "Error: Start date must be today or in the future", 400

            # Convert preferred_days to integer (required field)
            try:
                preferred_days = int(preferred_days)
                if preferred_days <= 0:
                    logger.error(f"Preferred days must be positive, got: {preferred_days}")
                    return "Error: Number of days must be a positive number", 400
            except (ValueError, TypeError) as e:
                logger.error(f"Invalid preferred days value: {preferred_days}, error: {e}")
                return "Error: Number of days must be a valid number", 400

            # Calculate drop-off date based on preferred_days
            drop_off_date = pick_up_date + timedelta(days=preferred_days - 1)
            logger.debug(f"Calculated drop-off date: {drop_off_date}")

            # Use pick_up_time for drop-off time as well
            drop_off_time = pick_up_time

            hotels = fetch_hotels(destination, budget, pick_up_date, drop_off_date)
            logger.info(f"Fetched {len(hotels)} hotels for {destination}")
            cars = fetch_cars(destination, budget, pick_up_date, drop_off_date, pick_up_time=pick_up_time, drop_off_time=drop_off_time)
            logger.info(f"Fetched {len(cars)} cars for {destination}")
            attractions = fetch_attractions(destination, pick_up_date, drop_off_date)
            logger.info(f"Fetched {len(attractions)} attractions for {destination}")
            restaurants = fetch_restaurants(destination, pick_up_date, drop_off_date)
            logger.info(f"Fetched {len(restaurants)} restaurants for {destination}")

            # Generate plans
            plans = generate_plans(hotels, cars, attractions, restaurants, destination, pick_up_date, drop_off_date, budget, preferred_days)
            if not plans:
                logger.error("No plans could be generated within the budget.")
                return "Error: No plans could be generated within your budget (or up to 20% more).", 400

            return render_template("results.html", plans=plans, destination=destination, budget=budget)
        except Exception as e:
            logger.error(f"Error processing POST request: {e}", exc_info=True)
            return f"Error: {str(e)}", 400
    logger.debug("Rendering index.html for GET request")
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        message = request.json.get("message")
        logger.debug(f"Received chat message: {message}")
        response = chatbot.handle_message(message)
        logger.debug(f"Chatbot response: {response}")
        return {"response": response}
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        return {"error": str(e)}, 400

if __name__ == "__main__":
    logger.info("Starting Flask application")
    app.run(host="0.0.0.0", port=5000)