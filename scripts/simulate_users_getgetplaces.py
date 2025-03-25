# scripts/simulate_users_getgetplaces.py
import sys
import os
# Add the directory containing the 'utils' module to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.database import Database
from datetime import datetime, timedelta
import random

db = Database()

def simulate_users(num_users=10):
    cities = ["Tampa", "Orlando", "Miami"]
    for _ in range(num_users):
        user_id = random.randint(1, 1000)
        username = f"getget_user{random.randint(1, 1000)}"
        email = f"{username}@getgetplaces.com"
        preferences = {"budget": random.uniform(500, 1500), "indoor": random.choice([True, False])}
        session = db.Session()
        session.execute(
            "INSERT INTO users (username, email, preferences) VALUES (:username, :email, :preferences)",
            {"username": username, "email": email, "preferences": preferences}
        )
        session.commit()
        session.close()

        start_date = datetime(2025, 3, 10)
        end_date = start_date + timedelta(days=random.randint(3, 7))
        destination = random.choice(cities)
        cost = random.uniform(500, 2000)
        feedback = {"rating": random.uniform(3, 5), "comments": "Good trip!"}
        session = db.Session()
        session.execute(
            "INSERT INTO trips (user_id, start_date, end_date, destination, cost, feedback) VALUES (:user_id, :start_date, :end_date, :destination, :cost, :feedback)",
            {"user_id": user_id, "start_date": start_date, "end_date": end_date, "destination": destination, "cost": cost, "feedback": feedback}
        )
        session.commit()
        session.close()

if __name__ == "__main__":
    simulate_users()