# models/recommendation.py
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
import logging
import os

logger = logging.getLogger(__name__)

class RecommendationModel:
    def __init__(self):
        self.model_path = "models/recommendation_model.pkl"
        self.model = None
        if not os.path.exists("models"):
            os.makedirs("models")

    def train(self):
        # Placeholder data (replace with real user data from database)
        data = pd.DataFrame({
            "user_id": [1, 1, 2, 2],
            "item_id": [101, 102, 103, 104],
            "rating": [4.5, 3.0, 5.0, 4.0],
            "price": [100, 150, 200, 120],
            "distance": [0.5, 2.0, 1.0, 1.5]
        })
        X = data[["price", "distance"]]
        y = data["rating"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = RandomForestRegressor()
        self.model.fit(X_train, y_train)
        joblib.dump(self.model, self.model_path)
        logger.info("Recommendation model trained and saved.")

    def load(self):
        if not self.model:
            try:
                self.model = joblib.load(self.model_path)
            except FileNotFoundError:
                logger.warning("Model not found, training a new one.")
                self.train()
        return self.model

    def predict_preference(self, price, distance):
        model = self.load()
        return model.predict([[price, distance]])[0]