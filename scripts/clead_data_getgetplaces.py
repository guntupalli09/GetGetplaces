# scripts/clean_data_getgetplaces.py
import pandas as pd
from utils.database import Database

db = Database()

def clean_data():
    # Clean hotels
    hotels = pd.read_sql_query("SELECT * FROM hotels", db.engine)
    hotels = hotels.dropna(subset=["name", "price", "rating"])
    hotels["price"] = hotels["price"].apply(lambda x: max(10, min(x, 1000)))
    hotels.to_sql("hotels", db.engine, if_exists="replace", index=False)

    # Clean MongoDB attractions
    attractions = pd.DataFrame(list(db.db.attractions.find()))
    attractions = attractions.dropna(subset=["name", "rating"])
    attractions["rating"] = attractions["rating"].apply(lambda x: max(0, min(x, 5)))
    db.db.attractions.drop()
    db.db.attractions.insert_many(attractions.to_dict("records"))

if __name__ == "__main__":
    clean_data()