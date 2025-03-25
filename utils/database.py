from sqlalchemy import create_engine, Column, Integer, String, Float, Date, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()

Base = declarative_base()

# Define your models
class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String)
    email = Column(String)
    preferences = Column(JSON)

class Hotel(Base):
    __tablename__ = 'hotels'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    price = Column(Float)
    rating = Column(Float)
    distance = Column(Float)
    city = Column(String)
    reviews = Column(JSON)
    lat = Column(Float)
    lng = Column(Float)

class Car(Base):
    __tablename__ = 'cars'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    price = Column(Float)
    rating = Column(Float)
    distance = Column(Float)
    company = Column(String)
    city = Column(String)
    reviews = Column(JSON)

class Trip(Base):
    __tablename__ = 'trips'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer)
    start_date = Column(Date)
    end_date = Column(Date)
    destination = Column(String)
    cost = Column(Float)
    feedback = Column(JSON)

class Database:
    def __init__(self):
        self.postgres_url = os.getenv("DATABASE_URL", "postgresql://postgres:Santhu09!@db.wlznuumoqtshmdcidlww.supabase.co:5432/postgres")
        self.engine = create_engine(self.postgres_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def insert_hotel(self, name, price, rating, distance, city, lat=0, lng=0, reviews=None):
        session = self.Session()
        hotel = Hotel(name=name, price=price, rating=rating, distance=distance, city=city, lat=lat, lng=lng, reviews=reviews or [])
        session.add(hotel)
        session.commit()
        session.close()

    def insert_car(self, name, price, rating, distance, company, city, reviews=None):
        session = self.Session()
        car = Car(name=name, price=price, rating=rating, distance=distance, company=company, city=city, reviews=reviews or [])
        session.add(car)
        session.commit()
        session.close()

    # The attractions and restaurants methods would need to be converted to use SQLAlchemy models if needed.
    # Similarly, for the weather data, if you plan to keep those functionalities, you would need to create respective SQLAlchemy models and use them here.

