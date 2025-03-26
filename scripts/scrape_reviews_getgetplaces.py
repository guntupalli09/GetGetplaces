# scripts/scrape_reviews_getgetplaces.py
from bs4 import BeautifulSoup
import requests
import logging
from utils.database import Database

logger = logging.getLogger(__name__)
db = Database()

def scrape_tripadvisor_reviews(city, num_reviews=10):
    url = f"https://www.tripadvisor.com/Search?q={city}+hotels"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        reviews = []
        for item in soup.find_all("div", class_="review")[:num_reviews]:
            review_text = item.get_text(strip=True)
            reviews.append(review_text)
        db.db.reviews.insert_many({"city": city, "review": r} for r in reviews)
        return reviews
    except requests.exceptions.RequestException as e:
        logger.error(f"Error scraping TripAdvisor for {city}: {e}")
        return []

if __name__ == "__main__":
    for city in ["Tampa", "Orlando", "Miami"]:
        reviews = scrape_tripadvisor_reviews(city)
        logger.info(f"Scraped {len(reviews)} reviews for {city}")