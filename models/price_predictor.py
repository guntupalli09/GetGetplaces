# models/price_predictor.py
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class PricePredictor:
    def __init__(self):
        self.forecast = None

    def train(self):
        # Placeholder data (replace with real historical price data)
        prices = pd.Series([100, 110, 120, 130], index=pd.date_range(start="2025-03-01", periods=4))
        model = ARIMA(prices, order=(1, 1, 1))
        model_fit = model.fit()
        self.forecast = model_fit.forecast(steps=7)
        logger.info("Price predictor trained.")

    def predict_price(self, base_price, date):
        if not self.forecast:
            self.train()
        date = pd.Timestamp(date)
        return self.forecast[date].item() if date in self.forecast.index else base_price