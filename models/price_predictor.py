# models/price_predictor.py
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class PricePredictor:
    def __init__(self):
        self.forecast = None
        self.model_fit = None
        self.last_training_date = None

    def train(self):
        # Placeholder data (replace with real historical price data)
        prices = pd.Series([100, 110, 120, 130], index=pd.date_range(start="2025-03-01", periods=4, freq="D"))
        self.last_training_date = prices.index[-1]  # Last date in the training data
        model = ARIMA(prices, order=(1, 1, 1))
        self.model_fit = model.fit()
        # Forecast 30 days into the future to cover a wider range
        self.forecast = self.model_fit.forecast(steps=30)
        logger.info(f"Price predictor trained. Forecasted prices:\n{self.forecast}")

    def predict_price(self, base_price, date):
        if not self.forecast:
            self.train()

        # Ensure date is a datetime or Timestamp
        if not isinstance(date, (datetime, pd.Timestamp)):
            logger.error(f"Invalid date type: {type(date)}, expected datetime or Timestamp")
            raise ValueError(f"Date must be a datetime or Timestamp, got {type(date)}")
        
        # Convert date to pd.Timestamp
        date = pd.Timestamp(date)
        logger.debug(f"Predicting price for date: {date}, base_price: {base_price}")

        # Ensure self.forecast is a Series
        if not isinstance(self.forecast, pd.Series):
            logger.error(f"self.forecast is not a pandas Series: {type(self.forecast)}")
            return base_price

        # Check if the date is within the forecast range
        if date in self.forecast.index:  # Simplified check
            predicted_price = self.forecast.loc[date]
            logger.debug(f"Date {date} found in forecast index. Predicted price: {predicted_price}, type: {type(predicted_price)}")
            if isinstance(predicted_price, pd.Series):
                logger.warning(f"Predicted price is a Series: {predicted_price}")
                predicted_price = predicted_price.iloc[0] if not predicted_price.empty else base_price
            return float(predicted_price)

        # If the date is not in the forecast index, extend the forecast or apply a trend
        logger.warning(f"Date {date} not in forecast index (last forecast date: {self.forecast.index[-1]}). Applying trend-based adjustment.")

        # Calculate how many days into the future the requested date is
        days_ahead = (date - self.last_training_date).days
        if days_ahead <= 0:
            logger.error(f"Requested date {date} is before the last training date {self.last_training_date}. Using base_price.")
            return base_price

        # Extend the forecast if necessary
        if days_ahead > len(self.forecast):
            logger.debug(f"Extending forecast to {days_ahead} days to cover {date}")
            self.forecast = self.model_fit.forecast(steps=days_ahead)
            logger.debug(f"Extended forecast:\n{self.forecast}")

        # Check again if the date is in the extended forecast
        if date in self.forecast.index:
            predicted_price = self.forecast.loc[date]
            logger.debug(f"Date {date} found in extended forecast. Predicted price: {predicted_price}, type: {type(predicted_price)}")
            if isinstance(predicted_price, pd.Series):
                logger.warning(f"Predicted price is a Series: {predicted_price}")
                predicted_price = predicted_price.iloc[0] if not predicted_price.empty else base_price
            return float(predicted_price)

        # Fallback: Use the last forecasted value and apply a trend
        last_forecasted_price = self.forecast.iloc[-1]
        # Estimate the trend from the forecast (e.g., average daily change)
        if len(self.forecast) > 1:
            daily_change = (self.forecast.iloc[-1] - self.forecast.iloc[0]) / (len(self.forecast) - 1)
        else:
            daily_change = 0
        days_beyond_forecast = (date - self.forecast.index[-1]).days
        adjusted_price = last_forecasted_price + (daily_change * days_beyond_forecast)
        logger.debug(f"Applied trend adjustment: last_forecasted_price={last_forecasted_price}, daily_change={daily_change}, days_beyond_forecast={days_beyond_forecast}, adjusted_price={adjusted_price}")

        # Scale the base_price by the ratio of the adjusted price to the base_price
        if base_price > 0:
            scaled_price = base_price * (adjusted_price / 100)  # Assuming the training data starts around 100
        else:
            scaled_price = adjusted_price
        logger.debug(f"Final scaled price for {date}: {scaled_price}")
        return float(scaled_price)