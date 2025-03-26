# chatbot/bot.py
import logging

logger = logging.getLogger(__name__)

class Chatbot:
    def handle_message(self, message):
        # Placeholder for Rasa or Dialogflow integration
        if "add museum" in message.lower():
            return "Adding a museum to your itinerary. Please specify the day."
        return "I can help with your itinerary! What would you like to do?"