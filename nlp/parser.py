# nlp/parser.py
import spacy
import logging

logger = logging.getLogger(__name__)
nlp = spacy.load("en_core_web_sm")

def parse_nlp_input(text):
    doc = nlp(text)
    destination = None
    budget = None
    preferences = []
    for ent in doc.ents:
        if ent.label_ == "GPE":
            destination = ent.text
        elif ent.label_ == "MONEY":
            budget = float(ent.text.replace("$", ""))
        elif ent.text.lower() in ["cheap", "luxury", "food", "museums"]:
            preferences.append(ent.text.lower())
    return destination, budget, preferences