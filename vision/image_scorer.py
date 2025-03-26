# vision/image_scorer.py
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import requests
import logging

logger = logging.getLogger(__name__)
model = VGG16(weights="imagenet", include_top=False)

def score_image(image_url):
    try:
        response = requests.get(image_url, stream=True)
        img = load_img(response.raw, target_size=(224, 224))
        img_array = preprocess_input(img_to_array(img))
        features = model.predict(img_array[np.newaxis, ...])
        return np.mean(features)
    except Exception as e:
        logger.error(f"Error scoring image: {e}")
        return 0