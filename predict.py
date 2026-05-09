"""
SignSpeak — Prediction Module

Provides SignPredictor class for single-image inference with
correct preprocessing and label mapping.
"""

import io
import base64
import numpy as np
from PIL import Image, ImageFilter
from tensorflow.keras.models import load_model

from model import NUM_CLASSES, INDEX_TO_LETTER, IMG_SIZE


class SignPredictor:
    """Reusable predictor for ASL fingerspelling images."""

    def __init__(self, model_path: str = "sign_model_v2.keras"):
        self.model = load_model(model_path)
        self.img_size = IMG_SIZE

    def preprocess(self, img: Image.Image) -> np.ndarray:
        """Convert a PIL Image to model-ready tensor.

        Pipeline:
          1. Convert to grayscale
          2. Resize to 28×28
          3. Normalize pixels to [0, 1]
          4. Reshape to (1, 28, 28, 1)
        """
        img = img.convert('L')
        img = img.resize((self.img_size, self.img_size), Image.LANCZOS)
        arr = np.array(img, dtype='float32') / 255.0
        return arr.reshape(1, self.img_size, self.img_size, 1)

    def predict(self, img: Image.Image) -> dict:
        """Run inference on a PIL Image.

        Returns:
            dict with keys: letter, confidence, top3
        """
        x = self.preprocess(img)
        probs = self.model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        letter = INDEX_TO_LETTER[idx]

        # Top-3 predictions
        top3_idx = np.argsort(probs)[-3:][::-1]
        top3 = [
            {"letter": INDEX_TO_LETTER[int(i)], "confidence": float(probs[i])}
            for i in top3_idx
        ]

        return {"letter": letter, "confidence": confidence, "top3": top3}

    def predict_from_base64(self, data_uri: str) -> dict:
        """Predict from a base64-encoded image (data URI from webcam/upload)."""
        if ',' in data_uri:
            data_uri = data_uri.split(',')[1]
        decoded = base64.b64decode(data_uri)
        img = Image.open(io.BytesIO(decoded))
        return self.predict(img)

    def predict_from_file(self, file_path: str) -> dict:
        """Predict from an image file path."""
        img = Image.open(file_path)
        return self.predict(img)
