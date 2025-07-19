"""Model loading and prediction utilities."""

import os
from typing import Dict

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from tensorflow import keras


MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/model.keras")
CLASSES_PATH = os.path.join(os.path.dirname(__file__), "../models/class_names.txt")
IMG_SIZE = (180, 180)


# Load model and classes once at startup
model = keras.models.load_model(MODEL_PATH)
with open(CLASSES_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines()]


class InvalidImageError(Exception):
    """Raised when an image file cannot be opened or processed."""


def predict_images(img_path: str) -> Dict[str, object]:
    """Predict the class for a single image path.

    Parameters
    ----------
    img_path : str
        Path to an image file on disk.

    Returns
    -------
    dict
        Dictionary containing the predicted class, confidence and
        probabilities for all classes.
    """

    try:
        img = Image.open(img_path).convert("RGB")
    except (FileNotFoundError, UnidentifiedImageError) as exc:
        # Re-raise as a more explicit error for the API layer to handle
        raise InvalidImageError("Cannot open supplied image") from exc

    img = img.resize(IMG_SIZE)
    arr = np.array(img)
    arr = np.expand_dims(arr, axis=0)  # Add batch dimension

    preds = model.predict(arr)[0]
    preds = tf.nn.softmax(
        preds
    ).numpy()  # Model outputs logits because it was trained with `from_logits=True`
    top_idx = int(np.argmax(preds))
    top_class = class_names[top_idx]
    confidence = float(preds[top_idx])

    return {
        "class": top_class,
        "confidence": confidence,
        "all_probs": dict(zip(class_names, preds.astype(float))),
    }
