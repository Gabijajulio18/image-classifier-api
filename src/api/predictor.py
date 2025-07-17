import tensorflow as tf
import numpy as np
import os
from tensorflow import keras
from PIL import Image


MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/model.keras")
CLASSES_PATH = os.path.join(os.path.dirname(__file__), "../models/class_names.txt")
IMG_SIZE = (180, 180)


# Load model and classses once at startup
model = keras.models.load_model(MODEL_PATH)
with open(CLASSES_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines()]


def predict_images(img_path: str):
    img = Image.open(img_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img)
    arr = np.expand_dims(arr, axis=0)  # Add batch dimension

    preds = model.predict(arr)[0]
    preds = tf.nn.softmax(
        preds
    ).numpy()  # Model outputs logits because it was trained with `from_logits=True`
    top_idx = np.argmax(preds)
    top_class = class_names[top_idx]
    confidence = float(preds[top_idx])

    # Return top-N
    return {
        "class": top_class,
        "confidence": confidence,
        "all_probs": dict(zip(class_names, preds.astype(float))),
    }
