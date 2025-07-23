"""Hyperparameter tuning for the flower classification model."""

from __future__ import annotations
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
from typing import Tuple
import keras_tuner as kt
import random
import numpy as np


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
IMG_SIZE: Tuple[int, int] = (180, 180)
BATCH_SIZE: int = 32
TRAIN_DATA_DIR = "data/flower_photos"  # Full training dataset
TEST_DATA_DIR = "data/flower_photos_test"  # Optional held-put test set
EPOCHS = 20

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Classes: {class_names}")


# ---------------------------------------------------------------------------
# Hypermodel definition
# ---------------------------------------------------------------------------


def build_model(hp: kt.HyperParameters) -> keras.Model:
    """Build and compile a CNN based on hyperparameters"""
    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = layers.Rescaling(1.0 / 255)(inputs)

    x = layers.Conv2D(
        filters=hp.Choice("conv1_filters", [32, 64, 128]),
        kernel_size=3,
        activation="relu",
    )(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(
        filters=hp.Choice("conv2_filters", [64, 128, 256]),
        kernel_size=3,
        activation="relu",
    )(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Flatten()(x)
    x = layers.Dropout(hp.Float("dropout", 0.1, 0.4, step=0.1))(x)
    x = layers.Dense(
        units=hp.Int("dense_units", 64, 256, step=32),
        activation="relu",
    )(x)

    outputs = layers.Dense(num_classes, activation="linear")(x)
    model = keras.Model(inputs, outputs)

    lr = hp.Choice("learning_rate", [1e-3, 5e-4, 1e-4])
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Hyperparameter search
# ---------------------------------------------------------------------------

tuner = kt.RandomSearch(
    build_model,
    objective="val_accuracy",
    max_trials=10,
    directory="kt_logs",
    project_name="flower_tuning",
    overwrite=True,
)

tuner.search(train_ds, validation_data=val_ds, epochs=10)

# Retrieve results
best_hp = tuner.get_best_hyperparameters(1)[0]
best_model = tuner.get_best_models(1)[0]

# --------------------------------------------------------------------------
# Save best model & classes
# --------------------------------------------------------------------------
os.makedirs("src/models", exist_ok=True)
best_model.save("src/models/kt_best_model.keras")
with open("src/models/class_names.txt", "w") as f:
    f.write("\n".join(class_names))

print("Model and classes names save to src/models/")

# Optionally evaluate on a separate test set if it exists---------------------
if os.path.exists(TEST_DATA_DIR):
    print("\n--- Evaluating on Test Set ---")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DATA_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    test_loss, test_acc = best_model.evaluate(test_ds)
    print(f"Test accuracy: {test_acc:.3f}")

    print("\nBest hyperparameters:")
    for name, value in best_hp.values.items():
        print(f"  {name}: {value}")
    print("\nBest model saved to src/models/kt_best_model.keras")
