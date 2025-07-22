"""Training script for the flower classification model"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os


# Settings
IMG_SIZE = (180, 180)
BATCH_SIZE = 32
TRAIN_DATA_DIR = "data/flower_photos"  # Full training dataset
TEST_DATA_DIR = "data/flower_photos_test"  # Optional held-put test set

# Load data
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Classes: {class_names}")


# Model
model = keras.Sequential(
    [
        layers.Rescaling(
            1.0 / 255, input_shape=IMG_SIZE + (3,)
        ),  # Normalize pixel values to [0, 1]
        layers.Conv2D(32, 3, activation="relu"),  # Learn patterns/features from images
        layers.MaxPooling2D(),  # Reduces image size, keeps most important info
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),  # Converts 3D feature to 1D vector
        layers.Dense(128, activation="relu"),  # Learns higher-level features
        layers.Dense(
            num_classes, activation="linear"
        ),  # Outputs probabilities for each class
    ]
)

model.compile(
    optimizer="adam",  # Good default optimizer
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)


# Train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
)


# Save model & classes
os.makedirs("src/models", exist_ok=True)
model.save("src/models/model.keras")  # Saved in Keras format
with open("src/models/class_names.txt", "w") as f:
    f.write("\n".join(class_names))

print("Model and classes names save to src/models/")

# Optionally evaluate on a separate test set if it exists
if os.path.exists(TEST_DATA_DIR):
    print("\n--- Evaluating on Test Set ---")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DATA_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test accuracy: {test_acc:.3f}")
