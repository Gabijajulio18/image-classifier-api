"""Train a baseline CNN on the flowers dataset with basic regularisation."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
from typing import Tuple


IMG_SIZE: Tuple[int, int] = (180, 180)
BATCH_SIZE = 32
TRAIN_DATA_DIR = "data/flower_photos"  # Full training dataset
TEST_DATA_DIR = "data/flower_photos_test"  # Optional held-put test set
EPOCHS = 40


# -----------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------
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


# -----------------------------------------------------------------------
# Improve performance by caching and prefetching
# -----------------------------------------------------------------------
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# -----------------------------------------------------------------------
# Data augmentation
# -----------------------------------------------------------------------
augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ]
)


# -----------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------
model = keras.Sequential(
    [
        layers.Rescaling(
            1.0 / 255, input_shape=IMG_SIZE + (3,)
        ),  # Normalize pixel values to [0, 1]
        layers.Conv2D(64, 3, activation="relu"),  # Learn patterns/features from images
        layers.MaxPooling2D(),  # Reduces image size, keeps most important info
        layers.Conv2D(128, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(256, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),  # Converts 3D feature to 1D vector
        layers.Dropout(0.4)
        layers.Dense(256, activation="relu"),  # Learns higher-level features
        layers.Dense(
            num_classes, activation="linear"
        ),  # Outputs probabilities for each class
    ]
)

lr = 1e-4
optimizer = keras.optimizers.Adam(learning_rate=lr)
model.compile(
    optimizer=optimizer,
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

stop_early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)


# Train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=[stop_early],
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
