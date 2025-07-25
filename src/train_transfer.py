"""Train a MobileNetV2 model using transfer learning."""

import os
from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
IMG_SIZE: Tuple[int, int] = (180, 180)
BATCH_SIZE = 32
TRAIN_DATA_DIR = "data/flower_photos"
TEST_DATA_DIR = "data/flower_photos_test"
INITIAL_EPOCHS = 5
FINE_TUNE_EPOCHS = 5

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
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

# Performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ---------------------------------------------------------------------------
# Preprocessing / Augmentation
# ---------------------------------------------------------------------------
augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
    ],
    name="augmentation",
)

# ---------------------------------------------------------------------------
# Model Definition
# ---------------------------------------------------------------------------
base_model = keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False

inputs = keras.Input(shape=IMG_SIZE + (3,))
x = layers.Rescaling(1.0 / 255)(inputs)
x = augmentation(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

os.makedirs("src/models", exist_ok=True)
checkpoint = keras.callbacks.ModelCheckpoint(
    "src/models/transfer_model.keras", save_best_only=True, monitor="val_accuracy"
)

print("\n--- Initial Training ---")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS,
    callbacks=[checkpoint],
)

# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------
base_model.trainable = True
model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print("\n--- Fine-tuning ---")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
    initial_epoch=history.epoch[-1] + 1,
    callbacks=[checkpoint],
)

# Load best model for evaluation / saving
best_model = keras.models.load_model("src/models/transfer_model.keras")

with open("src/models/class_names.txt", "w") as f:
    f.write("\n".join(class_names))

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
