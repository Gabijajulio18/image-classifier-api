"""Train a MobileNetV2 model using transfer learning."""

import argparse
import os
from typing import Tuple

import csv
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
IMG_SIZE: Tuple[int, int] = (180, 180)
DEFAULT_BATCH_SIZE = 32
DEFAULT_TRAIN_DIR = "data/flower_photos"
DEFAULT_TEST_DIR = "data/flower_photos_test"
DEFAULT_INITIAL_EPOCHS = 5
DEFAULT_FINE_TUNE_EPOCHS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MobileNetV2 via transfer learning"
    )
    parser.add_argument("--train-dir", default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--test-dir", default=DEFAULT_TEST_DIR)
    parser.add_argument("--model-out", default="src/models/transfer_model.keras")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--initial-epochs", type=int, default=DEFAULT_INITIAL_EPOCHS)
    parser.add_argument("--fine-epochs", type=int, default=DEFAULT_FINE_TUNE_EPOCHS)
    return parser.parse_args()


args = parse_args()


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    args.train_dir,
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=IMG_SIZE,
    batch_size=args.batch_size,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    args.train_dir,
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=IMG_SIZE,
    batch_size=args.batch_size,
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

os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
checkpoint = keras.callbacks.ModelCheckpoint(
    args.model_out, save_best_only=True, monitor="val_accuracy"
)

print("\n--- Initial Training ---")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=args.initial_epochs,
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
    epochs=args.initial_epochs + args.fine_epochs,
    initial_epoch=history.epoch[-1] + 1,
    callbacks=[checkpoint],
)

# Load best model for evaluation / saving
best_model = keras.models.load_model(args.model_out)

class_file = os.path.join(os.path.dirname(args.model_out), "class_names.txt")
with open(class_file, "w") as f:
    f.write("\n".join(class_names))

test_acc = None
if os.path.exists(args.test_dir):
    print("\n--- Evaluating on Test Set ---")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        args.test_dir,
        image_size=IMG_SIZE,
        batch_size=args.batch_size,
        shuffle=False,
    )
    _, test_acc = best_model.evaluate(test_ds)
    print(f"Test accuracy: {test_acc:.3f}")

# -------------------------------------------------------------------
# Log experiment result
# -------------------------------------------------------------------

log_path = "experiments.csv"
write_header = not os.path.exists(log_path)
with open(log_path, "a", newline="") as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow(["timestamp", "model", "val_accuracy", "test_accuracy"])
    writer.writerow(
        [
            datetime.now().isoformat(timespec="seconds"),
            "mobilenetv2",
            max(history.history.get("val_accuracy", [0])),
            test_acc,
        ]
    )
