"""Training script for the flower classification model.

This module can be run as a script and exposes several command line arguments so
experiments can be reproduced easily. By default it expects the TensorFlow
flowers dataset in ``data/flower_photos``. A small held out test set can be
placed in ``data/flower_photos_test``.
"""

import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
from typing import Tuple
import csv
from datetime import datetime


IMG_SIZE: Tuple[int, int] = (180, 180)
DEFAULT_BATCH_SIZE = 32
DEFAULT_TRAIN_DIR = "data/flower_photos"  # Full training dataset
DEFAULT_TEST_DIR = "data/flower_photos_test"  # Optional held-put test set
DEFAULT_EPOCHS = 50


def parse_args() -> argparse.Namespace:
    """Parse command line options."""

    parser = argparse.ArgumentParser(description="Train the flower classifier")
    parser.add_argument(
        "--train-dir", default=DEFAULT_TRAIN_DIR, help="Training data directory"
    )
    parser.add_argument(
        "--test-dir", default=DEFAULT_TEST_DIR, help="Optional test data directory"
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--model-path", default="src/models/model.keras")
    parser.add_argument("--class-path", default="src/models/class_names.txt")


def main() -> None:
    args = parse_args()
    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
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
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2),
            layers.RandomContrast(0.2),
            layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        ]
    )

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    model = keras.Sequential(
        [
            layers.Rescaling(1.0 / 255, input_shape=IMG_SIZE + (3,)),
            augmentation,
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="linear"),
        ]
    )

    lr = 1e-4
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    stop_early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)

    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[stop_early],
    )

    # Save model & classes
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    model.save(args.model_path)  # Saved in Keras format
    with open(args.class_path, "w") as f:
        f.write("\n".join(class_names))

    print("Model and classes names save to src/models/")

    # Optionally evaluate on a separate test set if it exists
    if os.path.exists(args.test_dir):
        print("\n--- Evaluating on Test Set ---")
        test_ds = tf.keras.utils.image_dataset_from_directory(
            args.test_dir,
            image_size=IMG_SIZE,
            batch_size=args.batch_size,
            shuffle=False,
        )
        test_loss, test_acc = model.evaluate(test_ds)
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
                "baseline_cnn",
                max(history.history.get("val_accuracy", [0])),
                test_acc,
            ]
        )


if __name__ == "__main__":
    main()
