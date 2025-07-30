"""Utility script for evaluating the trained model on the test dataset"""

import argparse
import tensorflow as tf
from tensorflow import keras
import numpy as np
from pprint import pprint
import os
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report
import json


IMG_SIZE = (180, 180)
DEFAULT_SIZE = 32
DEFAULT_MODEL = "src/models/model.keras"
DEFAULT_CLASS = "src/models/class_names.txt"
DEFAULT_TEST_DIR = "data/flower_photos_test"  # Separate test dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--class-path", default=DEFAULT_CLASS)
    parser.add_argument("--test-dir", default=DEFAULT_TEST_DIR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    assert os.path.exists(args.model_path), "Trained model not found."
    assert os.path.exists(args.class_path), "class_names.txt not found."
    assert os.path.exists(args.test_dir), "Test data directory not found"

    # Load model and class names
    model = keras.models.load_model(args.model_path)
    with open(args.class_path, "r") as f:
        class_names = [line.strip() for line in f.readlines()]

    # Load test set
    test_ds = tf.keras.utils.image_dataset_from_directory(
        args.test_dir, image_size=IMG_SIZE, batch_size=args.batch_size, shuffle=False
    )

    # Evaluate model
    test_loss, test_acc = model.evaluate(test_ds, verbose=2)
    print(f"\nTest accuracy: {test_acc:.4f}")

    # Predict labels
    y_true, y_pred = [], []
    for batch_images, batch_labels in test_ds:
        preds = model.predict(batch_images)
        y_true.extend(batch_labels.numpy())
        probs = tf.nn.softmax(preds, axis=1).numpy()  # Apply softmax to logits
        y_pred.extend(np.argmax(probs, axis=1))

    # Metric
    cm = confusion_matrix(y_true, y_pred)  # Raw counts for each class
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )  # Precision/recall/F1 per class

    # Log results to file
    results_dir = "evaluation_results"
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "classification_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    np.save(os.path.join(results_dir, "confusion_matrix.npy"), cm)
    print("\nSaved metrics to evaluation_results/")

    # Visualize the confusion matrix
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
    plt.close(fig)

    # Print metrics
    print("\nConfusion matrix:\n", cm)
    print("\nClassification report:")
    pprint(report)


if __name__ == "__main__":
    main()
