"""Compare baseline CNN and transfer learning models on the test dataset."""

import json
import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras

IMG_SIZE: Tuple[int, int] = (180, 180)
BATCH_SIZE = 32
TEST_DATA_DIR = "data/flower_photos_test"
BASELINE_MODEL = "src/models/model.keras"
TRANSFER_MODEL = "src/models/transfer_model.keras"


def load_test_ds() -> tf.data.Dataset:
    """Load the test dataset without shuffling."""
    return tf.keras.utils.image_dataset_from_directory(
        TEST_DATA_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )


def evaluate(model_path: str, ds: tf.data.Dataset, class_names: list) -> Dict:
    """Evaluate a model and return metrics."""
    model = keras.models.load_model(model_path)
    loss, acc = model.evaluate(ds, verbose=0)

    y_true, y_pred = [], []
    for batch_images, batch_labels in ds:
        preds = model.predict(batch_images, verbose=0)
        probs = tf.nn.softmax(preds, axis=1).numpy()
        y_true.extend(batch_labels.numpy())
        y_pred.extend(np.argmax(probs, axis=1))

    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )
    cm = confusion_matrix(y_true, y_pred)
    return {"loss": loss, "accuracy": acc, "report": report, "cm": cm}


def save_confusion_matrix(cm: np.ndarray, labels: list, path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    plt.savefig(path)
    plt.close(fig)


def main():
    assert os.path.exists(BASELINE_MODEL), "Baseline model not found"
    assert os.path.exists(TRANSFER_MODEL), "Transfer model not found"
    ds = load_test_ds()
    class_names = ds.class_names

    results_dir = "evaluation_results"
    os.makedirs(results_dir, exist_ok=True)

    baseline = evaluate(BASELINE_MODEL, ds, class_names)
    transfer = evaluate(TRANSFER_MODEL, ds, class_names)

    with open(os.path.join(results_dir, "baseline_report.json"), "w") as f:
        json.dump(baseline["report"], f, indent=2)
    np.save(os.path.join(results_dir, "baseline_cm.npy"), baseline["cm"])
    save_confusion_matrix(
        baseline["cm"], class_names, os.path.join(results_dir, "baseline_cm.png")
    )

    with open(os.path.join(results_dir, "transfer_report.json"), "w") as f:
        json.dump(transfer["report"], f, indent=2)
    np.save(os.path.join(results_dir, "transfer_cm.npy"), transfer["cm"])
    save_confusion_matrix(
        transfer["cm"], class_names, os.path.join(results_dir, "transfer_cm.png")
    )

    summary = [
        {
            "Model": "Baseline CNN",
            "Accuracy": baseline["accuracy"],
            "Macro F1": baseline["report"]["macro avg"]["f1-score"],
        },
        {
            "Model": "MobileNetV2",
            "Accuracy": transfer["accuracy"],
            "Macro F1": transfer["report"]["macro avg"]["f1-score"],
        },
    ]

    with open(os.path.join(results_dir, "comparison.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n| Model | Accuracy | Macro F1 |")
    print("|-------|---------:|---------:|")
    for row in summary:
        print(f"| {row['Model']} | {row['Accuracy']:.3f} | {row['Macro F1']:.3f} |")


if __name__ == "__main__":
    main()
