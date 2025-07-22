"""Utility script for evaluating the trained model on the test dataset"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from pprint import pprint
import os
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report
import json


IMG_SIZE = (180, 180)
BATCH_SIZE = 32
MODEL_PATH = "src/models/model.keras"
CLASS_PATH = "src/models/class_names.txt"
TEST_DATA_DIR = "data/flower_photos_test"  # Separate test dataset

assert os.path.exists(MODEL_PATH), "Trained model not found."
assert os.path.exists(CLASS_PATH), "class_names.txt not found."
assert os.path.exists(TEST_DATA_DIR), "Test data directory not found."


# Load model and class names
model = keras.models.load_model(MODEL_PATH)
with open(CLASS_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines()]


# Load test set
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DATA_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False
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
