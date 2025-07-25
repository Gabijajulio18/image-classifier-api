import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

IMG_SIZE = (180, 180)
BATCH_SIZE = 32
MODEL_PATH = "src/models/model.keras"
CLASS_PATH = "src/models/class_names.txt"
TEST_DATA_DIR = "data/flower_photos_test"

assert os.path.exists(MODEL_PATH), "Trained model not found"
assert os.path.exists(CLASS_PATH), "class_names.txt not found"
assert os.path.exists(TEST_DATA_DIR), "Test data directory not found"

model = tf.keras.models.load_model(MODEL_PATH)
with open(CLASS_PATH) as f:
    class_names = [line.strip() for line in f]

# load test set
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DATA_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False
)

# predictions
all_images = []
y_true = []
y_pred = []
for batch_images, batch_labels in test_ds:
    preds = model.predict(batch_images)
    probs = tf.nn.softmax(preds, axis=1)
    all_images.extend(batch_images)
    y_true.extend(batch_labels.numpy())
    y_pred.extend(tf.argmax(probs, axis=1).numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print("Confusion matrix:\n", cm)
print("\nClassification report:\n", report)

# Save misclassified image paths for manual inspection
mis_idx = np.where(y_true != y_pred)[0]
mis_dir = "misclassified_examples"
os.makedirs(mis_dir, exist_ok=True)
for idx in mis_idx:
    img = tf.keras.utils.array_to_img(all_images[idx])
    true_label = class_names[y_true[idx]]
    pred_label = class_names[y_pred[idx]]
    out_path = os.path.join(mis_dir, f"{idx}_{true_label}_as_{pred_label}.png")
    img.save(out_path)
print(f"Saved {len(mis_idx)} misclassified images to {mis_dir}/")
