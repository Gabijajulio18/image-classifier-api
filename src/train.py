import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import _tf_uses_legacy_keras
import os


# Settings
img_size = (180, 180)
batch_size = 32
train_data_dir = "data/flowers_photos"


# Load data
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_data_dir,
    validation_split=0.2,
    subset="training",
    seed=1337,
    img_size=img_size,
    batch_size=batch_size,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    train_data_dir,
    validation_splt=0.2,
    subset="validation",
    seed=1337,
    img_size=img_size,
    batch_size=batch_size,
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Classes: {class_names}")


# Model
model = keras.Sequential(
    [
        layers.Rescaling(1.0 / 255, input_shape=img_size + (3,)),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)


# Train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
)


# Save model & classes
os.makedirs("src/models", exist_ok=True)
model.save("src/models/model.keras")
with open("src/models/class_names.txt", "w") as f:
    f.write("\n".join(class_names))

print("Model and classes names save to src/models/")

test_data_dir = "data/flower_photos_test"
if os.path.exist(test_data_dir):
    print("\n--- Evaluating on Test Set ---")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_data_dir,
        img_size=img_size,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test accuracy: {test_ds:.3f}")
