import os
import subprocess
import time
import requests
import pytest


def ensure_dummy_model():
    """Create a tiny model so the API can start during tests."""
    model_path = "src/models/model.keras"
    if os.path.exists(model_path):
        return

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential(
        [
            layers.Input(shape=(180, 180, 3)),
            layers.Flatten(),
            layers.Dense(5),
        ]
    )
    model.save(model_path)
    with open("src/models/class_names.txt", "w") as f:
        f.write("\n".join(["daisy", "dandelion", "roses", "sunflowers", "tulips"]))


API_HOST = "http://127.0.0.1:8001"
DATA_DIR = "data/flower_photos_test/daisy"
SAMPLE_IMG = os.path.join(DATA_DIR, os.listdir(DATA_DIR)[0])  # Pick one sample image


def start_server():
    """Start the FastAPI server in a subprocess."""
    return subprocess.Popen(
        ["uvicorn", "src.api.main:app", "--port", "8001"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def wait_for_server():
    """Poll until the API server is responding"""
    for _ in range(100):
        try:
            requests.get(API_HOST, timeout=0.5)
            return
        except Exception:
            time.sleep(0.1)
    raise RuntimeError("API server failed to start")


@pytest.fixture(scope="module", autouse=True)
def api_server():
    """Spin up the API server for the duration of the tests"""
    ensure_dummy_model()
    proc = start_server()
    try:
        wait_for_server()
        yield
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_root():
    """Root endpoint should return a simple status message"""
    resp = requests.get(f"{API_HOST}/")
    assert resp.status_code == 200
    assert resp.json()["message"] == "Flower Classifier API running"


def test_predict_valid_image():
    """Uploading an actual image should yield a prediction"""
    with open(SAMPLE_IMG, "rb") as f:
        files = {"file": ("test.jpg", f, "image/jpeg")}
        resp = requests.post(f"{API_HOST}/predict", files=files)
    assert resp.status_code == 200
    body = resp.json()
    assert "class" in body and "confidence" in body


def test_predict_invalid_file():
    """Non-image uploads should return HTTP 400"""
    files = {"file": ("test.txt", b"notanimage", "text/plain")}
    resp = requests.post(f"{API_HOST}/predict", files=files)
    assert resp.status_code == 400
