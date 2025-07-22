import os
import subprocess
import time
import requests
import pytest

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
