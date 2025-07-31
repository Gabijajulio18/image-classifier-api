import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import importlib
from unittest.mock import Mock

import numpy as np
from fastapi.testclient import TestClient


def load_app(monkeypatch):
    """Load the FastAPI app with the model loading patched."""
    monkeypatch.setattr(
        "tensorflow.keras.models.load_model",
        lambda path: Mock(predict=Mock(return_value=np.array([[1, 0, 0, 0, 0]]))),
    )
    import src.api.predictor as predictor

    importlib.reload(predictor)

    import src.api.main as main

    importlib.reload(main)

    return main.app, main


def test_predict_endpoint(monkeypatch):
    app, main = load_app(monkeypatch)

    def fake_predict(_path):
        return {"class": "daisy", "confidence": 1.0, "all_probs": {"daisy": 1.0}}

    monkeypatch.setattr(main, "predict_images", fake_predict)
    client = TestClient(app)

    image_path = (
        Path(__file__).resolve().parents[1]
        / "misclassified_examples/104_dandelion_as_daisy.png"
    )
    with image_path.open("rb") as f:
        response = client.post(
            "/predict", files={"file": ("image.png", f, "image/png")}
        )

    assert response.status_code == 200
    assert response.json()["class"] == "daisy"
