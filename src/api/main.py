"""FastAPI application definition."""

import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from src.api.predictor import predict_images, InvalidImageError

app = FastAPI()


@app.get("/")
def root():
    """Simple health check endpoint"""
    return {"message": "Flower Classifier API running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict the class of an uploaded image."""

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()  # Read uploaded file into memory

    # Save bytes to a temporary file so Pillow/TensorFlow can open it
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    try:
        temp_file.write(contents)
        temp_file.close()
        try:
            result = predict_images(temp_file.name)
        except InvalidImageError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        try:
            temp_file.close()
        finally:
            if os.path.exists(temp_file.name):
                os.remove(temp_file.name)
    return result
