from fastapi import FastAPI, UploadFile, File
from src.api.predictor import predict_images
import os
import tempfile

app = FastAPI()


@app.get("/")
def root():
    return {"message": "Flower Classifier API running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict the class of an uploaded image"""

    contents = await file.read()

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    try:
        temp_file.write(contents)
        temp_file.close()
        result = predict_images(temp_file.name)
    finally:
        try:
            temp_file.close()
        finally:
            if os.path.exists(temp_file.name):
                os.remove(temp_file.name)
    return result
