from fastapi import FastAPI, UploadFile, File
from src.api.predictor import predict_images
import os

app = FastAPI()


@app.get("/")
def root():
    return {"message": "Flower Classifier API running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Save the uploaded file temporarily
    contents = await file.read()
    with open("temp_image.jpg", "wb") as f:
        f.write(contents)

    result = predict_images("temp_image.jpg")
    os.remove("temp_image.jpg")
    return result
