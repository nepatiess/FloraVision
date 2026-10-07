from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from src.inference import predict_image


app = FastAPI(
    title="FloraVision API",
    description="Flower species recognition API",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "message": "FloraVision API is running"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()

        image = Image.open(
            BytesIO(image_bytes)
        )

        predictions = predict_image(image)

        return {
            "filename": file.filename,
            "prediction": predictions[0]["class"],
            "confidence": predictions[0]["confidence"],
            "top3": predictions
        }

    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image."
        )