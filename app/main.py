import io
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from app.model import load_model, predict

app = FastAPI(title="Brain MRI classifier (research demo, not a medical device)")
model = load_model()

from fastapi.responses import RedirectResponse

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="File is not a readable image")
    probs = predict(model, img)
    return {
        "prediction": max(probs, key=probs.get),
        "probabilities": probs,
        "disclaimer": "Research demo only. Not a medical device.",
    }
