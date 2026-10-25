from pathlib import Path
import logging
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "model.joblib"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("heart-disease-api")

app = FastAPI(title="Heart Disease Classification API", version="1.0.0")
Instrumentator().instrument(app).expose(app)


class PatientFeatures(BaseModel):
    age: float = Field(ge=1, le=120)
    sex: int = Field(ge=0, le=1)
    cp: int = Field(ge=1, le=4)
    trestbps: float = Field(gt=0, le=300)
    chol: float = Field(gt=0, le=1000)
    fbs: int = Field(ge=0, le=1)
    restecg: int = Field(ge=0, le=2)
    thalach: float = Field(gt=0, le=300)
    exang: int = Field(ge=0, le=1)
    oldpeak: float = Field(ge=0, le=10)
    slope: int = Field(ge=1, le=3)
    ca: int = Field(ge=0, le=3)
    thal: int = Field(ge=3, le=7)


def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


model = load_model()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(patient: PatientFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not trained. Run src.train first.")
    values = patient.model_dump()
    frame = pd.DataFrame([values])
    prediction = int(model.predict(frame)[0])
    probabilities = model.predict_proba(frame)[0]
    confidence = float(max(probabilities))
    logger.info("prediction_request processed prediction=%s", prediction)
    return {
        "prediction": prediction,
        "label": "disease_present" if prediction == 1 else "disease_absent",
        "confidence": round(confidence, 4),
        "note": "Educational model output; not a medical diagnosis.",
    }
