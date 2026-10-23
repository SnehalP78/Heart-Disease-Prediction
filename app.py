from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = BASE_DIR / "public"
MODEL_DIR = BASE_DIR / "model"

app = FastAPI(title="Heart Disease Prediction")

app.mount(
    "/static",
    StaticFiles(directory=str(PUBLIC_DIR), check_dir=False),
    name="static"
)
)

# Load the trained model and its feature names
model = joblib.load(MODEL_DIR / "heart_model.pkl")
feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")


class PatientData(BaseModel):
    age: int = Field(ge=1, le=120)
    sex: int = Field(ge=0, le=1)
    cp: int = Field(ge=0, le=3)
    trestbps: int = Field(ge=1, le=300)
    chol: int = Field(ge=1, le=700)
    fbs: int = Field(ge=0, le=1)
    restecg: int = Field(ge=0, le=2)
    thalach: int = Field(ge=1, le=300)
    exang: int = Field(ge=0, le=1)
    oldpeak: float = Field(ge=0, le=10)
    slope: int = Field(ge=0, le=2)
    ca: int = Field(ge=0, le=4)
    thal: int = Field(ge=0, le=3)


@app.get("/")
def home():
    return FileResponse(str(PUBLIC_DIR / "index.html"))


@app.get("/api/health")
def health():
    return {"status": "API is running"}


@app.post("/api/predict")
def predict(patient: PatientData):
    try:
        patient_data = patient.model_dump()
        input_df = pd.DataFrame([patient_data], columns=feature_names)

        prediction = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]
        score = round(float(max(probabilities)) * 100, 2)

        result = (
            "Higher likelihood of heart disease according to this model"
            if prediction == 1
            else "Lower likelihood of heart disease according to this model"
        )

        return {
            "prediction": prediction,
            "result": result,
            "score": score,
            "note": (
                "Educational demonstration only. This result is not a medical "
                "diagnosis. Consult a qualified healthcare professional."
        
            )
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the server terminal for details."
        ) from exc
