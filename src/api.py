from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from scipy.sparse import hstack
from src.model_loader import load_model

MODEL_PATH = "models/model.pkl"
PREPROCESSOR_PATH = "data/processed/preprocessor.pkl"
HASHER_PATH = "data/processed/hasher.pkl"

HIGH_CARD_CAT = [
    "address",
    "AdCreationDate",
    "Subscription"
]

LOW_CARD_CAT = [
    "district",
    "HeatingType",
    "StructureType",
    "FloorLocation"
]

NUMERIC_FEATURES = [
    "GrossSquareMeters",
    "NetSquareMeters",
    "BuildingAge",
    "NumberOfRooms"
]

ALL_FEATURES = NUMERIC_FEATURES + LOW_CARD_CAT + HIGH_CARD_CAT

CATEGORIES = {
    0: "low",
    1: "medium",
    2: "high"
}

try:
    model = load_model("production")
except Exception:
    print("Falling back to staging model")
    model = load_model("staging")

preprocessor = joblib.load(PREPROCESSOR_PATH)
hasher = joblib.load(HASHER_PATH)

app = FastAPI(
    title="Housing Price Category API",
    description="Predicts housing price category (low / mid / high)",
    version="1.0",
)


class HousingInput(BaseModel):
    GrossSquareMeters: float
    NetSquareMeters: float
    BuildingAge: float
    NumberOfRooms: float
    district: str
    HeatingType: str
    StructureType: str
    FloorLocation: str
    address: str
    AdCreationDate: str
    Subscription: str


def preprocess_input(data: dict):
    df = pd.DataFrame([data])

    X_base = preprocessor.transform(df)

    X_high = df[HIGH_CARD_CAT].astype(str).values.tolist()
    X_hash = hasher.transform(X_high)

    X_final = hstack([X_base, X_hash])

    return X_final


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(input_data: HousingInput):
    data = input_data.model_dump()
    X = preprocess_input(data)
    prediction = model.predict(X)[0]

    print(f"Prediction: {prediction}")

    return {
        "price_category": CATEGORIES[int(prediction)]
    }

