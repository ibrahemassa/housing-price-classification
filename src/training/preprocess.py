import os
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction import FeatureHasher
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAW_DATA_PATH = "./data/HouseDataRaw.csv"
PROCESSED_DIR = "./data/processed"

RANDOM_STATE = 42
TEST_SIZE = 0.2

HIGH_CARD_CAT = [
    "address",
    "AdCreationDate",
    "Subscription",
    "district_heating",
    "district_floor",
]

LOW_CARD_CAT = ["district", "HeatingType", "StructureType", "FloorLocation"]

NUMERIC_FEATURES = [
    "GrossSquareMeters",
    "NetSquareMeters",
    "BuildingAge",
    "NumberOfRooms",
]


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df["price"] = (
        df["price"]
        .str.replace(",", "", regex=True)
        .str.replace("TL", "", regex=True)
        .str.extract(r"(\d+\.?\d*)")[0]
        .astype(float)
    )

    df["price_category"] = pd.qcut(df["price"], q=[0, 0.4, 0.8, 1.0], labels=[0, 1, 2])

    df = df.drop(columns=["price"])

    return df


def clean_square_meters(x):
    if pd.isna(x):
        return np.nan
    x = re.sub(r"[^\d]", "", str(x))
    return float(x) if x != "" else np.nan


def clean_building_age(x):
    if pd.isna(x):
        return np.nan
    x = str(x)
    if "Üzeri" in x:
        return 21
    return int(re.sub(r"[^\d]", "", x))


def clean_rooms(x):
    if pd.isna(x):
        return np.nan

    x = str(x).lower().strip()

    if "+" in x:
        x = x.split("+")[0]

    match = re.search(r"\d+(\.\d+)?", x)
    if match:
        return float(match.group())

    return np.nan


MONTH_MAP = {
    "Ocak": "01",
    "Şubat": "02",
    "Mart": "03",
    "Nisan": "04",
    "Mayıs": "05",
    "Haziran": "06",
    "Temmuz": "07",
    "Ağustos": "08",
    "Eylül": "09",
    "Ekim": "10",
    "Kasım": "11",
    "Aralık": "12",
}


def clean_date(x):
    if pd.isna(x):
        return "unknown"
    parts = str(x).split()
    if len(parts) == 3:
        month = MONTH_MAP.get(parts[1], "00")
        year = parts[2]
        return f"{year}-{month}"
    return "unknown"


def clean_address(x):
    if pd.isna(x):
        return "unknown"
    if isinstance(x, list):
        return " > ".join(x)
    return str(x)


def clean_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    df["GrossSquareMeters"] = df["GrossSquareMeters"].apply(clean_square_meters)
    df["NetSquareMeters"] = df["NetSquareMeters"].apply(clean_square_meters)

    df["BuildingAge"] = df["BuildingAge"].apply(clean_building_age)

    df["NumberOfRooms"] = df["NumberOfRooms"].apply(clean_rooms)

    df["AdCreationDate"] = df["AdCreationDate"].apply(clean_date)

    df["address"] = df["address"].apply(clean_address)

    df["Subscription"] = df["Subscription"].fillna("unknown")

    return df


def feature_cross(df: pd.DataFrame) -> pd.DataFrame:
    df["district_heating"] = (
        df["district"].astype(str) + "_" + df["HeatingType"].astype(str)
    )

    df["district_floor"] = (
        df["district"].astype(str) + "_" + df["FloorLocation"].astype(str)
    )

    return df


def build_preprocessor():
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    low_card_cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("low_cat", low_card_cat_pipeline, LOW_CARD_CAT),
        ],
        remainder="drop",
    )

    hasher = FeatureHasher(n_features=2**14, input_type="string")

    return preprocessor, hasher


def preprocess_and_split():
    df = load_data(RAW_DATA_PATH)
    df = clean_raw_columns(df)
    df = feature_cross(df)
    df = create_target(df)

    reference_dir = Path("data/reference")
    reference_dir.mkdir(parents=True, exist_ok=True)

    df_reference = df[
        ["district", "address", "HeatingType", "FloorLocation", "price_category"]
    ].copy()

    df_reference.to_parquet(reference_dir / "reference.parquet", index=False)

    X = df[NUMERIC_FEATURES + LOW_CARD_CAT + HIGH_CARD_CAT]
    y = df["price_category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor, hasher = build_preprocessor()

    X_train_base = preprocessor.fit_transform(X_train)
    X_test_base = preprocessor.transform(X_test)

    X_train_high = X_train[HIGH_CARD_CAT].astype(str).values.tolist()

    X_test_high = X_test[HIGH_CARD_CAT].astype(str).values.tolist()

    X_train_hash = hasher.transform(X_train_high)
    X_test_hash = hasher.transform(X_test_high)

    from scipy.sparse import hstack

    X_train_final = hstack([X_train_base, X_train_hash])
    X_test_final = hstack([X_test_base, X_test_hash])

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    joblib.dump((X_train_final, y_train), f"{PROCESSED_DIR}/train.pkl")
    joblib.dump((X_test_final, y_test), f"{PROCESSED_DIR}/test.pkl")
    joblib.dump(preprocessor, f"{PROCESSED_DIR}/preprocessor.pkl")
    joblib.dump(hasher, f"{PROCESSED_DIR}/hasher.pkl")

    print("Preprocessing complete.")
    print(f"Train shape: {X_train_final.shape}")
    print(f"Test shape: {X_test_final.shape}")


if __name__ == "__main__":
    preprocess_and_split()
