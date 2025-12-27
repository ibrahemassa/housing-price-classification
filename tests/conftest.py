import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import FeatureHasher
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_housing_data():
    """Sample housing data for testing."""
    return pd.DataFrame(
        {
            "GrossSquareMeters": [100.0, 150.0, 200.0],
            "NetSquareMeters": [80.0, 120.0, 160.0],
            "BuildingAge": [5, 10, 15],
            "NumberOfRooms": [2.0, 3.0, 4.0],
            "district": ["A", "B", "C"],
            "HeatingType": ["Type1", "Type2", "Type1"],
            "StructureType": ["Struct1", "Struct2", "Struct1"],
            "FloorLocation": ["Floor1", "Floor2", "Floor1"],
            "address": ["Address1", "Address2", "Address3"],
            "AdCreationDate": ["2024-01", "2024-02", "2024-03"],
            "Subscription": ["Sub1", "Sub2", "Sub1"],
            "price": ["100,000 TL", "200,000 TL", "300,000 TL"],
        }
    )


@pytest.fixture
def sample_housing_input():
    """Sample housing input for API testing."""
    return {
        "GrossSquareMeters": 100.0,
        "NetSquareMeters": 80.0,
        "BuildingAge": 5.0,
        "NumberOfRooms": 2.0,
        "district": "A",
        "HeatingType": "Type1",
        "StructureType": "Struct1",
        "FloorLocation": "Floor1",
        "address": "Address1",
        "AdCreationDate": "2024-01",
        "Subscription": "Sub1",
    }


@pytest.fixture
def mock_model():
    """Mock sklearn model."""
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    # Fit with dummy data
    X = np.random.rand(10, 5)
    y = np.random.randint(0, 3, 10)
    model.fit(X, y)
    return model


@pytest.fixture
def mock_preprocessor():
    """Mock preprocessor transformer."""
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
            (
                "num",
                numeric_pipeline,
                [
                    "GrossSquareMeters",
                    "NetSquareMeters",
                    "BuildingAge",
                    "NumberOfRooms",
                ],
            ),
            (
                "low_cat",
                low_card_cat_pipeline,
                ["district", "HeatingType", "StructureType", "FloorLocation"],
            ),
        ],
        remainder="drop",
    )

    # Fit with dummy data
    X = pd.DataFrame(
        {
            "GrossSquareMeters": [100.0, 150.0, 200.0],
            "NetSquareMeters": [80.0, 120.0, 160.0],
            "BuildingAge": [5, 10, 15],
            "NumberOfRooms": [2.0, 3.0, 4.0],
            "district": ["A", "B", "C"],
            "HeatingType": ["Type1", "Type2", "Type1"],
            "StructureType": ["Struct1", "Struct2", "Struct1"],
            "FloorLocation": ["Floor1", "Floor2", "Floor1"],
        }
    )
    preprocessor.fit(X)
    return preprocessor


@pytest.fixture
def mock_hasher():
    """Mock feature hasher."""
    hasher = FeatureHasher(n_features=2**14, input_type="string")
    return hasher


@pytest.fixture
def sample_reference_data():
    """Sample reference data for monitoring."""
    return pd.DataFrame(
        {
            "district": ["A", "B", "C", "A", "B"] * 20,
            "address": [f"Address{i}" for i in range(100)],
            "HeatingType": ["Type1", "Type2", "Type1", "Type2", "Type1"] * 20,
            "FloorLocation": ["Floor1", "Floor2", "Floor1", "Floor2", "Floor1"] * 20,
            "price_category": [0, 1, 2, 0, 1] * 20,
            "target": [0, 1, 2, 0, 1] * 20,
        }
    )


@pytest.fixture
def sample_production_data():
    """Sample production data for monitoring."""
    return pd.DataFrame(
        {
            "district": ["A", "B", "C", "A", "B"] * 20,
            "address": [f"Address{i}" for i in range(100)],
            "HeatingType": ["Type1", "Type2", "Type1", "Type2", "Type1"] * 20,
            "FloorLocation": ["Floor1", "Floor2", "Floor1", "Floor2", "Floor1"] * 20,
        }
    )


@pytest.fixture
def sample_predictions():
    """Sample predictions for monitoring."""
    return pd.DataFrame({"prediction": [0, 1, 2, 0, 1] * 20})
