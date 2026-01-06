"""
Shared fixtures for integration tests.
"""

import os
import shutil
import tempfile

import pandas as pd
import pytest


@pytest.fixture
def sample_housing_data():
    """Sample housing data for integration testing."""
    return pd.DataFrame(
        {
            "GrossSquareMeters": [100.0, 150.0, 200.0] * 10,
            "NetSquareMeters": [80.0, 120.0, 160.0] * 10,
            "BuildingAge": [5, 10, 15] * 10,
            "NumberOfRooms": [2.0, 3.0, 4.0] * 10,
            "district": ["A", "B", "C"] * 10,
            "HeatingType": ["Type1", "Type2", "Type1"] * 10,
            "StructureType": ["Struct1", "Struct2", "Struct1"] * 10,
            "FloorLocation": ["Floor1", "Floor2", "Floor1"] * 10,
            "address": [f"Address{i}" for i in range(30)],
            "AdCreationDate": ["2024-01", "2024-02", "2024-03"] * 10,
            "Subscription": ["Sub1", "Sub2", "Sub1"] * 10,
            "price": ["100,000 TL", "200,000 TL", "300,000 TL"] * 10,
        }
    )


@pytest.fixture
def temp_integration_dir():
    """Create temporary directory for integration tests."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_mlflow_client():
    """Mock MLflow client for integration tests."""
    from unittest.mock import MagicMock

    mock_client = MagicMock()
    return mock_client


@pytest.fixture
def complete_project_structure(temp_integration_dir):
    """Create a complete project structure for integration testing."""
    structure = {
        "data/raw": [],
        "data/processed": [],
        "data/reference": [],
        "data/production": [],
        "data/monitoring/plots": [],
        "models": [],
        "src/api": [],
        "src/training": [],
        "src/monitoring": [],
        "src/utils": [],
    }

    for dir_path in structure:
        full_path = os.path.join(temp_integration_dir, dir_path)
        os.makedirs(full_path, exist_ok=True)

    return temp_integration_dir




