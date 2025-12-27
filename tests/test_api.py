from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.api.api import CATEGORIES, HousingInput, app, preprocess_input


class TestAPI:
    """Tests for api.py"""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def sample_input_data(self):
        """Sample input data for API."""
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

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    @patch("src.api.api.log_input")
    @patch("src.api.api.log_prediction")
    @patch("src.api.api.preprocess_input")
    def test_predict_endpoint(
        self, mock_preprocess, mock_log_pred, mock_log_input, client, sample_input_data
    ):
        """Test predict endpoint."""
        from scipy.sparse import csr_matrix

        # Setup mocks
        mock_preprocess.return_value = csr_matrix(np.random.rand(1, 100))

        # Need to patch the model at the module level
        with patch("src.api.api.model") as mock_model:
            mock_model.predict.return_value = np.array([1])

            response = client.post("/predict", json=sample_input_data)

            assert response.status_code == 200
            assert "price_category" in response.json()
            assert response.json()["price_category"] in CATEGORIES.values()
            mock_log_input.assert_called_once()
            mock_log_pred.assert_called_once()

    @patch("src.api.api.hasher")
    @patch("src.api.api.preprocessor")
    def test_preprocess_input(self, mock_preprocessor, mock_hasher, sample_input_data):
        """Test preprocess_input function."""
        from scipy.sparse import csr_matrix

        mock_preprocessor.transform.return_value = np.random.rand(1, 10)
        mock_hasher.transform.return_value = csr_matrix(np.random.rand(1, 100))

        result = preprocess_input(sample_input_data)

        assert result is not None
        mock_preprocessor.transform.assert_called_once()
        mock_hasher.transform.assert_called_once()

    @patch("src.api.api.log_input")
    @patch("src.api.api.log_prediction")
    @patch("src.api.api.preprocess_input")
    def test_predict_different_categories(
        self, mock_preprocess, mock_log_pred, mock_log_input, client, sample_input_data
    ):
        """Test predict endpoint with different prediction categories."""
        from scipy.sparse import csr_matrix

        mock_preprocess.return_value = csr_matrix(np.random.rand(1, 100))

        with patch("src.api.api.model") as mock_model:
            for category_id in [0, 1, 2]:
                mock_model.predict.return_value = np.array([category_id])

                response = client.post("/predict", json=sample_input_data)

                assert response.status_code == 200
                assert response.json()["price_category"] == CATEGORIES[category_id]

    def test_predict_invalid_input(self, client):
        """Test predict endpoint with invalid input."""
        invalid_data = {"invalid": "data"}

        response = client.post("/predict", json=invalid_data)

        assert response.status_code == 422  # Validation error

    def test_housing_input_model(self, sample_input_data):
        """Test HousingInput Pydantic model."""
        input_model = HousingInput(**sample_input_data)

        assert input_model.GrossSquareMeters == 100.0
        assert input_model.district == "A"
        assert input_model.address == "Address1"

    def test_housing_input_model_validation(self):
        """Test HousingInput model validation."""
        invalid_data = {
            "GrossSquareMeters": "not_a_number",  # Should be float
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

        with pytest.raises(ValidationError):  # Pydantic validation error
            HousingInput(**invalid_data)
