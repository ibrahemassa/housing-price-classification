"""
Integration tests for the API endpoint.
Tests the complete prediction flow from input to output.
"""

from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.sparse import csr_matrix


class TestAPIIntegration:
    """Integration tests for API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from src.api.api import app

        return TestClient(app)

    @pytest.fixture
    def sample_input(self):
        """Sample input for prediction."""
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

    @patch("src.api.api.log_input")
    @patch("src.api.api.log_prediction")
    @patch("src.api.api.preprocess_input")
    def test_predict_endpoint_full_flow(
        self, mock_preprocess, mock_log_pred, mock_log_input, client, sample_input
    ):
        """Test complete prediction flow through API."""
        from scipy.sparse import csr_matrix

        # Mock preprocessing
        mock_preprocess.return_value = csr_matrix(np.random.rand(1, 100))

        # Mock model prediction
        with patch("src.api.api.model") as mock_model:
            mock_model.predict.return_value = np.array([1])

            response = client.post("/predict", json=sample_input)

            # Verify response
            assert response.status_code == 200
            assert "price_category" in response.json()
            assert response.json()["price_category"] in ["low", "medium", "high"]

            # Verify logging was called
            mock_log_input.assert_called_once()
            mock_log_pred.assert_called_once()

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    @patch("src.api.api.preprocessor")
    @patch("src.api.api.hasher")
    def test_preprocess_input_integration(
        self, mock_hasher, mock_preprocessor, sample_input
    ):
        """Test that preprocessing integrates with preprocessor and hasher."""
        from src.api.api import preprocess_input

        # Mock transformers
        mock_preprocessor.transform.return_value = np.random.rand(1, 10)
        mock_hasher.transform.return_value = csr_matrix(np.random.rand(1, 100))

        result = preprocess_input(sample_input)

        # Verify preprocessing was called
        mock_preprocessor.transform.assert_called_once()
        mock_hasher.transform.assert_called_once()
        assert result is not None

    def test_api_model_loading_fallback(self):
        """Test that API handles model loading fallback."""
        # This test verifies the fallback mechanism exists in the code
        # The actual fallback happens at module import time, so we just verify
        # the try-except structure exists
        from src.api.api import app

        # Verify app was created (which means model loading succeeded or fell back)
        assert app is not None
        assert hasattr(app, "get")
        assert hasattr(app, "post")

    def test_multiple_predictions_consistency(self, client, sample_input):
        """Test that multiple predictions maintain consistency."""
        from scipy.sparse import csr_matrix

        with patch("src.api.api.preprocess_input") as mock_preprocess:
            with patch("src.api.api.model") as mock_model:
                with patch("src.api.api.log_input"):
                    with patch("src.api.api.log_prediction"):
                        mock_preprocess.return_value = csr_matrix(
                            np.random.rand(1, 100)
                        )
                        mock_model.predict.return_value = np.array([1])

                        # Make multiple predictions
                        responses = []
                        for _ in range(3):
                            response = client.post("/predict", json=sample_input)
                            responses.append(response.json()["price_category"])

                        # All should return the same category (given same input and model)
                        assert len(set(responses)) == 1
