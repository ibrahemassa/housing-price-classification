from unittest.mock import MagicMock, patch

from src.utils.model_loader import load_model


class TestModelLoader:
    """Tests for model_loader.py"""

    @patch("src.utils.model_loader.mlflow.sklearn.load_model")
    def test_load_model_production(self, mock_load_model):
        """Test loading model with production alias."""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        result = load_model("production")

        mock_load_model.assert_called_once_with(
            "models:/HousingPriceClassifier@production"
        )
        assert result == mock_model

    @patch("src.utils.model_loader.mlflow.sklearn.load_model")
    def test_load_model_staging(self, mock_load_model):
        """Test loading model with staging alias."""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        result = load_model("staging")

        mock_load_model.assert_called_once_with(
            "models:/HousingPriceClassifier@staging"
        )
        assert result == mock_model

    @patch("src.utils.model_loader.mlflow.sklearn.load_model")
    def test_load_model_default_alias(self, mock_load_model):
        """Test loading model with default alias (production)."""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        result = load_model()

        mock_load_model.assert_called_once_with(
            "models:/HousingPriceClassifier@production"
        )
        assert result == mock_model

    @patch("src.utils.model_loader.mlflow.sklearn.load_model")
    def test_load_model_custom_alias(self, mock_load_model):
        """Test loading model with custom alias."""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        result = load_model("custom_alias")

        mock_load_model.assert_called_once_with(
            "models:/HousingPriceClassifier@custom_alias"
        )
        assert result == mock_model

    # @patch("src.utils.model_loader.mlflow.sklearn.load_model")
    # def test_load_model_handles_exception(self, mock_load_model):
    #     """Test that load_model properly propagates exceptions."""
    #     mock_load_model.side_effect = Exception("Model not found")
    #
    #     model = load_model(alias="Production")
    #
    #     assert model is not None

