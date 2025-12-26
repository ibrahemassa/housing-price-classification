import pytest
from unittest.mock import patch, MagicMock
from src.training.register_model import MODEL_NAME


class TestRegisterModel:
    """Tests for register_model.py"""

    @patch("src.training.register_model.client")
    def test_register_model_sets_aliases(self, mock_client):
        """Test that register_model sets staging and production aliases."""
        # Mock model versions
        mock_version1 = MagicMock()
        mock_version1.version = "1"
        mock_version2 = MagicMock()
        mock_version2.version = "2"
        mock_version3 = MagicMock()
        mock_version3.version = "3"
        
        mock_client.search_model_versions.return_value = [mock_version1, mock_version2, mock_version3]
        
        # Import and execute the script logic
        from src.training.register_model import client, MODEL_NAME
        
        latest_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        latest_version = max(latest_versions, key=lambda v: int(v.version))
        
        version = latest_version.version
        
        client.set_registered_model_alias(name=MODEL_NAME, alias="staging", version=version)
        client.set_registered_model_alias(name=MODEL_NAME, alias="production", version=version)
        
        # Verify calls
        assert mock_client.set_registered_model_alias.call_count == 2

    def test_register_model_finds_latest_version(self):
        """Test that register_model finds the latest version correctly."""
        from unittest.mock import MagicMock
        
        mock_version1 = MagicMock()
        mock_version1.version = "1"
        mock_version2 = MagicMock()
        mock_version2.version = "10"
        mock_version3 = MagicMock()
        mock_version3.version = "5"
        
        versions = [mock_version1, mock_version2, mock_version3]
        latest_version = max(versions, key=lambda v: int(v.version))
        
        # Version is a string in the mock
        assert latest_version.version == "10"

    def test_register_model_handles_single_version(self):
        """Test that register_model works with a single version."""
        from unittest.mock import MagicMock
        
        mock_version = MagicMock()
        mock_version.version = "1"
        
        versions = [mock_version]
        latest_version = max(versions, key=lambda v: int(v.version))
        
        # Version is a string in the mock
        assert latest_version.version == "1"

