import pytest
import pandas as pd
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch
from src.monitoring.plot import (
    plot_categorical_distribution,
    plot_prediction_distribution,
    plot_cardinality_growth,
    timestamp
)


class TestPlot:
    """Tests for plot.py"""

    @pytest.fixture
    def temp_plot_dir(self):
        """Create temporary directory for plot files."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)

    def test_timestamp_format(self):
        """Test that timestamp returns correct format."""
        ts = timestamp()
        assert len(ts) == 15  # YYYYMMDD_HHMMSS (8+1+6)
        assert "_" in ts
        assert ts.count("_") == 1
        assert ts.startswith("2025") or ts.startswith("2024")  # Year check

    def test_plot_categorical_distribution(self, temp_plot_dir, sample_reference_data, sample_production_data, monkeypatch):
        """Test plotting categorical distribution."""
        monkeypatch.setattr("src.monitoring.plot.PLOT_DIR", Path(temp_plot_dir))
        
        ref = sample_reference_data
        prod = sample_production_data
        
        path = plot_categorical_distribution(ref, prod, "district")
        
        path_str = str(path)
        assert Path(path).exists()
        assert path_str.endswith(".png")
        assert "district_distribution" in path_str

    def test_plot_prediction_distribution(self, temp_plot_dir, sample_reference_data, sample_predictions, monkeypatch):
        """Test plotting prediction distribution."""
        monkeypatch.setattr("src.monitoring.plot.PLOT_DIR", Path(temp_plot_dir))
        
        ref_preds = sample_reference_data["target"]
        prod_preds = sample_predictions["prediction"]
        
        path = plot_prediction_distribution(ref_preds, prod_preds)
        
        path_str = str(path)
        assert Path(path).exists()
        assert path_str.endswith(".png")
        assert "prediction_drift" in path_str

    def test_plot_cardinality_growth(self, temp_plot_dir, sample_reference_data, sample_production_data, monkeypatch):
        """Test plotting cardinality growth."""
        monkeypatch.setattr("src.monitoring.plot.PLOT_DIR", Path(temp_plot_dir))
        
        ref = sample_reference_data
        prod = sample_production_data
        
        path = plot_cardinality_growth(ref, prod, "address")
        
        path_str = str(path)
        assert Path(path).exists()
        assert path_str.endswith(".png")
        assert "address_cardinality" in path_str

    def test_plot_categorical_distribution_handles_missing_values(self, temp_plot_dir, monkeypatch):
        """Test that plot handles missing values in data."""
        monkeypatch.setattr("src.monitoring.plot.PLOT_DIR", Path(temp_plot_dir))
        
        ref = pd.DataFrame({"district": ["A", "B", "C"]})
        prod = pd.DataFrame({"district": ["A", "B"]})
        
        path = plot_categorical_distribution(ref, prod, "district")
        
        assert Path(path).exists()

    def test_plot_cardinality_growth_with_different_cardinalities(self, temp_plot_dir, monkeypatch):
        """Test cardinality growth with different unique value counts."""
        monkeypatch.setattr("src.monitoring.plot.PLOT_DIR", Path(temp_plot_dir))
        
        ref = pd.DataFrame({"address": [f"addr_{i}" for i in range(10)]})
        prod = pd.DataFrame({"address": [f"addr_{i}" for i in range(20)]})
        
        path = plot_cardinality_growth(ref, prod, "address")
        
        path_str = str(path)
        assert Path(path).exists()
        assert "address_cardinality" in path_str

