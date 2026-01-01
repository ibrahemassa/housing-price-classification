import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from src.utils.production_logger import log_input, log_prediction


class TestProductionLogger:
    """Tests for production_logger.py"""

    @pytest.fixture
    def temp_log_dir(self):
        """Create temporary directory for log files."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)

    def test_log_input_creates_new_file(self, temp_log_dir, monkeypatch):
        """Test that log_input creates a new file when it doesn't exist."""
        monkeypatch.setattr("src.utils.production_logger.LOG_DIR", Path(temp_log_dir))
        monkeypatch.setattr(
            "src.utils.production_logger.INPUT_LOG",
            Path(temp_log_dir) / "inputs.parquet",
        )

        data = {"GrossSquareMeters": 100.0, "NetSquareMeters": 80.0, "district": "A"}

        log_input(data)

        assert (
            Path(temp_log_dir) / "inputs.parquet"
            == Path(temp_log_dir) / "inputs.parquet"
        )
        df = pd.read_parquet(Path(temp_log_dir) / "inputs.parquet")
        assert len(df) == 1
        assert df["GrossSquareMeters"].iloc[0] == 100.0
        assert "timestamp" in df.columns

    def test_log_input_appends_to_existing_file(self, temp_log_dir, monkeypatch):
        """Test that log_input appends to existing file."""
        monkeypatch.setattr("src.utils.production_logger.LOG_DIR", Path(temp_log_dir))
        monkeypatch.setattr(
            "src.utils.production_logger.INPUT_LOG",
            Path(temp_log_dir) / "inputs.parquet",
        )

        # Create initial file
        initial_data = pd.DataFrame(
            [{"GrossSquareMeters": 50.0, "timestamp": datetime.utcnow()}]
        )
        initial_data.to_parquet(Path(temp_log_dir) / "inputs.parquet")

        data = {"GrossSquareMeters": 100.0, "NetSquareMeters": 80.0}
        log_input(data)

        df = pd.read_parquet(Path(temp_log_dir) / "inputs.parquet")
        assert len(df) == 2
        assert df["GrossSquareMeters"].iloc[1] == 100.0

    def test_log_prediction_creates_new_file(self, temp_log_dir, monkeypatch):
        """Test that log_prediction creates a new file when it doesn't exist."""
        monkeypatch.setattr("src.utils.production_logger.LOG_DIR", Path(temp_log_dir))
        monkeypatch.setattr(
            "src.utils.production_logger.PRED_LOG",
            Path(temp_log_dir) / "predictions.parquet",
        )

        log_prediction(1)

        df = pd.read_parquet(Path(temp_log_dir) / "predictions.parquet")
        assert len(df) == 1
        assert df["prediction"].iloc[0] == 1
        assert "timestamp" in df.columns

    def test_log_prediction_appends_to_existing_file(self, temp_log_dir, monkeypatch):
        """Test that log_prediction appends to existing file."""
        monkeypatch.setattr("src.utils.production_logger.LOG_DIR", Path(temp_log_dir))
        monkeypatch.setattr(
            "src.utils.production_logger.PRED_LOG",
            Path(temp_log_dir) / "predictions.parquet",
        )

        # Create initial file
        initial_data = pd.DataFrame([{"prediction": 0, "timestamp": datetime.utcnow()}])
        initial_data.to_parquet(Path(temp_log_dir) / "predictions.parquet")

        log_prediction(2)

        df = pd.read_parquet(Path(temp_log_dir) / "predictions.parquet")
        assert len(df) == 2
        assert df["prediction"].iloc[1] == 2

    def test_log_prediction_with_different_values(self, temp_log_dir, monkeypatch):
        """Test logging different prediction values."""
        monkeypatch.setattr("src.utils.production_logger.LOG_DIR", Path(temp_log_dir))
        monkeypatch.setattr(
            "src.utils.production_logger.PRED_LOG",
            Path(temp_log_dir) / "predictions.parquet",
        )

        for pred in [0, 1, 2]:
            log_prediction(pred)

        df = pd.read_parquet(Path(temp_log_dir) / "predictions.parquet")
        assert len(df) == 3
        assert set(df["prediction"].values) == {0, 1, 2}


