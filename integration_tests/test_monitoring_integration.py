"""
Integration tests for the monitoring flow.
Tests the complete monitoring workflow including drift detection.
"""

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


class TestMonitoringIntegration:
    """Integration tests for monitoring workflow."""

    @pytest.fixture
    def temp_monitoring_dir(self):
        """Create temporary directory for monitoring data."""
        temp_dir = tempfile.mkdtemp()
        os.makedirs(f"{temp_dir}/data/reference", exist_ok=True)
        os.makedirs(f"{temp_dir}/data/production", exist_ok=True)
        os.makedirs(f"{temp_dir}/data/monitoring/plots", exist_ok=True)
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def reference_data(self):
        """Create reference dataset."""
        return pd.DataFrame(
            {
                "district": ["A", "B", "C"] * 100,
                "address": [f"addr_{i}" for i in range(300)],
                "HeatingType": ["Type1", "Type2", "Type1"] * 100,
                "FloorLocation": ["Floor1", "Floor2", "Floor1"] * 100,
                "price_category": [0, 1, 2] * 100,
                "target": [0, 1, 2] * 100,
            }
        )

    @pytest.fixture
    def production_data(self):
        """Create production dataset."""
        return pd.DataFrame(
            {
                "district": ["A", "B", "C"] * 50,
                "address": [f"addr_{i}" for i in range(150)],
                "HeatingType": ["Type1", "Type2", "Type1"] * 50,
                "FloorLocation": ["Floor1", "Floor2", "Floor1"] * 50,
            }
        )

    @pytest.fixture
    def production_predictions(self):
        """Create production predictions."""
        return pd.DataFrame({"prediction": [0, 1, 2] * 50})

    def test_monitoring_flow_with_real_data(
        self,
        temp_monitoring_dir,
        reference_data,
        production_data,
        production_predictions,
    ):
        """Test complete monitoring flow with real data structures."""
        from src.monitoring.monitor_flow import categorical_psi, prediction_drift

        # Save reference data
        ref_path = Path(temp_monitoring_dir) / "data/reference/reference.parquet"
        reference_data.to_parquet(ref_path, index=False)

        # Test PSI calculation
        psi = categorical_psi(reference_data["district"], production_data["district"])
        assert isinstance(psi, (int, float))
        assert psi >= 0

        # Test prediction drift
        drift = prediction_drift(
            reference_data["price_category"], production_predictions["prediction"]
        )
        assert isinstance(drift, (int, float))
        assert drift >= 0

    @patch("src.monitoring.monitor_flow.Path.exists")
    @patch("src.monitoring.monitor_flow.pd.read_parquet")
    @patch("src.monitoring.monitor_flow.plot_categorical_distribution")
    @patch("src.monitoring.monitor_flow.plot_cardinality_growth")
    @patch("src.monitoring.monitor_flow.plot_prediction_distribution")
    @patch("src.monitoring.monitor_flow.create_markdown_artifact")
    @patch("src.monitoring.monitor_flow.subprocess.run")
    def test_monitoring_flow_integration(
        self,
        mock_subprocess,
        mock_artifact,
        mock_pred_plot,
        mock_card_plot,
        mock_dist_plot,
        mock_read_parquet,
        mock_exists,
        reference_data,
        production_data,
        production_predictions,
    ):
        """Test that monitoring flow integrates all components."""
        from src.monitoring.monitor_flow import run_monitor

        mock_exists.return_value = True
        mock_read_parquet.side_effect = [
            reference_data,
            production_data,
            production_predictions,
        ]
        mock_dist_plot.return_value = "path1.png"
        mock_card_plot.return_value = "path2.png"
        mock_pred_plot.return_value = "path3.png"
        mock_subprocess.return_value = MagicMock(returncode=0)

        # Execute monitoring flow
        run_monitor()

        # Verify all components were called
        mock_read_parquet.assert_called()
        mock_dist_plot.assert_called()
        mock_card_plot.assert_called()
        mock_pred_plot.assert_called()

    def test_plot_functions_integration(
        self, temp_monitoring_dir, reference_data, production_data
    ):
        """Test that plotting functions work together."""
        from src.monitoring.plot import (
            plot_cardinality_growth,
            plot_categorical_distribution,
            plot_prediction_distribution,
        )

        with patch(
            "src.monitoring.plot.PLOT_DIR",
            Path(temp_monitoring_dir) / "data/monitoring/plots",
        ):
            # Test categorical distribution plot
            path1 = plot_categorical_distribution(
                reference_data, production_data, "district"
            )
            assert Path(path1).exists()

            # Test cardinality growth plot
            path2 = plot_cardinality_growth(reference_data, production_data, "address")
            assert Path(path2).exists()

            # Test prediction distribution plot
            ref_preds = reference_data["target"]
            prod_preds = pd.Series([0, 1, 2] * 50)
            path3 = plot_prediction_distribution(ref_preds, prod_preds)
            assert Path(path3).exists()

    def test_drift_detection_thresholds(self):
        """Test that drift detection uses correct thresholds."""
        from src.monitoring.monitor_flow import (
            CARDINALITY_THRESHOLD,
            PRED_DRIFT_THRESHOLD,
            PSI_THRESHOLD,
        )

        assert PSI_THRESHOLD == 0.25
        assert CARDINALITY_THRESHOLD == 1.3
        assert PRED_DRIFT_THRESHOLD == 0.15
