import pandas as pd

from src.monitoring.monitor_flow import (
    categorical_psi,
    prediction_drift,
)


class TestMonitorFlow:
    """Tests for monitor_flow.py"""

    def test_categorical_psi_identical_distributions(self):
        """Test PSI with identical distributions."""
        ref = pd.Series(["A", "B", "C"] * 10)
        prod = pd.Series(["A", "B", "C"] * 10)

        psi = categorical_psi(ref, prod)

        # PSI should be close to 0 for identical distributions
        assert psi < 0.1

    def test_categorical_psi_different_distributions(self):
        """Test PSI with different distributions."""
        ref = pd.Series(["A"] * 100 + ["B"] * 100)
        prod = pd.Series(["A"] * 10 + ["B"] * 190)

        psi = categorical_psi(ref, prod)

        assert psi > 0.05

    def test_categorical_psi_handles_missing_categories(self):
        """Test PSI handles missing categories in one distribution."""
        ref = pd.Series(["A", "B", "C"] * 10)
        prod = pd.Series(["A", "B"] * 15)  # Missing "C"

        psi = categorical_psi(ref, prod)

        assert psi >= 0

    def test_prediction_drift_identical(self):
        """Test prediction drift with identical distributions."""
        ref = pd.Series([0, 1, 2] * 10)
        prod = pd.Series([0, 1, 2] * 10)

        drift = prediction_drift(ref, prod)

        # KL divergence should be close to 0 for identical distributions
        assert drift < 0.1

    def test_prediction_drift_different(self):
        """Test prediction drift with different distributions."""
        ref = pd.Series([0] * 100 + [1] * 50 + [2] * 50)
        prod = pd.Series([0] * 50 + [1] * 100 + [2] * 50)

        drift = prediction_drift(ref, prod)

        # KL divergence should be higher for different distributions
        assert drift > 0.1

    # @patch("src.monitoring.monitor_flow.training_pipeline")
    # def test_retraining_triggered(mock_train):
    #     monitoring_flow()
    #     mock_train.assert_called_once()

    # @patch("src.monitoring.monitor_flow.Path.exists")
    # @patch("src.monitoring.monitor_flow.pd.read_parquet")
    # @patch("src.monitoring.monitor_flow.create_markdown_artifact")
    # def test_run_monitor_with_drift(
    #     self,
    #     # mock_subprocess,
    #     # mock_artifact,
    #     # mock_pred_plot,
    #     # mock_card_plot,
    #     # mock_dist_plot,
    #     mock_read_parquet,
    #     mock_exists,
    #     sample_reference_data,
    # ):
    #     """Test monitoring when drift is detected."""
    #     # Create production data with high drift
    #     prod_inputs = pd.DataFrame(
    #         {
    #             "district": ["X"] * 200,  # Completely different district
    #             "address": [f"NewAddr{i}" for i in range(200)],  # High cardinality
    #         }
    #     )
    #     prod_preds = pd.DataFrame(
    #         {"prediction": [2] * 200}  # Different prediction distribution
    #     )
    #
    #     mock_exists.return_value = True
    #     mock_read_parquet.side_effect = [sample_reference_data, prod_inputs, prod_preds]
    #     # mock_dist_plot.return_value = "path1.png"
    #     # mock_card_plot.return_value = "path2.png"
    #     # mock_pred_plot.return_value = "path3.png"
    #
    #     run_monitor()
    #
    #     # Should trigger retraining if alerts >= 2
    # Note: The actual logic checks if alerts >= 2, and with high drift
    # we should have multiple alerts

    # @patch("src.monitoring.monitor_flow.subprocess.run")
    # def test_run_monitor_no_production_data(self, mock_run):
    #     mock_run.return_value.returncode = 0
    #     run_monitor()
    #
    # @patch("src.monitoring.monitor_flow.run_monitor")
    # def test_monitoring_flow(self, mock_run_monitor):
    #     """Test monitoring_flow calls run_monitor."""
    #     monitoring_flow()
    #     mock_run_monitor.assert_called_once()
    #
    # def test_categorical_psi_edge_cases(self):
    #     """Test PSI with edge cases."""
    #     # Empty series
    #     ref = pd.Series([])
    #     prod = pd.Series([])
    #     psi = categorical_psi(ref, prod)
    #     assert psi >= 0
    #
    #     # Single category
    #     ref = pd.Series(["A"] * 10)
    #     prod = pd.Series(["A"] * 10)
    #     psi = categorical_psi(ref, prod)
    #     assert psi < 0.1
    #
    # def test_prediction_drift_edge_cases(self):
    #     """Test prediction drift with edge cases."""
    #     # Single class
    #     ref = pd.Series([0] * 10)
    #     prod = pd.Series([0] * 10)
    #     drift = prediction_drift(ref, prod)
    #     assert drift >= 0
    #
    #     # Missing classes in one distribution
    #     ref = pd.Series([0, 1, 2] * 10)
    #     prod = pd.Series([0, 1] * 15)
    #     drift = prediction_drift(ref, prod)
    #     assert drift >= 0
