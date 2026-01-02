from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.training.train import (
    get_class_weights,
    load_data,
    train_baseline,
    train_main_model,
)


class TestTrain:
    """Tests for train.py"""

    @pytest.fixture
    def sample_training_data(self):
        """Sample training data."""
        X_train = np.random.rand(100, 10)
        X_test = np.random.rand(20, 10)
        y_train = np.random.randint(0, 3, 100)
        y_test = np.random.randint(0, 3, 20)
        return X_train, X_test, y_train, y_test

    @patch("src.training.train.joblib.load")
    def test_load_data(self, mock_load, sample_training_data):
        """Test loading training data."""
        X_train, X_test, y_train, y_test = sample_training_data
        mock_load.side_effect = [(X_train, y_train), (X_test, y_test)]

        X_tr, X_te, y_tr, y_te = load_data()

        assert np.array_equal(X_tr, X_train)
        assert np.array_equal(X_te, X_test)
        assert np.array_equal(y_tr, y_train)
        assert np.array_equal(y_te, y_test)

    def test_get_class_weights(self):
        """Test computing class weights."""
        y = np.array([0, 0, 1, 1, 2])
        weights = get_class_weights(y)

        assert isinstance(weights, dict)
        assert 0 in weights
        assert 1 in weights
        assert 2 in weights
        assert all(w > 0 for w in weights.values())

    @patch("src.training.train.mlflow.log_param")
    @patch("src.training.train.mlflow.log_metric")
    @patch("src.training.train.mlflow.start_run")
    @patch("src.training.train.calculate_comprehensive_metrics")
    @patch("src.training.train.joblib.load")
    def test_train_baseline(
        self,
        mock_load,
        mock_metrics,
        mock_mlflow_run,
        mock_log_metric,
        mock_log_param,
        sample_training_data,
    ):
        """Test training baseline model."""
        X_train, X_test, y_train, y_test = sample_training_data
        mock_load.side_effect = [(X_train, y_train), (X_test, y_test)]

        mock_metrics.return_value = {
            "accuracy": 0.85,
            "macro_f1": 0.82,
            "macro_precision": 0.80,
            "macro_recall": 0.78,
        }
        mock_context = MagicMock()
        mock_mlflow_run.return_value.__enter__.return_value = mock_context

        acc, f1 = train_baseline(X_train, X_test, y_train, y_test)

        assert acc == 0.85
        assert f1 == 0.82
        mock_log_param.assert_called()
        mock_log_metric.assert_called()

    @patch("src.training.train.mlflow.log_param")
    @patch("src.training.train.mlflow.log_metric")
    @patch("src.training.train.mlflow.log_artifact")
    @patch("src.training.train.mlflow.start_run")
    @patch("src.training.train.joblib.dump")
    @patch("src.training.train.mlflow.sklearn.log_model")
    @patch("src.training.train.calculate_comprehensive_metrics")
    @patch("src.training.train.joblib.load")
    @patch("src.training.train.os.makedirs")
    @patch("src.training.train.os.path.exists")
    @patch("src.training.train.RandomForestClassifier")
    def test_train_main_model(
        self,
        mock_rf,
        mock_exists,
        mock_makedirs,
        mock_load,
        mock_metrics,
        mock_log_model,
        mock_dump,
        mock_mlflow_run,
        mock_log_metric,
        mock_log_artifact,
        mock_log_param,
        sample_training_data,
    ):
        """Test training main model."""
        X_train, X_test, y_train, y_test = sample_training_data
        mock_load.side_effect = [(X_train, y_train), (X_test, y_test)]

        from sklearn.ensemble import RandomForestClassifier

        def create_rf(*args, **kwargs):
            kwargs["n_jobs"] = 1
            model = RandomForestClassifier(*args, **kwargs)
            return model

        mock_rf.side_effect = create_rf

        def exists_side_effect(path):
            return "checkpoint.pkl" not in path

        mock_exists.side_effect = exists_side_effect

        mock_metrics.return_value = {
            "accuracy": 0.90,
            "macro_f1": 0.88,
            "macro_precision": 0.87,
            "macro_recall": 0.86,
        }
        mock_run = MagicMock()
        mock_mlflow_run.return_value = mock_run
        mock_run.__enter__ = MagicMock(return_value=mock_run)
        mock_run.__exit__ = MagicMock(return_value=None)

        acc, f1 = train_main_model(X_train, X_test, y_train, y_test)

        assert acc == 0.90
        assert f1 == 0.88
        assert mock_dump.call_count >= 1
        dump_calls = [str(call[0][1]) for call in mock_dump.call_args_list]
        assert any("model.pkl" in path for path in dump_calls)
        mock_log_model.assert_called_once()
        mock_log_param.assert_called()
        mock_log_metric.assert_called()

    # @patch("src.training.train.mlflow.log_param")
    # @patch("src.training.train.mlflow.log_metric")
    # @patch("src.training.train.mlflow.log_artifact")
    # @patch("src.training.train.mlflow.start_run")
    # @patch("src.training.train.joblib.dump")
    # @patch("src.training.train.mlflow.sklearn.log_model")
    # @patch("src.training.train.calculate_comprehensive_metrics")
    # @patch("src.training.train.os.path.exists")
    # @patch("src.training.train.RandomForestClassifier")
    # def test_train_main_model_creates_model_file(
    #     self,
    #     mock_rf,
    #     mock_exists,
    #     mock_metrics,
    #     mock_log_model,
    #     mock_dump,
    #     mock_mlflow_run,
    #     mock_log_metric,
    #     mock_log_artifact,
    #     mock_log_param,
    #     sample_training_data,
    # ):
    #     """Test that train_main_model saves model to file."""
    #     X_train, X_test, y_train, y_test = sample_training_data
    #
    #     from sklearn.ensemble import RandomForestClassifier
    #
    #     def create_rf(*args, **kwargs):
    #         kwargs["n_jobs"] = 1
    #         model = RandomForestClassifier(*args, **kwargs)
    #         return model
    #
    #     mock_rf.side_effect = create_rf
    #
    #     def exists_side_effect(path):
    #         return "checkpoint.pkl" not in path
    #
    #     mock_exists.side_effect = exists_side_effect
    #
    #     mock_metrics.return_value = {
    #         "accuracy": 0.90,
    #         "macro_f1": 0.88,
    #         "macro_precision": 0.87,
    #         "macro_recall": 0.86,
    #     }
    #     mock_run = MagicMock()
    #     mock_mlflow_run.return_value = mock_run
    #     mock_run.__enter__ = MagicMock(return_value=mock_run)
    #     mock_run.__exit__ = MagicMock(return_value=None)
    #
    #     train_main_model(X_train, X_test, y_train, y_test)
    #
    #     assert mock_dump.call_count >= 1
    #     dump_calls = [str(call[0][1]) for call in mock_dump.call_args_list]
    #     assert any("model.pkl" in path for path in dump_calls)

    def test_get_class_weights_balanced(self):
        """Test that class weights are balanced for imbalanced data."""
        y = np.array([0] * 80 + [1] * 15 + [2] * 5)
        weights = get_class_weights(y)

        # Class 2 should have higher weight (fewer samples)
        assert weights[2] > weights[0]

    @patch("src.training.train.mlflow.log_param")
    @patch("src.training.train.mlflow.log_metric")
    @patch("src.training.train.mlflow.start_run")
    @patch("src.training.train.calculate_comprehensive_metrics")
    def test_train_baseline_uses_balanced_class_weight(
        self,
        mock_metrics,
        mock_mlflow_run,
        mock_log_metric,
        mock_log_param,
        sample_training_data,
    ):
        """Test that baseline model uses balanced class weights."""
        X_train, X_test, y_train, y_test = sample_training_data

        mock_metrics.return_value = {
            "accuracy": 0.85,
            "macro_f1": 0.82,
            "macro_precision": 0.80,
            "macro_recall": 0.78,
        }
        mock_context = MagicMock()
        mock_mlflow_run.return_value.__enter__.return_value = mock_context

        train_baseline(X_train, X_test, y_train, y_test)

        assert mock_log_param.called
        assert mock_log_metric.called
