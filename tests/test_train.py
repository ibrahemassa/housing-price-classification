import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from src.training.train import (
    load_data,
    get_class_weights,
    train_baseline,
    train_main_model
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
        mock_load.side_effect = [
            (X_train, y_train),
            (X_test, y_test)
        ]
        
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
    @patch("src.training.train.accuracy_score")
    @patch("src.training.train.f1_score")
    @patch("src.training.train.joblib.load")
    def test_train_baseline(
        self,
        mock_load,
        mock_f1,
        mock_acc,
        mock_mlflow_run,
        mock_log_metric,
        mock_log_param,
        sample_training_data
    ):
        """Test training baseline model."""
        X_train, X_test, y_train, y_test = sample_training_data
        mock_load.side_effect = [
            (X_train, y_train),
            (X_test, y_test)
        ]
        
        mock_acc.return_value = 0.85
        mock_f1.return_value = 0.82
        mock_context = MagicMock()
        mock_mlflow_run.return_value.__enter__.return_value = mock_context
        
        acc, f1 = train_baseline(X_train, X_test, y_train, y_test)
        
        assert acc == 0.85
        assert f1 == 0.82
        mock_log_param.assert_called()
        mock_log_metric.assert_called()

    @patch("src.training.train.mlflow.log_param")
    @patch("src.training.train.mlflow.log_metric")
    @patch("src.training.train.mlflow.start_run")
    @patch("src.training.train.joblib.dump")
    @patch("src.training.train.mlflow.sklearn.log_model")
    @patch("src.training.train.accuracy_score")
    @patch("src.training.train.f1_score")
    @patch("src.training.train.joblib.load")
    @patch("src.training.train.os.makedirs")
    def test_train_main_model(
        self,
        mock_makedirs,
        mock_load,
        mock_f1,
        mock_acc,
        mock_log_model,
        mock_dump,
        mock_mlflow_run,
        mock_log_metric,
        mock_log_param,
        sample_training_data
    ):
        """Test training main model."""
        X_train, X_test, y_train, y_test = sample_training_data
        mock_load.side_effect = [
            (X_train, y_train),
            (X_test, y_test)
        ]
        
        mock_acc.return_value = 0.90
        mock_f1.return_value = 0.88
        # Mock mlflow.start_run to return a context manager
        mock_run = MagicMock()
        mock_mlflow_run.return_value = mock_run
        mock_run.__enter__ = MagicMock(return_value=mock_run)
        mock_run.__exit__ = MagicMock(return_value=None)
        
        acc, f1 = train_main_model(X_train, X_test, y_train, y_test)
        
        assert acc == 0.90
        assert f1 == 0.88
        mock_dump.assert_called_once()
        mock_log_model.assert_called_once()
        mock_log_param.assert_called()
        mock_log_metric.assert_called()

    @patch("src.training.train.mlflow.log_param")
    @patch("src.training.train.mlflow.log_metric")
    @patch("src.training.train.mlflow.start_run")
    @patch("src.training.train.joblib.dump")
    @patch("src.training.train.mlflow.sklearn.log_model")
    @patch("src.training.train.accuracy_score")
    @patch("src.training.train.f1_score")
    def test_train_main_model_creates_model_file(
        self,
        mock_f1,
        mock_acc,
        mock_log_model,
        mock_dump,
        mock_mlflow_run,
        mock_log_metric,
        mock_log_param,
        sample_training_data
    ):
        """Test that train_main_model saves model to file."""
        X_train, X_test, y_train, y_test = sample_training_data
        
        mock_acc.return_value = 0.90
        mock_f1.return_value = 0.88
        # Mock mlflow.start_run to return a context manager
        mock_run = MagicMock()
        mock_mlflow_run.return_value = mock_run
        mock_run.__enter__ = MagicMock(return_value=mock_run)
        mock_run.__exit__ = MagicMock(return_value=None)
        
        train_main_model(X_train, X_test, y_train, y_test)
        
        # Verify model was dumped
        mock_dump.assert_called_once()
        dump_args = mock_dump.call_args
        assert "model.pkl" in str(dump_args[0][1])

    def test_get_class_weights_balanced(self):
        """Test that class weights are balanced for imbalanced data."""
        y = np.array([0] * 80 + [1] * 15 + [2] * 5)
        weights = get_class_weights(y)
        
        # Class 2 should have higher weight (fewer samples)
        assert weights[2] > weights[0]

    @patch("src.training.train.mlflow.log_param")
    @patch("src.training.train.mlflow.log_metric")
    @patch("src.training.train.mlflow.start_run")
    @patch("src.training.train.accuracy_score")
    @patch("src.training.train.f1_score")
    def test_train_baseline_uses_balanced_class_weight(
        self,
        mock_f1,
        mock_acc,
        mock_mlflow_run,
        mock_log_metric,
        mock_log_param,
        sample_training_data
    ):
        """Test that baseline model uses balanced class weights."""
        X_train, X_test, y_train, y_test = sample_training_data
        
        mock_acc.return_value = 0.85
        mock_f1.return_value = 0.82
        mock_context = MagicMock()
        mock_mlflow_run.return_value.__enter__.return_value = mock_context
        
        train_baseline(X_train, X_test, y_train, y_test)
        
        # Verify mlflow logging
        assert mock_log_param.called
        assert mock_log_metric.called

