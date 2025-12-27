from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.evaluation.evaluate import (
    evaluate_model,
    load_artifacts,
)


class TestEvaluation:
    """Tests for evaluate.py"""

    @patch("src.evaluation.evaluate.joblib.load")
    def test_load_artifacts(self, mock_load):
        """Test loading test data and model."""
        X_test = np.random.rand(10, 5)
        y_test = np.random.randint(0, 3, 10)
        model = RandomForestClassifier()
        model.fit(X_test, y_test)

        mock_load.side_effect = [(X_test, y_test), model]

        X, y, m = load_artifacts()

        assert mock_load.call_count == 2
        assert np.array_equal(X, X_test)
        assert np.array_equal(y, y_test)
        assert m == model

    @patch("src.evaluation.evaluate.joblib.load")
    @patch("src.evaluation.evaluate.accuracy_score")
    @patch("src.evaluation.evaluate.f1_score")
    @patch("src.evaluation.evaluate.classification_report")
    @patch("src.evaluation.evaluate.confusion_matrix")
    @patch("src.evaluation.evaluate.ConfusionMatrixDisplay")
    @patch("src.evaluation.evaluate.plt")
    @patch("src.evaluation.evaluate.os.makedirs")
    def test_evaluate_model(
        self,
        mock_makedirs,
        mock_plt,
        mock_cm_display,
        mock_cm,
        mock_report,
        mock_f1,
        mock_acc,
        mock_load,
    ):
        """Test evaluate_model function."""
        X_test = np.random.rand(10, 5)
        y_test = np.random.randint(0, 3, 10)
        model = RandomForestClassifier()
        model.fit(X_test, y_test)

        mock_load.side_effect = [(X_test, y_test), model]

        mock_acc.return_value = 0.85
        mock_f1.return_value = 0.82
        mock_cm.return_value = np.array([[3, 1, 0], [0, 2, 1], [0, 0, 3]])
        mock_display = MagicMock()
        mock_cm_display.return_value = mock_display
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        evaluate_model()

        # Verify metrics were called
        mock_acc.assert_called_once()
        mock_f1.assert_called_once()
        mock_report.assert_called_once()
        mock_cm.assert_called_once()
        mock_makedirs.assert_called_once()

    @patch("src.evaluation.evaluate.joblib.load")
    @patch("src.evaluation.evaluate.accuracy_score")
    @patch("src.evaluation.evaluate.f1_score")
    @patch("src.evaluation.evaluate.classification_report")
    @patch("src.evaluation.evaluate.confusion_matrix")
    @patch("src.evaluation.evaluate.ConfusionMatrixDisplay")
    @patch("src.evaluation.evaluate.plt")
    @patch("src.evaluation.evaluate.os.makedirs")
    def test_evaluate_model_saves_confusion_matrix(
        self,
        mock_makedirs,
        mock_plt,
        mock_cm_display,
        mock_cm,
        mock_report,
        mock_f1,
        mock_acc,
        mock_load,
    ):
        """Test that evaluate_model saves confusion matrix."""
        X_test = np.random.rand(10, 5)
        y_test = np.random.randint(0, 3, 10)
        model = RandomForestClassifier()
        model.fit(X_test, y_test)

        mock_load.side_effect = [(X_test, y_test), model]

        mock_acc.return_value = 0.85
        mock_f1.return_value = 0.82
        mock_cm.return_value = np.array([[3, 1, 0], [0, 2, 1], [0, 0, 3]])
        mock_display = MagicMock()
        mock_cm_display.return_value = mock_display
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        evaluate_model()

        # Verify plot was saved
        mock_plt.savefig.assert_called_once()
        mock_plt.close.assert_called_once()

    @patch("src.evaluation.evaluate.joblib.load")
    def test_load_artifacts_handles_file_not_found(self, mock_load):
        """Test that load_artifacts handles file not found."""
        mock_load.side_effect = FileNotFoundError("File not found")

        with pytest.raises(FileNotFoundError):
            load_artifacts()
