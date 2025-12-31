# from unittest.mock import MagicMock, patch
#
# import pytest
#
# from src.utils.pipelines import preprocess, promote, train, training_pipeline
#
#
# class TestPipelines:
#     """Tests for pipelines.py"""
#
#     @patch("src.utils.pipelines.subprocess.run")
#     def test_preprocess_task(self, mock_subprocess):
#         """Test preprocess task calls subprocess correctly."""
#         mock_subprocess.return_value = MagicMock(returncode=0)
#
#         preprocess()
#
#         mock_subprocess.assert_called_once()
#         call_args = mock_subprocess.call_args[0][0]
#         assert "python" in call_args
#         assert "preprocess.py" in call_args[1]
#
#     @patch("src.utils.pipelines.subprocess.run")
#     def test_train_task(self, mock_subprocess):
#         """Test train task calls subprocess correctly."""
#         mock_subprocess.return_value = MagicMock(returncode=0)
#
#         train()
#
#         mock_subprocess.assert_called_once()
#         call_args = mock_subprocess.call_args[0][0]
#         assert "python" in call_args
#         assert "train.py" in call_args[1]
#
#     @patch("src.utils.pipelines.subprocess.run")
#     def test_promote_task(self, mock_subprocess):
#         """Test promote task calls subprocess correctly."""
#         mock_subprocess.return_value = MagicMock(returncode=0)
#
#         promote()
#
#         mock_subprocess.assert_called_once()
#         call_args = mock_subprocess.call_args[0][0]
#         assert "python" in call_args
#         assert "register_model.py" in call_args[1]
#
#     @patch("src.utils.pipelines.subprocess.run")
#     def test_training_pipeline_executes_all_tasks(self, mock_subprocess):
#         """Test that training_pipeline executes all tasks in order."""
#         mock_subprocess.return_value = MagicMock(returncode=0)
#
#         training_pipeline()
#
#         assert mock_subprocess.call_count == 3
#
#     @patch("src.utils.pipelines.subprocess.run")
#     def test_preprocess_task_handles_failure(self, mock_subprocess):
#         """Test that preprocess task handles subprocess failure."""
#         mock_subprocess.side_effect = Exception("Subprocess failed")
#
#         with pytest.raises(Exception, match="Subprocess failed"):
#             preprocess()
#
#     @patch("src.utils.pipelines.subprocess.run")
#     def test_training_pipeline_with_check_flag(self, mock_subprocess):
#         """Test that training_pipeline uses check=True flag."""
#         mock_subprocess.return_value = MagicMock(returncode=0)
#
#         training_pipeline()
#
#         # Verify check=True is used
#         for call in mock_subprocess.call_args_list:
#             assert call.kwargs.get("check")

