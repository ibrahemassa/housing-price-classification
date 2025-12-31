import os

import joblib
import mlflow
import mlflow.sklearn

MODEL_NAME = "HousingPriceClassifier"
LOCAL_MODEL_PATH = "models/model.pkl"

mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(mlflow_uri)


def load_model(alias="production"):
    """
    Load model from MLflow registry.
    Falls back to local model file if MLflow loading fails (e.g., path issues in Docker).
    """
    model_uri = f"models:/{MODEL_NAME}@{alias}"

    try:
        model = mlflow.sklearn.load_model(model_uri)
        return model
    except Exception as e:
        error_msg = str(e)
        error_type = type(e).__name__

        print(f"MLflow model loading failed ({error_type}): {error_msg[:200]}")
        print(f"Falling back to local model file: {LOCAL_MODEL_PATH}")

        try:
            if os.path.exists(LOCAL_MODEL_PATH):
                model = joblib.load(LOCAL_MODEL_PATH)
                print(f"Successfully loaded model from {LOCAL_MODEL_PATH}")
                return model
            else:
                raise RuntimeError(
                    f"Neither MLflow model nor local model file found. "
                    f"MLflow error: {error_msg[:200]}, Local path: {LOCAL_MODEL_PATH}"
                )
        except Exception as local_error:
            raise RuntimeError(
                f"Failed to load model from both MLflow and local file. "
                f"MLflow error: {error_msg[:200]}, Local error: {str(local_error)}"
            ) from e
