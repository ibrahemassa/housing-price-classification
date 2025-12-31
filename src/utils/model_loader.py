import os
from glob import glob

import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

MODEL_NAME = "HousingPriceClassifier"
LOCAL_MODEL_PATH = "models/model.pkl"
FALLBACK_MODEL_PATH = "fallback/model.pkl"
MLRUNS_DIR = "mlruns"

mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(mlflow_uri)


def _find_model_in_mlruns(model_id: str) -> str | None:
    """
    Find the model.pkl file in the local mlruns directory using the model ID.
    This handles cases where MLflow stored absolute paths from a different machine.
    """
    # Search for model directory matching the model ID
    pattern = os.path.join(MLRUNS_DIR, "**", f"*{model_id}*", "**", "model.pkl")
    matches = glob(pattern, recursive=True)
    if matches:
        return matches[0]

    # Also check for artifacts/model.pkl pattern
    pattern = os.path.join(MLRUNS_DIR, "**", f"*{model_id}*", "artifacts", "model.pkl")
    matches = glob(pattern, recursive=True)
    if matches:
        return matches[0]

    return None


def _get_model_id_from_alias(alias: str) -> str | None:
    """Get the model ID from the MLflow registry using the alias."""
    try:
        client = MlflowClient()
        model_version = client.get_model_version_by_alias(MODEL_NAME, alias)
        # The source path contains the model ID
        source = model_version.source
        # Extract model ID from source path (format: .../models/m-<id>/...)
        if "/models/m-" in source:
            model_id = source.split("/models/m-")[1].split("/")[0]
            return f"m-{model_id}"
        # Also try extracting from run_id
        return model_version.run_id
    except Exception as e:
        print(f"Could not get model ID from alias '{alias}': {e}")
        return None


def load_model(alias="production"):
    """
    Load model from MLflow registry.
    Falls back to:
    1. Local mlruns directory (handles path issues from different machines)
    2. Local model file if MLflow loading fails (e.g., path issues in Docker).
    """
    model_uri = f"models:/{MODEL_NAME}@{alias}"
    errors = []

    # Try 1: Load directly from MLflow registry
    try:
        model = mlflow.sklearn.load_model(model_uri)
        print(f"Successfully loaded model from MLflow registry: {model_uri}")
        return model
    except Exception as e:
        error_msg = str(e)
        error_type = type(e).__name__
        errors.append(f"MLflow registry ({error_type}): {error_msg[:200]}")
        print(f"MLflow model loading failed ({error_type}): {error_msg[:200]}")

    # Try 2: Find model in local mlruns directory using model ID
    print("Attempting to load from local mlruns directory...")
    model_id = _get_model_id_from_alias(alias)
    if model_id:
        model_path = _find_model_in_mlruns(model_id)
        if model_path:
            try:
                model = joblib.load(model_path)
                print(f"Successfully loaded model from local mlruns: {model_path}")
                return model
            except Exception as e:
                errors.append(f"Local mlruns ({model_path}): {str(e)}")
        else:
            errors.append(
                f"Model artifacts not found in mlruns for model ID: {model_id}. "
                "The model.pkl file may not have been committed to the repository."
            )

    # Try 3: Fall back to baked-in fallback model (Docker image) - preferred for consistency
    print(f"Trying fallback model file: {FALLBACK_MODEL_PATH}")
    if os.path.exists(FALLBACK_MODEL_PATH):
        try:
            model = joblib.load(FALLBACK_MODEL_PATH)
            print(f"Successfully loaded model from {FALLBACK_MODEL_PATH}")
            return model
        except Exception as e:
            errors.append(f"Fallback model file: {str(e)}")
    else:
        errors.append(f"Fallback model file not found: {FALLBACK_MODEL_PATH}")

    # Try 4: Fall back to volume-mounted local model file
    print(f"Trying local model file: {LOCAL_MODEL_PATH}")
    if os.path.exists(LOCAL_MODEL_PATH):
        try:
            model = joblib.load(LOCAL_MODEL_PATH)
            print(f"Successfully loaded model from {LOCAL_MODEL_PATH}")
            return model
        except Exception as e:
            errors.append(f"Local model file: {str(e)}")
    else:
        errors.append(f"Local model file not found: {LOCAL_MODEL_PATH}")

    # All attempts failed - provide helpful error message
    error_details = "\n  - ".join(errors)
    raise RuntimeError(
        f"Failed to load model '{MODEL_NAME}' with alias '{alias}'.\n"
        f"Attempted sources:\n  - {error_details}\n\n"
        f"To fix this, please run the training script to create the model:\n"
        f"  python -m src.training.train\n"
        f"Then register the model:\n"
        f"  python -m src.training.register_model"
    )
