import mlflow
import mlflow.sklearn

MODEL_NAME = "HousingPriceClassifier"


def load_model(alias="production"):
    model_uri = f"models:/{MODEL_NAME}@{alias}"

    try:
        model = mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        raise RuntimeError("Model not found") from e

    return model
