import mlflow
import mlflow.sklearn

import os

MODEL_NAME = "HousingPriceClassifier"


def load_model(alias="production"):
    if os.getenv("CI") == "true":
        raise RuntimeError("Model loading disabled in CI")

    model_uri = f"models:/{MODEL_NAME}@{alias}"

    model = mlflow.sklearn.load_model(model_uri)

    return model
