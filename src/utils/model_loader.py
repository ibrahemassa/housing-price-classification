import mlflow
import mlflow.sklearn

MODEL_NAME = "HousingPriceClassifier"


def load_model(alias="production"):
    model_uri = f"models:/{MODEL_NAME}@{alias}"

    model = mlflow.sklearn.load_model(model_uri)

    return model
