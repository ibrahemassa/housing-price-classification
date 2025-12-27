from mlflow.tracking import MlflowClient

MODEL_NAME = "HousingPriceClassifier"

client = MlflowClient()

latest_versions = client.search_model_versions(f"name='{MODEL_NAME}'")

latest_version = max(latest_versions, key=lambda v: int(v.version))

version = latest_version.version

client.set_registered_model_alias(name=MODEL_NAME, alias="staging", version=version)

client.set_registered_model_alias(name=MODEL_NAME, alias="production", version=version)

print(f"Model v{version} assigned to @staging and @production")
