import joblib
import mlflow
import mlflow.sklearn
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import os

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

from catboost import CatBoostClassifier


PROCESSED_DIR = "./data/processed"
MODEL_DIR = "models"
MODEL_NAME = "HousingPriceClassifier"

TRAIN_PATH = f"{PROCESSED_DIR}/train.pkl"
TEST_PATH = f"{PROCESSED_DIR}/test.pkl"


RANDOM_STATE = 42

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("housing-price-classification")

def load_data():
    X_train, y_train = joblib.load(TRAIN_PATH)
    X_test, y_test = joblib.load(TEST_PATH)
    return X_train, X_test, y_train, y_test


def get_class_weights(y):
    classes = np.unique(y)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y
    )

    return dict(zip(classes, weights))


def train_baseline(X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name="baseline_logistic_regression"):
        model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            n_jobs=-1
        )

        sample_weights = compute_sample_weight(
            class_weight="balanced",
            y=y_train
)


        model.fit(X_train, y_train, sample_weight=sample_weights)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")

        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("macro_f1", f1)

        print(f"[Baseline] Accuracy: {acc:.4f} | Macro-F1: {f1:.4f}")

    return acc, f1


def train_main_model(X_train, X_test, y_train, y_test):
    class_weights = get_class_weights(y_train)

    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train
    )

    with mlflow.start_run(run_name="main_model_random_forest"):
        # model = XGBClassifier(
        #     n_estimators=300,
        #     max_depth=6,
        #     learning_rate=0.05,
        #     subsample=0.8,
        #     colsample_bytree=0.8,
        #     objective="multi:softprob",
        #     num_class=3,
        #     eval_metric="mlogloss",
        #     random_state=42,
        #     n_jobs=-1
        # )

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            class_weight=class_weights,
            n_jobs=-1
        )

        # model.fit(X_train, y_train, sample_weight=sample_weights, eval_set=[(X_test, y_test)], verbose=False)
        model.fit(X_train, y_train, sample_weight=sample_weights)
        # model._estimator_type = "classifier"
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")

        mlflow.log_param("model", "RandomForestClassifier")
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("checkpointing", "MLflow Registry")
        mlflow.log_param("fallback_enabled", True)
        mlflow.log_param("iterations", 300)
        mlflow.log_param("depth", 6)
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("macro_f1", f1)

        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = f"{MODEL_DIR}/model.pkl"
        joblib.dump(model, model_path)

        mlflow.sklearn.log_model(
            model,
            artifact_path="model_sklearn",
            registered_model_name=MODEL_NAME
        )

        # mlflow.xgboost.log_model(
        #     model,
        #     name="model",   
        #     registered_model_name=MODEL_NAME
        # )

        print(f"[RandomForest] Accuracy: {acc:.4f} | Macro-F1: {f1:.4f}")
        print(f"Model saved to {model_path}")

    return acc, f1


if __name__ == "__main__":
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data()

    # print("Training baseline model...")
    # train_baseline(X_train, X_test, y_train, y_test)

    print("Training main model...")
    train_main_model(X_train, X_test, y_train, y_test)

    print("Training complete.")
