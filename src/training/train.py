import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import (
    compute_class_weight,
    compute_sample_weight,
)

from src.utils.metrics import calculate_comprehensive_metrics

PROCESSED_DIR = "./data/processed"
MODEL_DIR = "models"
CHECKPOINT_DIR = f"{MODEL_DIR}/checkpoints"
MODEL_NAME = "HousingPriceClassifier"

TRAIN_PATH = f"{PROCESSED_DIR}/train.pkl"
TEST_PATH = f"{PROCESSED_DIR}/test.pkl"

RANDOM_STATE = 42
CHECKPOINT_INTERVAL = 20
TOTAL_ESTIMATORS = 100

mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_experiment("housing-price-classification")


def load_data():
    X_train, y_train = joblib.load(TRAIN_PATH)
    X_test, y_test = joblib.load(TEST_PATH)
    return X_train, X_test, y_train, y_test


def get_class_weights(y):
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)

    return dict(zip(classes, weights, strict=False))


def load_checkpoint(expected_n_features=None):
    """Load checkpoint if it exists and is compatible with current data.

    Args:
        expected_n_features: Number of features in current training data.
                           If provided, validates checkpoint compatibility.
    """
    checkpoint_path = f"{CHECKPOINT_DIR}/checkpoint.pkl"
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = joblib.load(checkpoint_path)
            if isinstance(checkpoint, dict):
                model = checkpoint.get("model")
                n_estimators = checkpoint.get("n_estimators", 0)

                if model is not None and expected_n_features is not None:
                    model_n_features = getattr(model, "n_features_in_", None)
                    if (
                        model_n_features is not None
                        and model_n_features != expected_n_features
                    ):
                        print(
                            f"⚠️ Checkpoint incompatible: model expects {model_n_features} features, "
                            f"but data has {expected_n_features} features. Training from scratch."
                        )
                        return None, 0

                return model, n_estimators
        except Exception:
            pass
    return None, 0


def save_checkpoint(model, n_estimators):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = f"{CHECKPOINT_DIR}/checkpoint.pkl"
    joblib.dump({"model": model, "n_estimators": n_estimators}, checkpoint_path)
    mlflow.log_artifact(checkpoint_path, "checkpoints")


def train_baseline(X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name="baseline_logistic_regression"):
        model = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)

        sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

        model.fit(X_train, y_train, sample_weight=sample_weights)
        preds = model.predict(X_test)

        y_proba = (
            model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        )

        metrics = calculate_comprehensive_metrics(
            y_test, preds, y_proba, labels=["low", "medium", "high"]
        )

        mlflow.log_param("model", "LogisticRegression")

        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)) and not isinstance(
                metric_value, bool
            ):
                mlflow.log_metric(metric_name, metric_value)

        acc = metrics["accuracy"]
        f1 = metrics["macro_f1"]
        print(
            f"[Baseline] Accuracy: {acc:.4f} | Macro-F1: {f1:.4f} | Precision: {metrics['macro_precision']:.4f} | Recall: {metrics['macro_recall']:.4f}"
        )

    return acc, f1


def train_main_model(X_train, X_test, y_train, y_test, resume_from_checkpoint=True):
    class_weights = get_class_weights(y_train)
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    n_features = X_train.shape[1]

    with mlflow.start_run(run_name="main_model_random_forest"):
        model, start_estimators = (
            load_checkpoint(expected_n_features=n_features)
            if resume_from_checkpoint
            else (None, 0)
        )

        if model is None:
            model = RandomForestClassifier(
                n_estimators=CHECKPOINT_INTERVAL,
                random_state=RANDOM_STATE,
                class_weight=class_weights,
                n_jobs=-1,
                warm_start=True,
            )
            model.fit(X_train, y_train, sample_weight=sample_weights)
            save_checkpoint(model, CHECKPOINT_INTERVAL)
            start_estimators = CHECKPOINT_INTERVAL
            print(f"Initial training: {CHECKPOINT_INTERVAL} estimators")
        else:
            print(f"Resuming from checkpoint: {start_estimators} estimators")

        for n_est in range(
            start_estimators + CHECKPOINT_INTERVAL,
            TOTAL_ESTIMATORS + 1,
            CHECKPOINT_INTERVAL,
        ):
            model.n_estimators = n_est
            model.fit(X_train, y_train, sample_weight=sample_weights)
            save_checkpoint(model, n_est)
            print(f"Checkpoint saved: {n_est} estimators")

        if model.n_estimators < TOTAL_ESTIMATORS:
            model.n_estimators = TOTAL_ESTIMATORS
            model.fit(X_train, y_train, sample_weight=sample_weights)
            save_checkpoint(model, TOTAL_ESTIMATORS)
        # model._estimator_type = "classifier"
        preds = model.predict(X_test)

        y_proba = (
            model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        )

        metrics = calculate_comprehensive_metrics(
            y_test, preds, y_proba, labels=["low", "medium", "high"]
        )

        mlflow.log_param("model", "RandomForestClassifier")
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("checkpointing", "enabled")
        mlflow.log_param("checkpoint_interval", CHECKPOINT_INTERVAL)
        mlflow.log_param("total_estimators", TOTAL_ESTIMATORS)
        mlflow.log_param("resumed_from_checkpoint", start_estimators > 0)

        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)) and not isinstance(
                metric_value, bool
            ):
                mlflow.log_metric(metric_name, metric_value)

        acc = metrics["accuracy"]
        f1 = metrics["macro_f1"]

        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = f"{MODEL_DIR}/model.pkl"
        joblib.dump(model, model_path)

        mlflow.sklearn.log_model(
            model, name="model_sklearn", registered_model_name=MODEL_NAME
        )

        # mlflow.xgboost.log_model(
        #     model,
        #     name="model",
        #     registered_model_name=MODEL_NAME
        # )

        print(
            f"[RandomForest] Accuracy: {acc:.4f} | Macro-F1: {f1:.4f} | Precision: {metrics['macro_precision']:.4f} | Recall: {metrics['macro_recall']:.4f}"
        )
        if "roc_auc_macro" in metrics:
            print(f"ROC-AUC (macro): {metrics['roc_auc_macro']:.4f}")
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
#
