import joblib
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"
OUTPUT_DIR = "models"

TEST_PATH = f"{PROCESSED_DIR}/test.pkl"
MODEL_PATH = f"{MODEL_DIR}/model.pkl"

LABELS = ["low", "mid", "high"]


def load_artifacts():
    X_test, y_test = joblib.load(TEST_PATH)
    model = joblib.load(MODEL_PATH)
    return X_test, y_test, model


def evaluate_model():
    X_test, y_test, model = load_artifacts()

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")

    print("Evaluation Results")
    print("--------------------")
    print(f"Accuracy   : {acc:.4f}")
    print(f"Macro F1   : {f1:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, preds, target_names=LABELS))

    # Confusion Matrix
    cm = confusion_matrix(y_test, preds, labels=LABELS)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title("Confusion Matrix – Price Category")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cm_path = f"{OUTPUT_DIR}/confusion_matrix.png"
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()

    print(f"Confusion matrix saved to {cm_path}")


if __name__ == "__main__":
    evaluate_model()
