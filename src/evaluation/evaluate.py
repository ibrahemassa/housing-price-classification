import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)

from src.utils.metrics import calculate_comprehensive_metrics

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

    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    metrics = calculate_comprehensive_metrics(y_test, preds, y_proba, labels=LABELS)

    acc = metrics["accuracy"]
    f1 = metrics["macro_f1"]

    print("Evaluation Results")
    print("=" * 50)
    print(f"Accuracy          : {acc:.4f}")
    print(f"Macro Precision   : {metrics['macro_precision']:.4f}")
    print(f"Macro Recall      : {metrics['macro_recall']:.4f}")
    print(f"Macro F1          : {f1:.4f}")
    print(f"Weighted Precision: {metrics['weighted_precision']:.4f}")
    print(f"Weighted Recall   : {metrics['weighted_recall']:.4f}")
    print(f"Weighted F1       : {metrics['weighted_f1']:.4f}")

    if "roc_auc_macro" in metrics:
        print(f"ROC-AUC (macro)   : {metrics['roc_auc_macro']:.4f}")
    if "roc_auc_weighted" in metrics:
        print(f"ROC-AUC (weighted): {metrics['roc_auc_weighted']:.4f}")

    print("\nPer-Class Metrics:")
    print("-" * 50)
    for label in LABELS:
        if f"precision_{label}" in metrics:
            print(
                f"{label:10s} - Precision: {metrics[f'precision_{label}']:.4f}, "
                f"Recall: {metrics[f'recall_{label}']:.4f}, "
                f"F1: {metrics[f'f1_{label}']:.4f}"
            )

    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=LABELS))

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
