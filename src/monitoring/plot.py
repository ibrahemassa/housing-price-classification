from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PLOT_DIR = Path("data/monitoring/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def plot_categorical_distribution(ref, prod, column):
    ref_counts = ref[column].value_counts(normalize=True).head(10)
    prod_counts = prod[column].value_counts(normalize=True).head(10)

    df = pd.concat(
        [ref_counts, prod_counts], axis=1, keys=["reference", "production"]
    ).fillna(0)

    ax = df.plot(kind="bar", figsize=(10, 5))
    ax.set_title(f"Distribution Drift — {column}")
    ax.set_ylabel("Proportion")
    plt.tight_layout()

    path = PLOT_DIR / f"{column}_distribution_{timestamp()}.png"
    plt.savefig(path)
    plt.close()

    return path


def plot_prediction_distribution(ref_preds, prod_preds):
    ref_counts = ref_preds.value_counts(normalize=True).sort_index()
    prod_counts = prod_preds.value_counts(normalize=True).sort_index()

    df = pd.concat(
        [ref_counts, prod_counts], axis=1, keys=["reference", "production"]
    ).fillna(0)

    ax = df.plot(kind="bar", figsize=(6, 4))
    ax.set_title("Prediction Distribution Drift")
    ax.set_ylabel("Proportion")
    plt.tight_layout()

    path = PLOT_DIR / f"prediction_drift_{timestamp()}.png"
    plt.savefig(path)
    plt.close()

    return path


def plot_cardinality_growth(ref, prod, column):
    ref_card = ref[column].nunique()
    prod_card = prod[column].nunique()

    df = pd.DataFrame(
        {
            "dataset": ["reference", "production"],
            "unique_values": [ref_card, prod_card],
        }
    )

    ax = df.set_index("dataset").plot(kind="bar", legend=False, figsize=(5, 4))
    ax.set_title(f"Cardinality Growth — {column}")
    ax.set_ylabel("Unique Values")
    plt.tight_layout()

    path = PLOT_DIR / f"{column}_cardinality_{timestamp()}.png"
    plt.savefig(path)
    plt.close()

    return path
