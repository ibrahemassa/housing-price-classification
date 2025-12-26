from prefect import flow, task
import subprocess
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter
from scipy.stats import entropy
from src.monitoring.plot import (
    plot_categorical_distribution,
    plot_prediction_distribution,
    plot_cardinality_growth,
)
from prefect.artifacts import create_markdown_artifact
import sys
# print(sys.path)

REFERENCE_PATH = Path("data/reference/reference.parquet")
PROD_INPUTS = Path("data/production/inputs.parquet")
PROD_PREDS = Path("data/production/predictions.parquet")
PIPELINE = "src/utils/pipelines.py"

CAT_FEATURE = "district"
HIGH_CARD_FEATURE = "address"

PSI_THRESHOLD = 0.25
CARDINALITY_THRESHOLD = 1.3
PRED_DRIFT_THRESHOLD = 0.15

logging.basicConfig(level=logging.INFO)

def categorical_psi(ref_series, prod_series):
    ref = Counter(ref_series)
    prod = Counter(prod_series)
    keys = set(ref) | set(prod)

    r = np.array([ref[k] for k in keys], dtype=float)
    p = np.array([prod[k] for k in keys], dtype=float)

    r /= r.sum()
    p /= p.sum()

    eps = 1e-6
    return np.sum((p - r) * np.log((p + eps) / (r + eps)))


def prediction_drift(ref_preds, prod_preds):
    r = ref_preds.value_counts().sort_index()
    p = prod_preds.value_counts().sort_index()
    r, p = r.align(p, fill_value=0)
    return entropy(r.values, p.values)

@task
def run_monitor():
    if not PROD_INPUTS.exists() or not PROD_PREDS.exists():
        logging.info("No production data yet — skipping monitoring")
        # return

    ref = pd.read_parquet(REFERENCE_PATH)
    prod_inputs = pd.read_parquet(PROD_INPUTS)
    prod_preds = pd.read_parquet(PROD_PREDS)

    alerts = 0
    # alerts = 3

    d_psi = categorical_psi(ref[CAT_FEATURE], prod_inputs[CAT_FEATURE])
    logging.info(f"District PSI: {d_psi:.4f}")
    if d_psi > PSI_THRESHOLD:
        alerts += 1

    card_ratio = (
        prod_inputs[HIGH_CARD_FEATURE].nunique()
        / max(ref[HIGH_CARD_FEATURE].nunique(), 1)
    )
    logging.info(f"Address cardinality ratio: {card_ratio:.2f}")
    if card_ratio > CARDINALITY_THRESHOLD:
        alerts += 1

    pred_drift = prediction_drift(ref["price_category"], prod_preds["prediction"])
    logging.info(f"Prediction KL drift: {pred_drift:.4f}")
    if pred_drift > PRED_DRIFT_THRESHOLD:
        alerts += 1

    try:
        district_plot = plot_categorical_distribution(
            ref, prod_inputs, "district"
        )

        cardinality_plot = plot_cardinality_growth(
            ref, prod_inputs, "address"
        )

        prediction_plot = plot_prediction_distribution(
            ref["target"], prod_preds["prediction"]
        )

        create_markdown_artifact(
            key="drift-plots",
            markdown=f"""
### Drift Monitoring Plots

    - District distribution: `{district_plot}`
    - Address cardinality: `{cardinality_plot}`
    - Prediction drift: `{prediction_plot}`
    """
        )

    except Exception as e:
        logging.warning(f"Plotting failed: {e}")

    if alerts >= 2:
        logging.error("Drift detected — triggering retraining pipeline")
        subprocess.run([sys.executable, PIPELINE], check=True)
    else:
        logging.info("Model stable")

@flow(name="housing-monitoring-flow")
def monitoring_flow():
    run_monitor()

if __name__ == "__main__":
    monitoring_flow()

