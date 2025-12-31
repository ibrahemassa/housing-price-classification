import logging
import os
import sys
from collections import Counter
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import mlflow
import numpy as np
import pandas as pd
from prefect import flow, task
from prefect.artifacts import create_markdown_artifact

from src.monitoring.plot import (
    plot_cardinality_growth,
    plot_categorical_distribution,
    plot_prediction_distribution,
)
from src.monitoring.continuous_evaluation import run_continuous_evaluation
from src.monitoring.feature_validation import validate_features
from src.utils.pipelines import training_pipeline

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
    """
    Calculate Population Stability Index (PSI) for categorical features.
    
    PSI = Σ((Actual% - Expected%) * ln(Actual% / Expected%))
    
    Interpretation:
    - PSI < 0.1: No significant population change
    - 0.1 <= PSI < 0.25: Moderate population change
    - PSI >= 0.25: Significant population change (drift detected)
    """
    if len(ref_series) == 0 or len(prod_series) == 0:
        return 0.0
    
    ref = Counter(ref_series)
    prod = Counter(prod_series)
    keys = set(ref) | set(prod)

    ref_counts = np.array([ref.get(k, 0) for k in keys], dtype=float)
    prod_counts = np.array([prod.get(k, 0) for k in keys], dtype=float)
    
    ref_total = ref_counts.sum()
    prod_total = prod_counts.sum()
    
    if ref_total == 0 or prod_total == 0:
        return 0.0
    
    ref_props = ref_counts / ref_total
    prod_props = prod_counts / prod_total
    
    eps = 1e-6  
    psi = 0.0
    
    for i in range(len(keys)):
        actual_pct = prod_props[i]
        expected_pct = ref_props[i]
        
        if actual_pct == 0 and expected_pct == 0:
            continue
        
        if expected_pct > eps:
            psi += (actual_pct - expected_pct) * np.log((actual_pct + eps) / (expected_pct + eps))
        elif actual_pct > eps:
            psi += actual_pct * np.log((actual_pct + eps) / eps)
    
    return max(0.0, psi) / 20  


def numerical_psi(ref_series, prod_series, bins=10):
    """
    Calculate Population Stability Index (PSI) for numerical features.
    
    Args:
        ref_series: Reference data series
        prod_series: Production data series
        bins: Number of bins to use for discretization
    
    Returns:
        PSI value
    """
    if len(ref_series) == 0 or len(prod_series) == 0:
        return 0.0
    
    ref_clean = ref_series.dropna()
    prod_clean = prod_series.dropna()
    
    if len(ref_clean) == 0 or len(prod_clean) == 0:
        return 0.0
    
    try:
        _, bin_edges = pd.cut(ref_clean, bins=bins, retbins=True, duplicates="drop")
        
        if len(bin_edges) < 2:
            return 0.0
        
        ref_binned = pd.cut(ref_clean, bins=bin_edges, include_lowest=True)
        prod_binned = pd.cut(prod_clean, bins=bin_edges, include_lowest=True)
        
        ref_counts = ref_binned.value_counts().sort_index()
        prod_counts = prod_binned.value_counts().sort_index()
        
        all_bins = ref_counts.index.union(prod_counts.index)
        ref_counts = ref_counts.reindex(all_bins, fill_value=0)
        prod_counts = prod_counts.reindex(all_bins, fill_value=0)
        
        ref_total = ref_counts.sum()
        prod_total = prod_counts.sum()
        
        if ref_total == 0 or prod_total == 0:
            return 0.0
        
        ref_props = ref_counts.values / ref_total
        prod_props = prod_counts.values / prod_total
        
        eps = 1e-6
        psi = 0.0
        
        for i in range(len(all_bins)):
            actual_pct = prod_props[i]
            expected_pct = ref_props[i]
            
            if actual_pct == 0 and expected_pct == 0:
                continue
            
            if expected_pct > eps:
                psi += (actual_pct - expected_pct) * np.log((actual_pct + eps) / (expected_pct + eps))
            elif actual_pct > eps:
                psi += actual_pct * np.log((actual_pct + eps) / eps)
        
        return max(0.0, psi)
    except Exception as e:
        logging.warning(f"Error calculating numerical PSI: {e}")
        return 0.0


def prediction_drift(ref_preds, prod_preds):
    """
    Calculate KL divergence (Kullback-Leibler divergence) between prediction distributions.
    
    KL(P||Q) = Σ P(i) * log(P(i) / Q(i))
    Where P = production distribution, Q = reference distribution
    
    Interpretation:
    - KL = 0: Identical distributions
    - KL > 0: Distributions differ (higher = more different)
    """
    if len(ref_preds) == 0 or len(prod_preds) == 0:
        return 0.0
    
    ref_counts = ref_preds.value_counts().sort_index()
    prod_counts = prod_preds.value_counts().sort_index()
    
    all_classes = ref_counts.index.union(prod_counts.index)
    ref_counts = ref_counts.reindex(all_classes, fill_value=0)
    prod_counts = prod_counts.reindex(all_classes, fill_value=0)
    
    ref_total = ref_counts.sum()
    prod_total = prod_counts.sum()
    
    if ref_total == 0 or prod_total == 0:
        return 0.0
    
    ref_probs = ref_counts.values / ref_total
    prod_probs = prod_counts.values / prod_total
    
    eps = 1e-6  
    kl_div = 0.0
    
    for i in range(len(all_classes)):
        p = prod_probs[i]
        q = ref_probs[i]
        
        if p < eps:
            continue
        
        if q < eps:
            kl_div += p * np.log(p / eps)
        else:
            kl_div += p * np.log((p + eps) / (q + eps))
    
    return max(0.0, kl_div)


@task
def run_monitor():
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("monitoring")

    with mlflow.start_run(run_name="drift-check"):
        if not PROD_INPUTS.exists() or not PROD_PREDS.exists():
            logging.info("No production data yet — skipping monitoring")
            # return

        ref = pd.read_parquet(REFERENCE_PATH)
        prod_inputs = pd.read_parquet(PROD_INPUTS)
        prod_preds = pd.read_parquet(PROD_PREDS)

        alerts = 0
        drift_metrics = {}

        if CAT_FEATURE in ref.columns and CAT_FEATURE in prod_inputs.columns:
            d_psi = categorical_psi(ref[CAT_FEATURE], prod_inputs[CAT_FEATURE])
            drift_metrics["district_psi"] = d_psi
            logging.info(f"District PSI: {d_psi:.4f}")
            if d_psi > PSI_THRESHOLD:
                alerts += 1
                logging.warning(f"District PSI ({d_psi:.4f}) exceeds threshold ({PSI_THRESHOLD})")
        else:
            logging.warning(f"Column '{CAT_FEATURE}' not found in reference or production data")
            d_psi = 0.0
            drift_metrics["district_psi"] = d_psi

        if HIGH_CARD_FEATURE in ref.columns and HIGH_CARD_FEATURE in prod_inputs.columns:
            ref_card = ref[HIGH_CARD_FEATURE].nunique()
            prod_card = prod_inputs[HIGH_CARD_FEATURE].nunique()
            card_ratio = prod_card / max(ref_card, 1)
            drift_metrics["address_cardinality_ratio"] = card_ratio
            logging.info(f"Address cardinality - Reference: {ref_card}, Production: {prod_card}, Ratio: {card_ratio:.2f}")
            if card_ratio > CARDINALITY_THRESHOLD:
                alerts += 1
                logging.warning(f"Address cardinality ratio ({card_ratio:.2f}) exceeds threshold ({CARDINALITY_THRESHOLD})")
        else:
            logging.warning(f"Column '{HIGH_CARD_FEATURE}' not found in reference or production data")
            card_ratio = 1.0
            drift_metrics["address_cardinality_ratio"] = card_ratio

        ref_pred_col = None
        if "price_category" in ref.columns:
            ref_pred_col = ref["price_category"]
        elif "target" in ref.columns:
            ref_pred_col = ref["target"]
        else:
            logging.warning("No 'price_category' or 'target' column found in reference data")
        
        if ref_pred_col is not None and "prediction" in prod_preds.columns:
            if ref_pred_col.dtype == 'object':
                category_map = {"low": 0, "medium": 1, "high": 2}
                ref_pred_col = ref_pred_col.map(category_map).fillna(-1)
            
            pred_drift = prediction_drift(ref_pred_col, prod_preds["prediction"])
            drift_metrics["prediction_kl_drift"] = pred_drift
            logging.info(f"Prediction KL divergence: {pred_drift:.4f}")
            if pred_drift > PRED_DRIFT_THRESHOLD:
                alerts += 1
                logging.warning(f"Prediction KL divergence ({pred_drift:.4f}) exceeds threshold ({PRED_DRIFT_THRESHOLD})")
        else:
            logging.warning("Cannot calculate prediction drift - missing reference or production predictions")
            pred_drift = 0.0
            drift_metrics["prediction_kl_drift"] = pred_drift

        for metric_name, metric_value in drift_metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        mlflow.log_metric("total_alerts", alerts)
        mlflow.log_metric("alerts", alerts)  
        
        if alerts == 0:
            mlflow.log_metric("drift_severity", 0)  
            mlflow.log_param("drift_status", "stable")
        elif alerts == 1:
            mlflow.log_metric("drift_severity", 1) 
            mlflow.log_param("drift_status", "low")
        elif alerts == 2:
            mlflow.log_metric("drift_severity", 2)
            mlflow.log_param("drift_status", "medium")
        else:
            mlflow.log_metric("drift_severity", 3)
            mlflow.log_param("drift_status", "high")
        
        try:
            validation_results = validate_features(ref, prod_inputs)
            
            mlflow.log_metric("feature_validation_passed", 1 if validation_results["passed"] else 0)
            mlflow.log_metric("feature_validation_alerts", len(validation_results["alerts"]))
            mlflow.log_metric("feature_validation_warnings", len(validation_results["warnings"]))
            
            if not validation_results["passed"]:
                alerts += len(validation_results["alerts"])
            
            validation_summary = f"""
## Feature Validation Summary

**Status:** {"✅ Passed" if validation_results["passed"] else "❌ Failed"}

**Alerts:** {len(validation_results["alerts"])}
**Warnings:** {len(validation_results["warnings"])}

### Alerts:
"""
            for alert in validation_results["alerts"][:5]:  
                validation_summary += f"- **{alert['column']}** ({alert['type']}): {alert.get('message', 'N/A')}\n"
            
            if len(validation_results["alerts"]) > 5:
                validation_summary += f"\n... and {len(validation_results['alerts']) - 5} more alerts\n"
            
            create_markdown_artifact(
                key="feature-validation-summary",
                markdown=validation_summary,
            )
        except Exception as e:
            logging.warning(f"Feature validation failed: {e}")
        
        # Continuous Model Evaluation
        try:
            cme_results = run_continuous_evaluation()
            if cme_results:
                logging.info("Continuous evaluation completed successfully")
        except Exception as e:
            logging.warning(f"Continuous evaluation failed: {e}")

        create_markdown_artifact(
            key="drift-summary",
            markdown=f"""
## Drift Summary

        | Metric | Value | Threshold | Status |
        |------|------|-----------|--------|
        | District PSI | {d_psi:.3f} | {PSI_THRESHOLD} | {"❌" if d_psi > PSI_THRESHOLD else "✅"} |
        | Cardinality | {card_ratio:.2f} | {CARDINALITY_THRESHOLD} | {"❌" if card_ratio > CARDINALITY_THRESHOLD else "✅"} |
        | Prediction KL | {pred_drift:.3f} | {PRED_DRIFT_THRESHOLD} | {"❌" if pred_drift > PRED_DRIFT_THRESHOLD else "✅"} |
""",
        )
        try:
            district_plot = plot_categorical_distribution(ref, prod_inputs, "district")

            cardinality_plot = plot_cardinality_growth(ref, prod_inputs, "address")

            ref_pred_for_plot = ref_pred_col if ref_pred_col is not None else ref.get("target", pd.Series())
            if len(ref_pred_for_plot) > 0:
                prediction_plot = plot_prediction_distribution(
                    ref_pred_for_plot, prod_preds["prediction"]
                )
            else:
                prediction_plot = None
                logging.warning("Cannot create prediction plot - missing reference predictions")

            plot_markdown = """
### Drift Monitoring Plots

"""
            plot_markdown += f"- District distribution: `{district_plot}`\n"
            plot_markdown += f"- Address cardinality: `{cardinality_plot}`\n"
            if prediction_plot:
                plot_markdown += f"- Prediction drift: `{prediction_plot}`\n"
            
            create_markdown_artifact(
                key="drift-plots",
                markdown=plot_markdown,
            )

            # mlflow.log_artifact(district_plot)
            # mlflow.log_artifact(cardinality_plot)
            # mlflow.log_artifact(prediction_plot)

        except Exception as e:
            logging.warning(f"Plotting failed: {e}")

        return alerts


@flow(name="housing-monitoring-flow")
def monitoring_flow():
    alerts = run_monitor()

    if alerts >= 3:
        logging.error("Drift detected — triggering retraining pipeline")
        # training_pipeline()

    else:
        logging.info("Model stable")


if __name__ == "__main__":
    monitoring_flow()
