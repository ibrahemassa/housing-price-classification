"""
Continuous Model Evaluation (CME) module.
Tracks model performance degradation by comparing production metrics to reference.
"""

import logging
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import joblib
import mlflow
import numpy as np
import pandas as pd

from src.utils.metrics import (
    calculate_class_distribution,
    calculate_comprehensive_metrics,
    calculate_prediction_statistics,
    compare_metrics,
)
from src.utils.model_loader import load_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REFERENCE_PATH = Path("data/reference/reference.parquet")
PROD_INPUTS = Path("data/production/inputs.parquet")
PROD_PREDS = Path("data/production/predictions.parquet")
MODEL_PATH = Path("models/model.pkl")
PREPROCESSOR_PATH = Path("data/processed/preprocessor.pkl")
HASHER_PATH = Path("data/processed/hasher.pkl")

LABELS = ["low", "medium", "high"]
CATEGORIES = {0: "low", 1: "medium", 2: "high"}


def load_reference_metrics():
    """Load reference dataset and calculate baseline metrics."""
    if not REFERENCE_PATH.exists():
        logger.warning("Reference data not found")
        return None
    
    ref_data = pd.read_parquet(REFERENCE_PATH)
    
    try:
        model = load_model("production")
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        hasher = joblib.load(HASHER_PATH)
    except Exception as e:
        logger.error(f"Failed to load model/preprocessors: {e}")
        return None
    
    if "price_category" in ref_data.columns:
        y_true = ref_data["price_category"].map({"low": 0, "medium": 1, "high": 2})
    elif "target" in ref_data.columns:
        y_true = ref_data["target"]
    else:
        logger.error("No target column found in reference data")
        return None
    
    try:
        import joblib
        from scipy.sparse import hstack
        
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        hasher = joblib.load(HASHER_PATH)
        
        HIGH_CARD_CAT = ["address", "AdCreationDate", "Subscription"]
        LOW_CARD_CAT = ["district", "HeatingType", "StructureType", "FloorLocation"]
        NUMERIC_FEATURES = [
            "GrossSquareMeters",
            "NetSquareMeters",
            "BuildingAge",
            "NumberOfRooms",
        ]
        
        predictions = []
        probabilities = []
        
        for _, row in ref_data.iterrows():
            try:
                df = pd.DataFrame([row])
                X_base = preprocessor.transform(df)
                X_high = df[HIGH_CARD_CAT].astype(str).values.tolist()
                X_hash = hasher.transform(X_high)
                X = hstack([X_base, X_hash])
                
                pred = model.predict(X)[0]
                predictions.append(pred)
                
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X)[0]
                    probabilities.append(proba)
            except Exception as e:
                logger.warning(f"Failed to predict for row: {e}")
                continue
        
        y_pred = np.array(predictions)
        
        y_proba = np.array(probabilities) if probabilities else None
        metrics = calculate_comprehensive_metrics(
            y_true.values, y_pred, y_proba, labels=LABELS
        )
        
        class_dist = calculate_class_distribution(y_true.values)
        metrics.update(class_dist)
        
        return metrics
    except Exception as e:
        logger.error(f"Failed to calculate reference metrics: {e}")
        return None


def evaluate_production_performance():
    """
    Evaluate production model performance using recent production data.
    This is a simplified version - in practice, you'd need ground truth labels.
    """
    if not PROD_INPUTS.exists() or not PROD_PREDS.exists():
        logger.warning("Production data not found")
        return None
    
    prod_inputs = pd.read_parquet(PROD_INPUTS)
    prod_preds = pd.read_parquet(PROD_PREDS)
    
    if len(prod_preds) == 0:
        logger.warning("No production predictions available")
        return None
    
    pred_stats = calculate_prediction_statistics(prod_preds["prediction"].values)
    
    class_dist = calculate_class_distribution(prod_preds["prediction"].values)
    
    metrics = {
        "total_predictions": len(prod_preds),
        **pred_stats,
        **class_dist,
    }
    
    return metrics


def run_continuous_evaluation():
    """
    Run continuous model evaluation comparing production to reference.
    """
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("monitoring")
    
    with mlflow.start_run(run_name="continuous-evaluation"):
        ref_metrics = load_reference_metrics()
        
        if ref_metrics is None:
            logger.warning("Could not load reference metrics")
            return None
        
        prod_metrics = evaluate_production_performance()
        
        if prod_metrics is None:
            logger.warning("Could not evaluate production metrics")
            return None
        
        for key, value in ref_metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                mlflow.log_metric(f"reference_{key}", value)
        
        for key, value in prod_metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                mlflow.log_metric(f"production_{key}", value)
        
        if "class_proportions" in ref_metrics and "class_proportions" in prod_metrics:
            ref_props = ref_metrics["class_proportions"]
            prod_props = prod_metrics["class_proportions"]
            
            for class_label in ref_props.keys():
                if class_label in prod_props:
                    ref_prop = ref_props[class_label]
                    prod_prop = prod_props[class_label]
                    change = prod_prop - ref_prop
                    mlflow.log_metric(f"class_{class_label}_proportion_change", change)
        
        logger.info("Continuous evaluation completed")
        return {"reference": ref_metrics, "production": prod_metrics}


if __name__ == "__main__":
    run_continuous_evaluation()

