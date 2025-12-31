"""
Feature validation with statistical checks (Great Expectations style).
Validates feature distributions, detects skew, and checks data quality.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to Python path if not already there
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureValidator:
    """Validates features using statistical checks."""
    
    def __init__(self, reference_data: pd.DataFrame):
        """
        Initialize validator with reference data.
        
        Args:
            reference_data: Reference dataset to use as baseline
        """
        self.reference_data = reference_data
        self.reference_stats = self._calculate_reference_stats()
    
    def _calculate_reference_stats(self) -> Dict:
        """Calculate reference statistics for all features."""
        stats_dict = {}
        
        for col in self.reference_data.columns:
            if col in ["target", "price_category", "timestamp"]:
                continue
            
            series = self.reference_data[col].dropna()
            
            if len(series) == 0:
                continue
            
            col_stats = {"column": col}
            
            if pd.api.types.is_numeric_dtype(series):
                col_stats["type"] = "numerical"
                col_stats["mean"] = float(series.mean())
                col_stats["std"] = float(series.std())
                col_stats["median"] = float(series.median())
                col_stats["min"] = float(series.min())
                col_stats["max"] = float(series.max())
                col_stats["q25"] = float(series.quantile(0.25))
                col_stats["q75"] = float(series.quantile(0.75))
                col_stats["iqr"] = float(col_stats["q75"] - col_stats["q25"])
                
                # Skewness and kurtosis
                col_stats["skewness"] = float(stats.skew(series.dropna()))
                col_stats["kurtosis"] = float(stats.kurtosis(series.dropna()))
                
                # Missing values
                col_stats["missing_count"] = int(series.isna().sum())
                col_stats["missing_pct"] = float(series.isna().sum() / len(series))
            else:
                col_stats["type"] = "categorical"
                col_stats["unique_count"] = int(series.nunique())
                col_stats["value_counts"] = series.value_counts().to_dict()
                col_stats["most_frequent"] = series.mode().iloc[0] if len(series.mode()) > 0 else None
                col_stats["most_frequent_pct"] = float(
                    (series == col_stats["most_frequent"]).sum() / len(series)
                ) if col_stats["most_frequent"] is not None else 0.0
                col_stats["missing_count"] = int(series.isna().sum())
                col_stats["missing_pct"] = float(series.isna().sum() / len(series))
            
            stats_dict[col] = col_stats
        
        return stats_dict
    
    def validate_production_data(
        self,
        production_data: pd.DataFrame,
        tolerance: float = 0.1,
        skew_threshold: float = 2.0,
    ) -> Dict:
        """
        Validate production data against reference statistics.
        
        Args:
            production_data: Production data to validate
            tolerance: Relative tolerance for numerical features (default 10%)
            skew_threshold: Threshold for skewness detection (default 2.0)
        
        Returns:
            Dictionary with validation results and alerts
        """
        validation_results = {
            "validations": {},
            "alerts": [],
            "warnings": [],
            "passed": True,
        }
        
        for col, ref_stats in self.reference_stats.items():
            if col not in production_data.columns:
                validation_results["warnings"].append({
                    "column": col,
                    "type": "missing_column",
                    "message": f"Column {col} missing in production data",
                })
                continue
            
            prod_series = production_data[col].dropna()
            
            if len(prod_series) == 0:
                validation_results["alerts"].append({
                    "column": col,
                    "type": "empty_column",
                    "message": f"Column {col} is empty in production data",
                })
                validation_results["passed"] = False
                continue
            
            col_validation = {
                "column": col,
                "type": ref_stats["type"],
                "checks": [],
                "passed": True,
            }
            
            if ref_stats["type"] == "numerical":
                # Check mean shift
                prod_mean = prod_series.mean()
                mean_shift = abs(prod_mean - ref_stats["mean"]) / max(abs(ref_stats["mean"]), 1e-6)
                
                if mean_shift > tolerance:
                    col_validation["checks"].append({
                        "check": "mean_shift",
                        "status": "failed",
                        "reference": ref_stats["mean"],
                        "production": prod_mean,
                        "shift": mean_shift,
                    })
                    col_validation["passed"] = False
                    validation_results["alerts"].append({
                        "column": col,
                        "type": "mean_shift",
                        "severity": "high" if mean_shift > 0.2 else "medium",
                        "message": f"Mean shift of {mean_shift*100:.2f}% detected",
                        "reference": ref_stats["mean"],
                        "production": prod_mean,
                    })
                else:
                    col_validation["checks"].append({
                        "check": "mean_shift",
                        "status": "passed",
                        "reference": ref_stats["mean"],
                        "production": prod_mean,
                        "shift": mean_shift,
                    })
                
                # Check std shift
                prod_std = prod_series.std()
                std_shift = abs(prod_std - ref_stats["std"]) / max(ref_stats["std"], 1e-6)
                
                if std_shift > tolerance:
                    col_validation["checks"].append({
                        "check": "std_shift",
                        "status": "failed",
                        "reference": ref_stats["std"],
                        "production": prod_std,
                        "shift": std_shift,
                    })
                    validation_results["warnings"].append({
                        "column": col,
                        "type": "std_shift",
                        "message": f"Std deviation shift of {std_shift*100:.2f}% detected",
                    })
                
                # Check for outliers (using IQR method)
                q1 = prod_series.quantile(0.25)
                q3 = prod_series.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = ((prod_series < lower_bound) | (prod_series > upper_bound)).sum()
                outlier_pct = outliers / len(prod_series)
                
                if outlier_pct > 0.05:  # More than 5% outliers
                    col_validation["checks"].append({
                        "check": "outliers",
                        "status": "warning",
                        "outlier_count": int(outliers),
                        "outlier_pct": outlier_pct,
                    })
                    validation_results["warnings"].append({
                        "column": col,
                        "type": "high_outliers",
                        "message": f"{outlier_pct*100:.2f}% outliers detected",
                    })
                
                # Check skewness
                prod_skew = stats.skew(prod_series.dropna())
                if abs(prod_skew) > skew_threshold:
                    col_validation["checks"].append({
                        "check": "skewness",
                        "status": "warning",
                        "reference_skew": ref_stats["skewness"],
                        "production_skew": prod_skew,
                    })
                    validation_results["warnings"].append({
                        "column": col,
                        "type": "high_skew",
                        "message": f"High skewness detected: {prod_skew:.2f}",
                    })
                
                # Check missing values
                prod_missing_pct = prod_series.isna().sum() / len(production_data)
                if prod_missing_pct > ref_stats["missing_pct"] + 0.05:  # 5% increase
                    col_validation["checks"].append({
                        "check": "missing_values",
                        "status": "warning",
                        "reference_missing_pct": ref_stats["missing_pct"],
                        "production_missing_pct": prod_missing_pct,
                    })
                    validation_results["warnings"].append({
                        "column": col,
                        "type": "increased_missing",
                        "message": f"Missing values increased from {ref_stats['missing_pct']*100:.2f}% to {prod_missing_pct*100:.2f}%",
                    })
            
            else:  # Categorical
                # Check for new categories
                ref_categories = set(ref_stats.get("value_counts", {}).keys())
                prod_categories = set(prod_series.unique())
                new_categories = prod_categories - ref_categories
                
                if new_categories:
                    new_cat_pct = sum(
                        (prod_series == cat).sum() for cat in new_categories
                    ) / len(prod_series)
                    
                    col_validation["checks"].append({
                        "check": "new_categories",
                        "status": "warning" if new_cat_pct < 0.1 else "failed",
                        "new_categories": list(new_categories),
                        "new_categories_pct": new_cat_pct,
                    })
                    
                    if new_cat_pct > 0.1:
                        validation_results["alerts"].append({
                            "column": col,
                            "type": "new_categories",
                            "severity": "high",
                            "message": f"{new_cat_pct*100:.2f}% of data in new categories",
                            "new_categories": list(new_categories),
                        })
                        col_validation["passed"] = False
                    else:
                        validation_results["warnings"].append({
                            "column": col,
                            "type": "new_categories",
                            "message": f"New categories detected: {list(new_categories)}",
                        })
                
                # Check distribution shift (using chi-square test)
                if len(ref_categories) > 0 and len(prod_categories) > 0:
                    common_cats = ref_categories & prod_categories
                    if len(common_cats) > 1:
                        ref_counts = [
                            ref_stats["value_counts"].get(cat, 0) for cat in common_cats
                        ]
                        prod_counts = [
                            (prod_series == cat).sum() for cat in common_cats
                        ]
                        
                        # Normalize to proportions
                        ref_total = sum(ref_counts)
                        prod_total = sum(prod_counts)
                        
                        if ref_total > 0 and prod_total > 0:
                            ref_props = [c / ref_total for c in ref_counts]
                            prod_props = [c / prod_total for c in prod_counts]
                            
                            # Chi-square test
                            try:
                                chi2, p_value = stats.chisquare(
                                    [p * prod_total for p in prod_props],
                                    [p * prod_total for p in ref_props],
                                )
                                
                                if p_value < 0.05:  # Significant distribution shift
                                    col_validation["checks"].append({
                                        "check": "distribution_shift",
                                        "status": "failed",
                                        "chi2": chi2,
                                        "p_value": p_value,
                                    })
                                    col_validation["passed"] = False
                                    validation_results["alerts"].append({
                                        "column": col,
                                        "type": "distribution_shift",
                                        "severity": "medium",
                                        "message": f"Significant distribution shift (p={p_value:.4f})",
                                    })
                            except Exception as e:
                                logger.warning(f"Chi-square test failed for {col}: {e}")
            
            validation_results["validations"][col] = col_validation
            
            if not col_validation["passed"]:
                validation_results["passed"] = False
        
        return validation_results


def validate_features(reference_data: pd.DataFrame, production_data: pd.DataFrame) -> Dict:
    """
    Convenience function to validate production features against reference.
    
    Args:
        reference_data: Reference dataset
        production_data: Production dataset
    
    Returns:
        Validation results dictionary
    """
    validator = FeatureValidator(reference_data)
    return validator.validate_production_data(production_data)

