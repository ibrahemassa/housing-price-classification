"""
MLOps Monitoring Dashboard
A comprehensive dashboard for monitoring ML model performance, drift detection, and feature validation.
"""

import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.stats import ks_2samp

from src.monitoring.monitor_flow import (
    CARDINALITY_THRESHOLD,
    CAT_FEATURE,
    HIGH_CARD_FEATURE,
    PRED_DRIFT_THRESHOLD,
    PSI_THRESHOLD,
    categorical_psi,
    numerical_psi,
)

CATPPUCCIN_COLORS = {
    "base": "#1e1e2e",  # Base
    "mantle": "#181825",  # Mantle
    "crust": "#11111b",  # Crust
    "text": "#cdd6f4",  # Text
    "subtext0": "#a6adc8",  # Subtext0
    "subtext1": "#bac2de",  # Subtext1
    "surface0": "#313244",  # Surface0
    "surface1": "#45475a",  # Surface1
    "surface2": "#585b70",  # Surface2
    "overlay0": "#6c7086",  # Overlay0
    "overlay1": "#7f849c",  # Overlay1
    "overlay2": "#9399b2",  # Overlay2
    "blue": "#89b4fa",  # Blue
    "lavender": "#b4befe",  # Lavender
    "sapphire": "#74c7ec",  # Sapphire
    "sky": "#89dceb",  # Sky
    "teal": "#94e2d5",  # Teal
    "green": "#a6e3a1",  # Green
    "yellow": "#f9e2af",  # Yellow
    "peach": "#fab387",  # Peach
    "maroon": "#eba0ac",  # Maroon
    "red": "#f38ba8",  # Red
    "mauve": "#cba6f7",  # Mauve
    "pink": "#f5c2e7",  # Pink
    "flamingo": "#f2cdcd",  # Flamingo
    "rosewater": "#f5e0dc",  # Rosewater
}

# Page configuration with Catppuccin theme
st.set_page_config(
    page_title="MLOps Monitoring Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom CSS for Catppuccin theme
st.markdown(
    f"""
<style>
    /* Main background */
    .stApp {{
        background-color: {CATPPUCCIN_COLORS['base']};
        color: {CATPPUCCIN_COLORS['text']};
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {CATPPUCCIN_COLORS['mantle']};
    }}

    /* Headers */
    h1, h2, h3, h4, h5, h6 {{
        color: {CATPPUCCIN_COLORS['text']};
    }}

    /* Text */
    p, div, span {{
        color: {CATPPUCCIN_COLORS['text']};
    }}

    /* Metrics */
    [data-testid="stMetricValue"] {{
        color: {CATPPUCCIN_COLORS['blue']};
    }}

    /* Buttons */
    .stButton > button {{
        background-color: {CATPPUCCIN_COLORS['blue']};
        color: {CATPPUCCIN_COLORS['base']};
        border: none;
    }}

    .stButton > button:hover {{
        background-color: {CATPPUCCIN_COLORS['sapphire']};
    }}

    /* Selectbox */
    [data-baseweb="select"] {{
        background-color: {CATPPUCCIN_COLORS['surface0']};
        color: {CATPPUCCIN_COLORS['text']};
    }}

    /* Dataframes */
    .dataframe {{
        background-color: {CATPPUCCIN_COLORS['surface0']};
        color: {CATPPUCCIN_COLORS['text']};
    }}

    /* Success messages */
    .stSuccess {{
        background-color: {CATPPUCCIN_COLORS['green']}20;
        border-left: 4px solid {CATPPUCCIN_COLORS['green']};
    }}

    /* Warning messages */
    .stWarning {{
        background-color: {CATPPUCCIN_COLORS['yellow']}20;
        border-left: 4px solid {CATPPUCCIN_COLORS['yellow']};
    }}

    /* Error messages */
    .stError {{
        background-color: {CATPPUCCIN_COLORS['red']}20;
        border-left: 4px solid {CATPPUCCIN_COLORS['red']};
    }}

    /* Info messages */
    .stInfo {{
        background-color: {CATPPUCCIN_COLORS['blue']}20;
        border-left: 4px solid {CATPPUCCIN_COLORS['blue']};
    }}
</style>
""",
    unsafe_allow_html=True,
)

# Constants
REFERENCE_PATH = Path("data/reference/reference.parquet")
PROD_INPUTS = Path("data/production/inputs.parquet")
PROD_PREDS = Path("data/production/predictions.parquet")
PLOT_DIR = Path("data/monitoring/plots")

# MLflow setup
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(MLFLOW_URI)


@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_reference_data():
    """Load reference dataset."""
    if REFERENCE_PATH.exists():
        return pd.read_parquet(REFERENCE_PATH)
    return None


@st.cache_data(ttl=60)  # Cache for 1 minute
def load_production_data():
    """Load production inputs and predictions."""
    inputs = None
    preds = None

    if PROD_INPUTS.exists():
        inputs = pd.read_parquet(PROD_INPUTS)
        if "timestamp" in inputs.columns:
            inputs["timestamp"] = pd.to_datetime(inputs["timestamp"])

    if PROD_PREDS.exists():
        preds = pd.read_parquet(PROD_PREDS)
        if "timestamp" in preds.columns:
            preds["timestamp"] = pd.to_datetime(preds["timestamp"])

    return inputs, preds


@st.cache_data(ttl=300)
def get_mlflow_metrics():
    """Fetch latest monitoring metrics from MLflow."""
    try:
        mlflow.set_experiment("monitoring")
        runs = mlflow.search_runs(
            experiment_ids=None,
            order_by=["start_time DESC"],
            max_results=10,
        )

        if runs.empty:
            return (None, None)

        # Get latest run
        latest_run = runs.iloc[0]

        metrics = {
            "district_psi": latest_run.get("metrics.district_psi", None),
            "address_cardinality_ratio": latest_run.get(
                "metrics.address_cardinality_ratio", None
            ),
            "prediction_kl_drift": latest_run.get("metrics.prediction_kl_drift", None),
            "total_alerts": latest_run.get("metrics.total_alerts", None),
            "alerts": latest_run.get(
                "metrics.total_alerts", None
            ),  # Backward compatibility
            "drift_severity": latest_run.get("metrics.drift_severity", None),
            "feature_validation_alerts": latest_run.get(
                "metrics.feature_validation_alerts", None
            ),
            "run_id": latest_run.get("run_id", None),
            "start_time": latest_run.get("start_time", None),
        }

        # Convert None to 0 for numeric metrics to avoid NaN
        numeric_metrics = [
            "district_psi",
            "address_cardinality_ratio",
            "prediction_kl_drift",
            "total_alerts",
            "alerts",
            "drift_severity",
            "feature_validation_alerts",
        ]
        for key in numeric_metrics:
            if key in metrics and (metrics[key] is None or pd.isna(metrics[key])):
                metrics[key] = 0

        return (metrics, runs)
    except Exception as e:
        st.error(f"Error fetching MLflow metrics: {e}")
        return (None, None)


@st.cache_data(ttl=300)
def get_model_metrics():
    """Fetch model performance metrics from MLflow."""
    try:
        # Use the correct experiment name from train.py
        mlflow.set_experiment("housing-price-classification")
        runs = mlflow.search_runs(
            experiment_ids=None,
            order_by=["start_time DESC"],
            max_results=5,
        )

        if runs.empty:
            return None

        metrics_list = []
        for _, run in runs.iterrows():
            metrics_dict = {
                "run_id": run.get("run_id", ""),
                "accuracy": run.get("metrics.accuracy", None),
                "macro_precision": run.get("metrics.macro_precision", None),
                "macro_recall": run.get("metrics.macro_recall", None),
                "macro_f1": run.get("metrics.macro_f1", None),
                "weighted_precision": run.get("metrics.weighted_precision", None),
                "weighted_recall": run.get("metrics.weighted_recall", None),
                "weighted_f1": run.get("metrics.weighted_f1", None),
                "roc_auc_macro": run.get("metrics.roc_auc_macro", None),
                "roc_auc_weighted": run.get("metrics.roc_auc_weighted", None),
                # Per-class metrics:
                "precision_low": run.get("metrics.precision_low", None),
                "recall_low": run.get("metrics.recall_low", None),
                "f1_low": run.get("metrics.f1_low", None),
                "precision_medium": run.get("metrics.precision_medium", None),
                "recall_medium": run.get("metrics.recall_medium", None),
                "f1_medium": run.get("metrics.f1_medium", None),
                "precision_high": run.get("metrics.precision_high", None),
                "recall_high": run.get("metrics.recall_high", None),
                "f1_high": run.get("metrics.f1_high", None),
                "model": run.get("params.model", "Unknown"),
                "start_time": run.get("start_time", None),
            }
            metrics_list.append(metrics_dict)

        return pd.DataFrame(metrics_list)
    except Exception as e:
        st.error(f"Error fetching model metrics: {e}")
        return None


# Use the corrected numerical_psi from monitor_flow
calculate_numerical_psi = numerical_psi


def safe_get_metric(metrics_dict, key, default=0):
    """Safely get metric value, handling None and NaN."""
    if metrics_dict is None:
        return default

    value = metrics_dict.get(key, default)

    # Handle None and NaN
    if value is None or (isinstance(value, (int, float)) and pd.isna(value)):
        return default

    return value


def apply_catppuccin_theme(fig):
    """Apply Catppuccin Mocha theme to a Plotly figure."""
    fig.update_layout(
        plot_bgcolor=CATPPUCCIN_COLORS["base"],
        paper_bgcolor=CATPPUCCIN_COLORS["mantle"],
        font_color=CATPPUCCIN_COLORS["text"],
        title_font_color=CATPPUCCIN_COLORS["text"],
        xaxis={
            "gridcolor": CATPPUCCIN_COLORS["surface0"],
            "linecolor": CATPPUCCIN_COLORS["overlay0"],
        },
        yaxis={
            "gridcolor": CATPPUCCIN_COLORS["surface0"],
            "linecolor": CATPPUCCIN_COLORS["overlay0"],
        },
    )
    return fig


def get_prefect_flow_status():
    """Get Prefect flow run status."""
    try:
        from prefect import get_client

        client = get_client()
        # Get recent flow runs
        flows = client.read_flows()
        runs = []
        for flow in flows[:5]:  # Get last 5 flows
            flow_runs = client.read_flow_runs(flow_filter={"id": {"any_": [flow.id]}})
            for run in flow_runs[:3]:  # Get last 3 runs per flow
                runs.append(
                    {
                        "flow_name": flow.name,
                        "run_id": str(run.id),
                        "state": run.state_type.value if run.state_type else "UNKNOWN",
                        "start_time": run.start_time,
                        "end_time": run.end_time,
                    }
                )
        return sorted(runs, key=lambda x: x.get("start_time") or "", reverse=True)[:10]
    except Exception:
        # Prefect not available or not configured
        return None


def calculate_feature_statistics(ref_df, prod_df):
    """Calculate comprehensive statistics for all features."""
    stats_list = []

    for col in ref_df.columns:
        if col == "target" or col == "price_category":
            continue

        if col not in prod_df.columns:
            continue

        ref_series = ref_df[col].dropna()
        prod_series = prod_df[col].dropna()

        if len(ref_series) == 0 or len(prod_series) == 0:
            continue

        stat_dict = {"feature": col}

        # Check if numerical or categorical
        if pd.api.types.is_numeric_dtype(ref_series):
            # Numerical statistics
            stat_dict["type"] = "numerical"
            stat_dict["ref_mean"] = ref_series.mean()
            stat_dict["prod_mean"] = prod_series.mean()
            stat_dict["ref_std"] = ref_series.std()
            stat_dict["prod_std"] = prod_series.std()
            stat_dict["ref_median"] = ref_series.median()
            stat_dict["prod_median"] = prod_series.median()

            # Drift metrics
            stat_dict["psi"] = calculate_numerical_psi(ref_series, prod_series)

            # Kolmogorov-Smirnov test
            if len(ref_series) > 0 and len(prod_series) > 0:
                ks_stat, ks_pvalue = ks_2samp(ref_series, prod_series)
                stat_dict["ks_statistic"] = ks_stat
                stat_dict["ks_pvalue"] = ks_pvalue
                stat_dict["drift_detected"] = ks_pvalue < 0.05
        else:
            # Categorical statistics
            stat_dict["type"] = "categorical"
            stat_dict["ref_unique"] = ref_series.nunique()
            stat_dict["prod_unique"] = prod_series.nunique()
            stat_dict["cardinality_ratio"] = stat_dict["prod_unique"] / max(
                stat_dict["ref_unique"], 1
            )

            # PSI for categorical
            stat_dict["psi"] = categorical_psi(ref_series, prod_series)
            stat_dict["drift_detected"] = stat_dict["psi"] > PSI_THRESHOLD

        stats_list.append(stat_dict)

    return pd.DataFrame(stats_list)


def main():
    """Main dashboard application."""
    st.title("📊 MLOps Monitoring Dashboard")
    st.markdown("**Continuous Model Evaluation & Drift Detection**")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        [
            "Overview",
            "Drift Monitoring",
            "Feature Statistics",
            "Performance Metrics",
            "Production Data",
            "Alerts",
        ],
    )

    # Load data
    ref_data = load_reference_data()
    prod_inputs, prod_preds = load_production_data()
    mlflow_metrics, mlflow_runs = get_mlflow_metrics()
    model_metrics_df = get_model_metrics()
    prefect_runs = get_prefect_flow_status()

    # Check if data is available and route pages
    if page == "Overview":
        if ref_data is None:
            st.warning(
                "⚠️ Reference data not found. Please ensure reference.parquet exists."
            )
        if prod_inputs is None or prod_preds is None:
            st.warning(
                "⚠️ Production data not found. Waiting for production predictions..."
            )
            st.info(
                "The dashboard will update automatically once production data is available."
            )
        if ref_data is not None:
            show_overview(
                ref_data,
                prod_inputs,
                prod_preds,
                mlflow_metrics,
                model_metrics_df,
                prefect_runs,
            )
    elif page == "Drift Monitoring":
        if ref_data is None or prod_inputs is None or prod_preds is None:
            st.warning("⚠️ Reference and production data required for drift monitoring.")
        else:
            show_drift_monitoring(ref_data, prod_inputs, prod_preds, mlflow_metrics)
    elif page == "Feature Statistics":
        if ref_data is None or prod_inputs is None:
            st.warning(
                "⚠️ Reference and production data required for feature statistics."
            )
        else:
            show_feature_statistics(ref_data, prod_inputs)
    elif page == "Performance Metrics":
        show_performance_metrics(model_metrics_df, ref_data, prod_preds)
    elif page == "Production Data":
        if prod_inputs is None or prod_preds is None:
            st.warning(
                "⚠️ Production data not found. Waiting for production predictions..."
            )
        else:
            show_production_data(prod_inputs, prod_preds)
    elif page == "Alerts":
        if ref_data is None:
            st.warning(
                "⚠️ Reference data not found. Please ensure reference.parquet exists."
            )
        show_alerts(ref_data, prod_inputs, prod_preds, mlflow_metrics)


def show_overview(
    ref_data,
    prod_inputs,
    prod_preds,
    mlflow_metrics,
    model_metrics_df,
    prefect_runs=None,
):
    """Overview dashboard with key metrics."""
    st.header("📈 Overview Dashboard")

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Predictions",
            len(prod_preds),
            delta=f"+{len(prod_preds)} total",
        )

    with col2:
        if mlflow_metrics:
            # Get alerts from total_alerts or alerts (for backward compatibility)
            alerts = safe_get_metric(mlflow_metrics, "total_alerts") or safe_get_metric(
                mlflow_metrics, "alerts", 0
            )
            alerts = int(alerts)

            st.metric(
                "Active Alerts",
                alerts,
                delta="⚠️" if alerts > 0 else "✅",
                delta_color="inverse",
            )
        else:
            st.metric("Active Alerts", "N/A")

    with col3:
        if mlflow_metrics:
            psi = safe_get_metric(mlflow_metrics, "district_psi", 0.0)
            psi = float(psi)

            status = "⚠️" if psi > PSI_THRESHOLD else "✅"
            st.metric(
                "District PSI",
                f"{psi:.3f}",
                delta=status,
                delta_color="inverse",
            )
        else:
            st.metric("District PSI", "N/A")

    with col4:
        if model_metrics_df is not None and not model_metrics_df.empty:
            latest_acc = model_metrics_df.iloc[0].get("accuracy")
            if latest_acc is not None:
                st.metric(
                    "Model Accuracy",
                    f"{latest_acc:.3f}",
                )
            else:
                st.metric("Model Accuracy", "N/A")
        else:
            st.metric("Model Accuracy", "N/A")

    st.divider()

    # Recent predictions timeline
    st.subheader("📊 Recent Predictions Timeline")
    if "timestamp" in prod_preds.columns:
        preds_by_time = (
            prod_preds.groupby(prod_preds["timestamp"].dt.floor("H"))
            .size()
            .reset_index(name="count")
        )

        fig = px.line(
            preds_by_time,
            x="timestamp",
            y="count",
            title="Predictions Over Time",
            labels={"timestamp": "Time", "count": "Number of Predictions"},
            color_discrete_sequence=[CATPPUCCIN_COLORS["blue"]],
        )
        fig.update_layout(height=300)
        fig = apply_catppuccin_theme(fig)
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Timestamp data not available for predictions.")

    # Prediction distribution
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 Prediction Distribution")
        pred_counts = prod_preds["prediction"].value_counts().sort_index()
        categories = {0: "Low", 1: "Medium", 2: "High"}
        pred_labels = [categories.get(int(k), str(k)) for k in pred_counts.index]

        fig = px.pie(
            values=pred_counts.values,
            names=pred_labels,
            title="Production Predictions",
            color_discrete_sequence=[
                CATPPUCCIN_COLORS["blue"],
                CATPPUCCIN_COLORS["green"],
                CATPPUCCIN_COLORS["yellow"],
            ],
        )
        fig = apply_catppuccin_theme(fig)
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.subheader("📉 Model Performance Trend")
        if model_metrics_df is not None and not model_metrics_df.empty:
            # Filter out None values
            df_clean = model_metrics_df.dropna(subset=["accuracy", "macro_f1"])
            if not df_clean.empty:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=df_clean.index,
                        y=df_clean["accuracy"],
                        name="Accuracy",
                        mode="lines+markers",
                        line={"color": CATPPUCCIN_COLORS["blue"], "width": 2},
                        marker={"color": CATPPUCCIN_COLORS["blue"]},
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=df_clean.index,
                        y=df_clean["macro_f1"],
                        name="Macro F1",
                        mode="lines+markers",
                        line={"color": CATPPUCCIN_COLORS["green"], "width": 2},
                        marker={"color": CATPPUCCIN_COLORS["green"]},
                    )
                )
                fig.update_layout(
                    title="Model Performance Over Time",
                    xaxis_title="Run Index",
                    yaxis_title="Score",
                    height=300,
                )
                fig = apply_catppuccin_theme(fig)
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("No performance metrics available.")
        else:
            st.info("No model metrics available.")

    # Prefect Flow Status
    st.divider()
    st.subheader("🔄 Prefect Flow Status")

    if prefect_runs:
        # Show recent flow runs
        flow_df = pd.DataFrame(prefect_runs)
        # Format the dataframe
        display_cols = ["flow_name", "state", "start_time"]
        available_cols = [col for col in display_cols if col in flow_df.columns]
        if available_cols:
            st.dataframe(flow_df[available_cols], width="stretch")
    else:
        st.info(
            "Prefect client not available. Ensure Prefect server is running and configured."
        )


def show_drift_monitoring(ref_data, prod_inputs, prod_preds, mlflow_metrics):
    """Drift monitoring dashboard."""
    st.header("🔍 Drift Monitoring")
    st.markdown("**Data Distribution Shift Detection**")

    # Drift metrics summary
    if mlflow_metrics:
        metrics = mlflow_metrics

        col1, col2, col3 = st.columns(3)

        with col1:
            psi = safe_get_metric(metrics, "district_psi", 0.0)
            psi = float(psi)
            status = "⚠️ Drift Detected" if psi > PSI_THRESHOLD else "✅ Stable"
            st.metric(
                "District PSI",
                f"{psi:.4f}",
                delta=status,
                delta_color="inverse",
            )
            st.caption(f"Threshold: {PSI_THRESHOLD}")

        with col2:
            card_ratio = safe_get_metric(metrics, "address_cardinality_ratio", 0.0)
            card_ratio = float(card_ratio)
            status = (
                "⚠️ High Growth" if card_ratio > CARDINALITY_THRESHOLD else "✅ Normal"
            )
            st.metric(
                "Address Cardinality Ratio",
                f"{card_ratio:.2f}",
                delta=status,
                delta_color="inverse",
            )
            st.caption(f"Threshold: {CARDINALITY_THRESHOLD}")

        with col3:
            pred_drift = safe_get_metric(metrics, "prediction_kl_drift", 0.0)
            pred_drift = float(pred_drift)
            status = (
                "⚠️ Drift Detected" if pred_drift > PRED_DRIFT_THRESHOLD else "✅ Stable"
            )
            st.metric(
                "Prediction KL Drift",
                f"{pred_drift:.4f}",
                delta=status,
                delta_color="inverse",
            )
            st.caption(f"Threshold: {PRED_DRIFT_THRESHOLD}")

    st.divider()

    # Categorical feature drift
    st.subheader("📊 Categorical Feature Drift")

    col1, col2 = st.columns(2)

    with col1:
        # District distribution comparison
        ref_dist = ref_data[CAT_FEATURE].value_counts(normalize=True).head(10)
        prod_dist = prod_inputs[CAT_FEATURE].value_counts(normalize=True).head(10)

        # Align indices
        all_districts = ref_dist.index.union(prod_dist.index)
        ref_aligned = ref_dist.reindex(all_districts, fill_value=0)
        prod_aligned = prod_dist.reindex(all_districts, fill_value=0)

        df_dist = pd.DataFrame(
            {
                "Reference": ref_aligned,
                "Production": prod_aligned,
            }
        )

        fig = px.bar(
            df_dist,
            barmode="group",
            title=f"{CAT_FEATURE.title()} Distribution Comparison",
            labels={"value": "Proportion", "index": "District"},
            color_discrete_sequence=[
                CATPPUCCIN_COLORS["blue"],
                CATPPUCCIN_COLORS["green"],
            ],
        )
        fig = apply_catppuccin_theme(fig)
        st.plotly_chart(fig, width="stretch")

    with col2:
        # Address cardinality
        ref_card = ref_data[HIGH_CARD_FEATURE].nunique()
        prod_card = prod_inputs[HIGH_CARD_FEATURE].nunique()

        df_card = pd.DataFrame(
            {
                "Dataset": ["Reference", "Production"],
                "Unique Values": [ref_card, prod_card],
            }
        )

        fig = px.bar(
            df_card,
            x="Dataset",
            y="Unique Values",
            title=f"{HIGH_CARD_FEATURE.title()} Cardinality",
            color="Dataset",
            color_discrete_sequence=[
                CATPPUCCIN_COLORS["blue"],
                CATPPUCCIN_COLORS["green"],
            ],
        )
        fig = apply_catppuccin_theme(fig)
        st.plotly_chart(fig, width="stretch")

    # Prediction drift
    st.subheader("🎯 Prediction Distribution Drift")

    if "price_category" in ref_data.columns:
        ref_pred_dist = (
            ref_data["price_category"].value_counts(normalize=True).sort_index()
        )
    else:
        # If price_category doesn't exist, use target
        ref_pred_dist = ref_data["target"].value_counts(normalize=True).sort_index()

    prod_pred_dist = prod_preds["prediction"].value_counts(normalize=True).sort_index()

    # Align indices
    all_preds = ref_pred_dist.index.union(prod_pred_dist.index)
    ref_pred_aligned = ref_pred_dist.reindex(all_preds, fill_value=0)
    prod_pred_aligned = prod_pred_dist.reindex(all_preds, fill_value=0)

    categories = {0: "Low", 1: "Medium", 2: "High"}
    pred_labels = [categories.get(int(k), str(k)) for k in all_preds]

    df_pred = pd.DataFrame(
        {
            "Reference": ref_pred_aligned.values,
            "Production": prod_pred_aligned.values,
        },
        index=pred_labels,
    )

    fig = px.bar(
        df_pred,
        barmode="group",
        title="Prediction Distribution Comparison",
        labels={"value": "Proportion", "index": "Price Category"},
        color_discrete_sequence=[CATPPUCCIN_COLORS["blue"], CATPPUCCIN_COLORS["green"]],
    )
    fig = apply_catppuccin_theme(fig)
    st.plotly_chart(fig, width="stretch")

    # Numerical feature drift (if available)
    st.subheader("📈 Numerical Feature Drift")

    numeric_cols = prod_inputs.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ["timestamp"]]

    if numeric_cols:
        selected_numeric = st.selectbox("Select Numerical Feature", numeric_cols)

        if selected_numeric in ref_data.columns:
            col1, col2 = st.columns(2)

            with col1:
                # Distribution comparison
                fig = go.Figure()
                fig.add_trace(
                    go.Histogram(
                        x=ref_data[selected_numeric].dropna(),
                        name="Reference",
                        opacity=0.7,
                        nbinsx=30,
                        marker_color=CATPPUCCIN_COLORS["blue"],
                    )
                )
                fig.add_trace(
                    go.Histogram(
                        x=prod_inputs[selected_numeric].dropna(),
                        name="Production",
                        opacity=0.7,
                        nbinsx=30,
                        marker_color=CATPPUCCIN_COLORS["green"],
                    )
                )
                fig.update_layout(
                    title=f"{selected_numeric} Distribution",
                    xaxis_title=selected_numeric,
                    yaxis_title="Frequency",
                    barmode="overlay",
                )
                fig = apply_catppuccin_theme(fig)
                st.plotly_chart(fig, width="stretch")

            with col2:
                # Box plot comparison
                ref_values = ref_data[selected_numeric].dropna()
                prod_values = prod_inputs[selected_numeric].dropna()

                fig = go.Figure()
                fig.add_trace(
                    go.Box(
                        y=ref_values,
                        name="Reference",
                        marker_color=CATPPUCCIN_COLORS["blue"],
                    )
                )
                fig.add_trace(
                    go.Box(
                        y=prod_values,
                        name="Production",
                        marker_color=CATPPUCCIN_COLORS["green"],
                    )
                )
                fig.update_layout(
                    title=f"{selected_numeric} Box Plot Comparison",
                    yaxis_title=selected_numeric,
                )
                fig = apply_catppuccin_theme(fig)
                st.plotly_chart(fig, width="stretch")

            # Calculate and display PSI
            psi = calculate_numerical_psi(
                ref_data[selected_numeric].dropna(),
                prod_inputs[selected_numeric].dropna(),
            )
            st.metric(f"{selected_numeric} PSI", f"{psi:.4f}")


def show_feature_statistics(ref_data, prod_inputs):
    """Feature statistics and validation dashboard."""
    st.header("📊 Feature Statistics & Validation")
    st.markdown("**Comprehensive Feature Analysis**")

    # Calculate feature statistics
    feature_stats = calculate_feature_statistics(ref_data, prod_inputs)

    if feature_stats.empty:
        st.warning("No feature statistics available.")
        return

    # Summary table
    st.subheader("Feature Summary")

    # Format the dataframe for display
    display_cols = ["feature", "type", "psi", "drift_detected"]
    if "cardinality_ratio" in feature_stats.columns:
        display_cols.insert(-1, "cardinality_ratio")

    available_cols = [col for col in display_cols if col in feature_stats.columns]
    display_df = feature_stats[available_cols].copy()

    # Format drift_detected
    if "drift_detected" in display_df.columns:
        display_df["drift_detected"] = display_df["drift_detected"].map(
            {True: "⚠️ Yes", False: "✅ No"}
        )

    # Format PSI
    if "psi" in display_df.columns:
        display_df["psi"] = display_df["psi"].round(4)

    st.dataframe(display_df, width="stretch")

    st.divider()

    # Detailed feature analysis
    st.subheader("Detailed Feature Analysis")

    selected_feature = st.selectbox("Select Feature", feature_stats["feature"].tolist())

    feature_row = feature_stats[feature_stats["feature"] == selected_feature].iloc[0]

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Feature Type", feature_row["type"].title())
        if "psi" in feature_row:
            st.metric("PSI", f"{feature_row['psi']:.4f}")
        if "drift_detected" in feature_row:
            status = (
                "⚠️ Drift Detected" if feature_row["drift_detected"] else "✅ No Drift"
            )
            st.metric("Drift Status", status)

    with col2:
        if feature_row["type"] == "numerical":
            if "ref_mean" in feature_row:
                st.metric("Reference Mean", f"{feature_row['ref_mean']:.2f}")
                st.metric("Production Mean", f"{feature_row['prod_mean']:.2f}")
            if "ks_pvalue" in feature_row:
                st.metric("KS p-value", f"{feature_row['ks_pvalue']:.4f}")
        else:
            if "ref_unique" in feature_row:
                st.metric("Reference Unique", int(feature_row["ref_unique"]))
                st.metric("Production Unique", int(feature_row["prod_unique"]))
            if "cardinality_ratio" in feature_row:
                st.metric(
                    "Cardinality Ratio", f"{feature_row['cardinality_ratio']:.2f}"
                )

    # Feature distribution visualization
    if selected_feature in ref_data.columns and selected_feature in prod_inputs.columns:
        if pd.api.types.is_numeric_dtype(ref_data[selected_feature]):
            # Numerical feature
            fig = make_subplots(
                rows=1, cols=2, subplot_titles=("Reference", "Production")
            )

            fig.add_trace(
                go.Histogram(
                    x=ref_data[selected_feature].dropna(),
                    name="Reference",
                    nbinsx=30,
                    marker_color=CATPPUCCIN_COLORS["blue"],
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Histogram(
                    x=prod_inputs[selected_feature].dropna(),
                    name="Production",
                    nbinsx=30,
                    marker_color=CATPPUCCIN_COLORS["green"],
                ),
                row=1,
                col=2,
            )

            fig.update_layout(
                title=f"{selected_feature} Distribution Comparison",
                showlegend=False,
                height=400,
            )
            fig = apply_catppuccin_theme(fig)
            st.plotly_chart(fig, width="stretch")
        else:
            # Categorical feature
            ref_counts = ref_data[selected_feature].value_counts().head(10)
            prod_counts = prod_inputs[selected_feature].value_counts().head(10)

            all_values = ref_counts.index.union(prod_counts.index)
            ref_aligned = ref_counts.reindex(all_values, fill_value=0)
            prod_aligned = prod_counts.reindex(all_values, fill_value=0)

            df_cat = pd.DataFrame(
                {
                    "Reference": ref_aligned,
                    "Production": prod_aligned,
                }
            )

            fig = px.bar(
                df_cat,
                barmode="group",
                title=f"{selected_feature} Value Counts",
                color_discrete_sequence=[
                    CATPPUCCIN_COLORS["blue"],
                    CATPPUCCIN_COLORS["green"],
                ],
            )
            fig = apply_catppuccin_theme(fig)
            st.plotly_chart(fig, width="stretch")


def show_performance_metrics(model_metrics_df, ref_data, prod_preds):
    """Performance metrics dashboard."""
    st.header("📈 Performance Metrics")
    st.markdown("**Model Evaluation & Continuous Monitoring**")

    if model_metrics_df is None or model_metrics_df.empty:
        st.warning("No model performance metrics available from MLflow.")
        return

    # Filter valid metrics
    valid_metrics = model_metrics_df.dropna(subset=["accuracy", "macro_f1"])

    if valid_metrics.empty:
        st.warning("No valid performance metrics found.")
        return

    # Latest metrics
    st.subheader("Latest Model Performance")

    latest = valid_metrics.iloc[0]

    # Overall metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Accuracy",
            f"{latest['accuracy']:.4f}" if pd.notna(latest.get("accuracy")) else "N/A",
        )
    with col2:
        st.metric(
            "Macro Precision",
            (
                f"{latest['macro_precision']:.4f}"
                if pd.notna(latest.get("macro_precision"))
                else "N/A"
            ),
        )
    with col3:
        st.metric(
            "Macro Recall",
            (
                f"{latest['macro_recall']:.4f}"
                if pd.notna(latest.get("macro_recall"))
                else "N/A"
            ),
        )
    with col4:
        st.metric(
            "Macro F1",
            f"{latest['macro_f1']:.4f}" if pd.notna(latest.get("macro_f1")) else "N/A",
        )
    with col5:
        if pd.notna(latest.get("roc_auc_macro")):
            st.metric("ROC-AUC (macro)", f"{latest['roc_auc_macro']:.4f}")
        else:
            st.metric("Model Type", latest.get("model", "Unknown"))

    # Weighted metrics
    st.subheader("Weighted Metrics (Account for Class Imbalance)")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Weighted Precision",
            (
                f"{latest['weighted_precision']:.4f}"
                if pd.notna(latest.get("weighted_precision"))
                else "N/A"
            ),
        )
    with col2:
        st.metric(
            "Weighted Recall",
            (
                f"{latest['weighted_recall']:.4f}"
                if pd.notna(latest.get("weighted_recall"))
                else "N/A"
            ),
        )
    with col3:
        st.metric(
            "Weighted F1",
            (
                f"{latest['weighted_f1']:.4f}"
                if pd.notna(latest.get("weighted_f1"))
                else "N/A"
            ),
        )

    # Per-class metrics
    st.subheader("Per-Class Metrics")
    class_metrics_data = []
    for class_name in ["low", "medium", "high"]:
        if pd.notna(latest.get(f"precision_{class_name}")):
            class_metrics_data.append(
                {
                    "Class": class_name.title(),
                    "Precision": latest[f"precision_{class_name}"],
                    "Recall": latest[f"recall_{class_name}"],
                    "F1": latest[f"f1_{class_name}"],
                }
            )

    if class_metrics_data:
        class_metrics_df = pd.DataFrame(class_metrics_data)
        st.dataframe(class_metrics_df, width="stretch", hide_index=True)

    # Model info
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model Type", latest.get("model", "Unknown"))
    with col2:
        if latest.get("start_time"):
            st.metric(
                "Last Training",
                (
                    latest["start_time"].strftime("%Y-%m-%d")
                    if hasattr(latest["start_time"], "strftime")
                    else "N/A"
                ),
            )

    st.divider()

    # Performance trends
    st.subheader("Performance Trends")

    fig = go.Figure()

    # Add all available metrics to the plot
    metrics_to_plot = [
        ("accuracy", CATPPUCCIN_COLORS["blue"]),
        ("macro_precision", CATPPUCCIN_COLORS["green"]),
        ("macro_recall", CATPPUCCIN_COLORS["yellow"]),
        ("macro_f1", CATPPUCCIN_COLORS["mauve"]),
    ]

    for metric_name, color in metrics_to_plot:
        if metric_name in valid_metrics.columns:
            metric_data = valid_metrics[metric_name].dropna()
            if not metric_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=metric_data.index,
                        y=metric_data.values,
                        name=metric_name.replace("_", " ").title(),
                        mode="lines+markers",
                        line={"color": color, "width": 2},
                        marker={"color": color},
                    )
                )

    fig.update_layout(
        title="Model Performance Over Training Runs",
        xaxis_title="Run Index",
        yaxis_title="Score",
        height=400,
        hovermode="x unified",
    )
    fig = apply_catppuccin_theme(fig)
    st.plotly_chart(fig, width="stretch")

    # Performance comparison table
    st.subheader("All Training Runs - Comprehensive Metrics")
    display_cols = [
        "run_id",
        "model",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "roc_auc_macro",
    ]
    available_cols = [col for col in display_cols if col in valid_metrics.columns]

    # Format the dataframe for better display
    display_df = valid_metrics[available_cols].copy()
    for col in display_df.select_dtypes(include=[float]).columns:
        display_df[col] = display_df[col].apply(
            lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
        )

    st.dataframe(display_df, width="stretch")

    # Production prediction statistics
    st.divider()
    st.subheader("Production Prediction Statistics")

    if "price_category" in ref_data.columns:
        # Calculate accuracy if we have ground truth (this is a simplified example)
        st.info(
            "💡 For continuous evaluation, ground truth labels are needed to calculate production accuracy."
        )

        # Show prediction distribution
        pred_counts = prod_preds["prediction"].value_counts().sort_index()
        categories = {0: "Low", 1: "Medium", 2: "High"}

        col1, col2 = st.columns(2)

        with col1:
            fig = px.pie(
                values=pred_counts.values,
                names=[categories.get(int(k), str(k)) for k in pred_counts.index],
                title="Production Prediction Distribution",
                color_discrete_sequence=[
                    CATPPUCCIN_COLORS["blue"],
                    CATPPUCCIN_COLORS["green"],
                    CATPPUCCIN_COLORS["yellow"],
                ],
            )
            fig = apply_catppuccin_theme(fig)
            st.plotly_chart(fig, width="stretch")

        with col2:
            # Prediction statistics
            st.metric("Total Predictions", len(prod_preds))
            st.metric("Unique Predictions", prod_preds["prediction"].nunique())
            if "timestamp" in prod_preds.columns:
                time_span = (
                    prod_preds["timestamp"].max() - prod_preds["timestamp"].min()
                )
                st.metric(
                    "Time Span",
                    f"{time_span.days} days" if hasattr(time_span, "days") else "N/A",
                )


def show_production_data(prod_inputs, prod_preds):
    """Production data exploration dashboard."""
    st.header("📦 Production Data")
    st.markdown("**Live Production Inputs & Predictions**")

    # Data summary
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Inputs", len(prod_inputs))
    with col2:
        st.metric("Total Predictions", len(prod_preds))
    with col3:
        if "timestamp" in prod_inputs.columns:
            latest = prod_inputs["timestamp"].max()
            st.metric(
                "Latest Input",
                (
                    latest.strftime("%Y-%m-%d %H:%M")
                    if hasattr(latest, "strftime")
                    else "N/A"
                ),
            )

    st.divider()

    # Data preview
    st.subheader("Recent Production Inputs")

    if "timestamp" in prod_inputs.columns:
        recent_inputs = prod_inputs.sort_values("timestamp", ascending=False).head(100)
        st.dataframe(recent_inputs, width="stretch")
    else:
        st.dataframe(prod_inputs.head(100), width="stretch")

    st.divider()

    # Predictions preview
    st.subheader("Recent Predictions")

    if "timestamp" in prod_preds.columns:
        recent_preds = prod_preds.sort_values("timestamp", ascending=False).head(100)
        st.dataframe(recent_preds, width="stretch")
    else:
        st.dataframe(prod_preds.head(100), width="stretch")

    # Feature distributions
    st.divider()
    st.subheader("Feature Distributions")

    feature_to_plot = st.selectbox(
        "Select Feature to Visualize", prod_inputs.columns.tolist()
    )

    if pd.api.types.is_numeric_dtype(prod_inputs[feature_to_plot]):
        fig = px.histogram(
            prod_inputs,
            x=feature_to_plot,
            title=f"{feature_to_plot} Distribution",
            nbins=30,
            color_discrete_sequence=[CATPPUCCIN_COLORS["blue"]],
        )
        fig = apply_catppuccin_theme(fig)
        st.plotly_chart(fig, width="stretch")
    else:
        value_counts = prod_inputs[feature_to_plot].value_counts().head(20)
        fig = px.bar(
            x=value_counts.index,
            y=value_counts.values,
            title=f"{feature_to_plot} Value Counts",
            labels={"x": feature_to_plot, "y": "Count"},
            color_discrete_sequence=[CATPPUCCIN_COLORS["blue"]],
        )
        fig = apply_catppuccin_theme(fig)
        st.plotly_chart(fig, width="stretch")


def show_alerts(ref_data, prod_inputs, prod_preds, mlflow_metrics):
    """Alerts and warnings dashboard."""
    st.header("🚨 Alerts & Warnings")
    st.markdown("**System Health & Drift Alerts**")

    alerts = []
    warnings = []

    # Check drift metrics
    if mlflow_metrics:
        metrics = mlflow_metrics

        # District PSI
        psi = safe_get_metric(metrics, "district_psi", 0.0)
        psi = float(psi)
        if psi > PSI_THRESHOLD:
            alerts.append(
                {
                    "type": "🔴 Critical",
                    "metric": "District PSI",
                    "value": f"{psi:.4f}",
                    "threshold": PSI_THRESHOLD,
                    "message": f"District distribution drift detected (PSI: {psi:.4f} > {PSI_THRESHOLD})",
                }
            )
        elif psi > PSI_THRESHOLD * 0.8:
            warnings.append(
                {
                    "type": "🟡 Warning",
                    "metric": "District PSI",
                    "value": f"{psi:.4f}",
                    "threshold": PSI_THRESHOLD,
                    "message": f"District PSI approaching threshold ({psi:.4f} / {PSI_THRESHOLD})",
                }
            )

        # Cardinality
        card_ratio = safe_get_metric(metrics, "address_cardinality_ratio", 0.0)
        card_ratio = float(card_ratio)
        if card_ratio > CARDINALITY_THRESHOLD:
            alerts.append(
                {
                    "type": "🔴 Critical",
                    "metric": "Address Cardinality",
                    "value": f"{card_ratio:.2f}",
                    "threshold": CARDINALITY_THRESHOLD,
                    "message": f"High cardinality growth detected (ratio: {card_ratio:.2f} > {CARDINALITY_THRESHOLD})",
                }
            )

        # Prediction drift
        pred_drift = safe_get_metric(metrics, "prediction_kl_drift", 0.0)
        pred_drift = float(pred_drift)
        if pred_drift > PRED_DRIFT_THRESHOLD:
            alerts.append(
                {
                    "type": "🔴 Critical",
                    "metric": "Prediction Drift",
                    "value": f"{pred_drift:.4f}",
                    "threshold": PRED_DRIFT_THRESHOLD,
                    "message": f"Prediction distribution drift detected (KL: {pred_drift:.4f} > {PRED_DRIFT_THRESHOLD})",
                }
            )

    # Feature statistics alerts
    feature_stats = calculate_feature_statistics(ref_data, prod_inputs)
    if not feature_stats.empty:
        drifted_features = feature_stats[feature_stats.get("drift_detected", False)]
        if not drifted_features.empty:
            for _, row in drifted_features.iterrows():
                alerts.append(
                    {
                        "type": "🟠 Alert",
                        "metric": f"Feature: {row['feature']}",
                        "value": (
                            f"PSI: {row.get('psi', 'N/A'):.4f}"
                            if "psi" in row
                            else "N/A"
                        ),
                        "threshold": "N/A",
                        "message": f"Drift detected in feature '{row['feature']}'",
                    }
                )

    # Display alerts
    if alerts:
        st.subheader("🔴 Active Alerts")
        for alert in alerts:
            with st.expander(f"{alert['type']} - {alert['metric']}", expanded=True):
                st.write(f"**Message:** {alert['message']}")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Current Value:** {alert['value']}")
                with col2:
                    st.write(f"**Threshold:** {alert['threshold']}")
    else:
        st.success("✅ No active alerts. System is healthy.")

    # Display warnings
    if warnings:
        st.subheader("🟡 Warnings")
        for warning in warnings:
            with st.expander(f"{warning['type']} - {warning['metric']}"):
                st.write(f"**Message:** {warning['message']}")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Current Value:** {warning['value']}")
                with col2:
                    st.write(f"**Threshold:** {warning['threshold']}")

    # System status
    st.divider()
    st.subheader("System Status")

    status_items = [
        ("Reference Data", "✅ Available" if ref_data is not None else "❌ Missing"),
        (
            "Production Inputs",
            "✅ Available" if prod_inputs is not None else "❌ Missing",
        ),
        (
            "Production Predictions",
            "✅ Available" if prod_preds is not None else "❌ Missing",
        ),
        ("MLflow Connection", "✅ Connected" if mlflow_metrics else "❌ Disconnected"),
    ]

    for item, status in status_items:
        st.write(f"- **{item}:** {status}")

    # Recommendations
    if alerts:
        st.divider()
        st.subheader("💡 Recommendations")
        st.info(
            """
        **Actions to consider:**
        1. Review the drifted features and investigate root causes
        2. Consider retraining the model with updated data
        3. Check for data pipeline issues or external factors
        4. Evaluate model performance on recent data
        5. Consider implementing algorithmic fallback if drift persists
        """
        )


if __name__ == "__main__":
    main()
