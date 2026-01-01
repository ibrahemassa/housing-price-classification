# MLOps Monitoring Dashboard

A comprehensive Streamlit-based dashboard for monitoring ML model performance, drift detection, and feature validation.

## Features

The monitoring dashboard provides the following capabilities:

### 📈 Overview Dashboard
- **Key Metrics**: Total predictions, active alerts, drift metrics, and model accuracy
- **Recent Predictions Timeline**: Time-series visualization of prediction volume
- **Prediction Distribution**: Pie chart showing distribution of price categories
- **Model Performance Trend**: Historical accuracy and F1 score trends

### 🔍 Drift Monitoring
- **PSI (Population Stability Index)**: Measures distribution shift for categorical features
- **Cardinality Growth**: Tracks high-cardinality feature growth (e.g., addresses)
- **Prediction Drift**: KL divergence between reference and production predictions
- **Feature Distribution Comparisons**: Visual comparisons of reference vs production data
- **Numerical Feature Drift**: PSI and statistical tests for numerical features

### 📊 Feature Statistics & Validation
- **Comprehensive Feature Analysis**: Statistics for all features (numerical and categorical)
- **Drift Detection**: Automatic detection of feature drift using PSI and KS tests
- **Feature Distribution Visualization**: Interactive plots for each feature
- **Statistical Validation**: Kolmogorov-Smirnov tests for numerical features

### 📈 Performance Metrics
- **Model Performance Tracking**: Accuracy, F1 score, and other metrics from MLflow
- **Performance Trends**: Historical performance visualization
- **Training Run Comparison**: Table of all training runs with metrics
- **Production Statistics**: Prediction distribution and statistics

### 📦 Production Data
- **Data Exploration**: Browse recent production inputs and predictions
- **Feature Distributions**: Interactive visualization of production feature distributions
- **Data Preview**: Recent production data tables

### 🚨 Alerts & Warnings
- **Active Alerts**: Critical alerts for drift detection
- **Warnings**: Early warning indicators approaching thresholds
- **System Status**: Health check of all system components
- **Recommendations**: Actionable recommendations when issues are detected

## Requirements

The dashboard requires:
- Python 3.13+
- Streamlit
- Plotly
- MLflow (for metrics)
- Production data (inputs.parquet and predictions.parquet)
- Reference data (reference.parquet)

## Running the Dashboard

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the dashboard
python run_dashboard.py

# Or directly with Streamlit
streamlit run src/monitoring/dashboard.py --server.port=8501
```

The dashboard will be available at `http://localhost:8501`

### Docker

```bash
# Build and run with docker-compose
docker-compose up dashboard

# Or build the image separately
docker build -f Dockerfile.dashboard -t mlops-dashboard .
docker run -p 8501:8501 \
  -v ./data:/app/data \
  -v ./mlflow.db:/app/mlflow.db \
  -v ./mlruns:/app/mlruns \
  -e MLFLOW_TRACKING_URI=http://mlflow-server:5000 \
  mlops-dashboard
```

### Environment Variables

- `MLFLOW_TRACKING_URI`: MLflow tracking server URI (default: `sqlite:///mlflow.db`)
- `DASHBOARD_PORT`: Port for the dashboard (default: 8501)

## Data Requirements

The dashboard expects the following data files:

1. **Reference Data**: `data/reference/reference.parquet`
   - Baseline dataset used for drift comparison
   - Should include all features and target variable

2. **Production Inputs**: `data/production/inputs.parquet`
   - Logged production input features
   - Should include timestamp column

3. **Production Predictions**: `data/production/predictions.parquet`
   - Logged production predictions
   - Should include timestamp column

## Dashboard Pages

### Overview
Main dashboard with key metrics and quick visualizations.

### Drift Monitoring
Detailed drift analysis with PSI metrics, distribution comparisons, and statistical tests.

### Feature Statistics
Comprehensive feature-level analysis with drift detection and validation.

### Performance Metrics
Model performance tracking from MLflow experiments.

### Production Data
Exploration of live production data and predictions.

### Alerts
System health, alerts, warnings, and recommendations.

## Integration with MLflow

The dashboard automatically fetches:
- Monitoring metrics (PSI, cardinality, prediction drift)
- Model performance metrics (accuracy, F1 score)
- Training run history
- Model registry information

Ensure MLflow is running and accessible at the configured `MLFLOW_TRACKING_URI`.

## Thresholds

Default drift detection thresholds:
- **PSI Threshold**: 0.25 (categorical features)
- **Cardinality Threshold**: 1.3 (high-cardinality features)
- **Prediction Drift Threshold**: 0.15 (KL divergence)

These can be adjusted in `src/monitoring/monitor_flow.py`.

## Troubleshooting

### Dashboard shows "No production data"
- Ensure the API is logging inputs and predictions
- Check that `data/production/inputs.parquet` and `data/production/predictions.parquet` exist
- Verify the production logger is working correctly

### MLflow metrics not loading
- Check MLflow server is running
- Verify `MLFLOW_TRACKING_URI` environment variable
- Ensure monitoring flow has been run at least once

### Missing reference data
- Ensure `data/reference/reference.parquet` exists
- This should be created during the training pipeline

## Architecture

The dashboard is built with:
- **Streamlit**: Web framework for the dashboard UI
- **Plotly**: Interactive visualizations
- **MLflow**: Metrics and model tracking
- **Pandas**: Data processing
- **Scipy**: Statistical tests (KS test, PSI calculation)

## Future Enhancements

Potential improvements:
- Real-time updates via WebSocket
- Custom alert rules configuration
- Export capabilities for reports
- Integration with notification systems
- Model performance comparison across versions
- A/B testing visualization




