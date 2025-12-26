# Integration Tests

This directory contains integration tests for the MLOps housing price classification project. Integration tests verify that multiple components work together correctly.

## Test Files

### 1. `test_training_pipeline.py`
Tests the complete training pipeline workflow:
- Data preprocessing and artifact creation
- Model training and saving
- Model registration with MLflow
- Pipeline component integration

### 2. `test_api_integration.py`
Tests the API endpoint integration:
- Complete prediction flow (input → preprocessing → prediction → logging)
- Health check endpoint
- Model loading fallback mechanism
- Multiple prediction consistency

### 3. `test_monitoring_integration.py`
Tests the monitoring workflow integration:
- Drift detection with real data structures
- Monitoring flow component integration
- Plotting functions integration
- Drift threshold validation

### 4. `test_mlops_pipeline.py`
End-to-end integration tests for the complete MLOps pipeline:
- Data flow from preprocessing to training
- Model lifecycle (train → register → load)
- Production logging integration
- API to logging integration
- Monitoring to retraining integration
- Complete workflow from start to finish

## Running Integration Tests

### Run all integration tests:
```bash
pytest integration_tests/ -v
```

### Run a specific integration test file:
```bash
pytest integration_tests/test_api_integration.py -v
```

### Run with coverage:
```bash
pytest integration_tests/ --cov=src --cov-report=html
```

## Test Structure

Integration tests use:
- **Temporary directories** for isolated test environments
- **Mocking** for external dependencies (MLflow, file I/O)
- **Fixtures** for shared test data and setup
- **Real component interactions** where possible

## Differences from Unit Tests

- **Unit Tests** (`tests/`): Test individual functions in isolation
- **Integration Tests** (`integration_tests/`): Test multiple components working together

## Prerequisites

- All dependencies from `requirements.txt` installed
- Virtual environment activated
- MLflow tracking URI configured (uses SQLite by default)

## Notes

- Integration tests may take longer to run than unit tests
- Some tests require mocked external services (MLflow, file system)
- Tests are designed to be independent and can run in any order

