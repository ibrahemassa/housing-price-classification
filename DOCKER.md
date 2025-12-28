# Docker Setup Guide

This project includes a complete Docker setup for the MLOps pipeline with training, API, monitoring, Prefect server, and MLflow server.

## Environment Configuration

All environment variables are configured in the `.env` file. Copy `.env.example` to `.env` and modify as needed:

```bash
cp .env.example .env
```

The `.env` file contains:
- `PYTHONPATH` - Python path configuration
- `PREFECT_API_URL` - Prefect server API URL
- `MLFLOW_TRACKING_URI` - MLflow server tracking URI
- `GIT_PYTHON_REFRESH` - Git warning suppression
- Port configurations for services

All services in `docker-compose.yml` use `env_file: - .env` to load these variables.

## Architecture

The Docker setup consists of 5 main services:

- **Prefect Server**: Flow orchestration and scheduling (port 4200)
- **MLflow Server**: Model tracking and registry UI (port 5000)
- **Training Container**: Runs the Prefect training pipeline (`src/utils/pipelines.py`)
- **Monitoring Container**: Runs the Prefect monitoring flow and can trigger retraining when drift is detected
- **API Container**: Serves the FastAPI application for predictions (port 8081)

## Quick Start

### Build and Run with Docker Compose

```bash
# Build and start all services
docker-compose up --build

# Run in detached mode
docker-compose up -d

# View logs for all services
docker-compose logs -f

# View logs for a specific service
docker-compose logs -f api
docker-compose logs -f train
docker-compose logs -f monitor
docker-compose logs -f prefect-server
docker-compose logs -f mlflow-server

# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

### Service URLs

Once all services are running:

- **Prefect UI**: http://localhost:4200
- **MLflow UI**: http://localhost:5000
- **API**: http://localhost:8081
- **API Health Check**: http://localhost:8081/health

## Services Overview

### Prefect Server

The Prefect server orchestrates and schedules flows:

- **Port**: 4200
- **UI**: http://localhost:4200
- **Purpose**: Manages training and monitoring flows
- **Database**: SQLite (stored in Docker volume `prefect-db`)

### MLflow Server

The MLflow server tracks experiments and manages model registry:

- **Port**: 5000
- **UI**: http://localhost:5000
- **Purpose**: Model versioning, experiment tracking, model registry
- **Backend**: SQLite database (`mlflow.db`)
- **Artifacts**: Stored in `mlruns/` directory

### Training Container

Runs the complete training pipeline:

1. **Preprocess**: `src/training/preprocess.py`
2. **Train**: `src/training/train.py`
3. **Register Model**: `src/training/register_model.py`

**To run training manually:**

```bash
# Run training pipeline
docker-compose run train python -m src.utils.pipelines

# Or execute directly in the running container
docker-compose exec train python -m src.utils.pipelines
```

### Monitoring Container

Runs the monitoring flow that:

- Monitors data drift (PSI, cardinality, prediction distribution)
- Generates drift plots
- **Automatically triggers retraining** when drift is detected (≥2 alerts)

The monitoring flow runs continuously and checks for drift. When drift is detected, it triggers the training pipeline automatically.

**To run monitoring manually:**

```bash
# Run monitoring flow
docker-compose run monitor python -m src.monitoring.monitor_flow

# Or execute directly in the running container
docker-compose exec monitor python -m src.monitoring.monitor_flow
```

**How Monitoring Triggers Training:**

When the monitoring flow detects drift (≥2 alerts from PSI, cardinality, or prediction drift), it automatically triggers the training pipeline by executing:

```python
subprocess.run([sys.executable, "src/utils/pipelines.py"], check=True)
```

Since both containers share the same codebase via mounted volumes, the monitoring container can directly execute the training pipeline script.

### API Container

Serves predictions via FastAPI:

- **Port**: 8081
- **Health Check**: `GET /health`
- **Prediction**: `POST /predict`

## Volumes

The containers use the following volume mounts:

- `./data` - Data directory (raw, processed, production, reference, monitoring plots)
- `./models` - Trained model files
- `./mlflow.db` - MLflow tracking database
- `./mlruns` - MLflow run artifacts
- `prefect-db` (Docker volume) - Prefect server database

## API Usage

### Health Check

```bash
curl http://localhost:8081/health
```

### Prediction Request

```bash
curl -X POST "http://localhost:8081/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "GrossSquareMeters": 100.0,
    "NetSquareMeters": 80.0,
    "BuildingAge": 10.0,
    "NumberOfRooms": 3.0,
    "district": "district1",
    "HeatingType": "central",
    "StructureType": "concrete",
    "FloorLocation": "middle",
    "address": "123 Main St",
    "AdCreationDate": "2024-01-01",
    "Subscription": "premium"
  }'
```

Expected response:

```json
{
  "price_category": "low"
}
```

## Environment Variables

### Training Container

- `PREFECT_API_URL`: http://prefect-server:4200/api
- `MLFLOW_TRACKING_URI`: http://mlflow-server:5000
- `PYTHONPATH`: /app

### Monitoring Container

- `PREFECT_API_URL`: http://prefect-server:4200/api
- `MLFLOW_TRACKING_URI`: http://mlflow-server:5000
- `PYTHONPATH`: /app

### API Container

- `MLFLOW_TRACKING_URI`: http://mlflow-server:5000
- `PYTHONPATH`: /app

## Workflow

### Complete MLOps Workflow

1. **Training**: Train container runs the training pipeline
   - Preprocesses data
   - Trains model
   - Registers model in MLflow

2. **API**: API container serves predictions
   - Loads model from MLflow registry
   - Logs inputs and predictions

3. **Monitoring**: Monitor container continuously monitors for drift
   - Checks PSI, cardinality, prediction distribution
   - Generates drift plots
   - **Triggers retraining** when drift detected

4. **Prefect**: Orchestrates and schedules flows
   - Can schedule monitoring to run periodically
   - Tracks flow runs and status

5. **MLflow**: Tracks experiments and manages models
   - Stores model versions
   - Tracks metrics and parameters
   - Manages model registry (staging/production)

### Monitoring → Training Trigger

When monitoring detects drift:

1. Monitoring flow detects ≥2 alerts (PSI > 0.25, cardinality ratio > 1.3, or prediction drift > 0.15)
2. Logs error: "Drift detected — triggering retraining pipeline"
3. Executes training pipeline: `python src/utils/pipelines.py`
4. Training pipeline runs: preprocess → train → register model
5. New model is registered in MLflow and promoted

## Individual Container Commands

### Build Individual Containers

```bash
# Build training container
docker build -f Dockerfile.train -t mlops-train .

# Build API container
docker build -f Dockerfile.api -t mlops-api .

# Build monitoring container
docker build -f Dockerfile.monitor -t mlops-monitor .
```

### Run Individual Containers

```bash
# Run training (requires Prefect and MLflow servers)
docker run --rm \
  --network mlops-project_mlops-network \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/mlflow.db:/app/mlflow.db \
  -v $(pwd)/mlruns:/app/mlruns \
  -e PREFECT_API_URL=http://prefect-server:4200/api \
  -e MLFLOW_TRACKING_URI=http://mlflow-server:5000 \
  mlops-train

# Run API (requires MLflow server)
docker run --rm \
  --network mlops-project_mlops-network \
  -p 8081:8081 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/mlflow.db:/app/mlflow.db \
  -v $(pwd)/mlruns:/app/mlruns \
  -e MLFLOW_TRACKING_URI=http://mlflow-server:5000 \
  mlops-api

# Run monitoring (requires Prefect and MLflow servers, and train container)
docker run --rm \
  --network mlops-project_mlops-network \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/mlflow.db:/app/mlflow.db \
  -v $(pwd)/mlruns:/app/mlruns \
  -e PREFECT_API_URL=http://prefect-server:4200/api \
  -e MLFLOW_TRACKING_URI=http://mlflow-server:5000 \
  mlops-monitor
```

## Troubleshooting

### Check Service Health

```bash
# Check all services are running
docker-compose ps

# Check service logs
docker-compose logs prefect-server
docker-compose logs mlflow-server
docker-compose logs train
docker-compose logs monitor
docker-compose logs api
```

### Common Issues

1. **Prefect server not accessible**: Ensure it's running and healthy before starting other services
2. **MLflow server not accessible**: Check port 5000 is not already in use
3. **Training fails**: Ensure `data/processed/train.pkl` and `test.pkl` exist
4. **API can't load model**: Check MLflow server is running and model is registered
5. **Monitoring can't trigger training**: Ensure both containers share the same network and volumes

### Reset Everything

```bash
# Stop and remove all containers, networks, and volumes
docker-compose down -v

# Remove images (optional)
docker-compose down --rmi all

# Clean rebuild
docker-compose build --no-cache
docker-compose up
```

## Notes

- Ensure `data/processed/` contains `train.pkl` and `test.pkl` before running training
- The API will fall back to staging model if production model is not available
- MLflow tracking uses the MLflow server (http://mlflow-server:5000) when available, falls back to local SQLite
- Models are saved to `models/model.pkl` and registered in MLflow
- Monitoring plots are saved to `data/monitoring/plots/`
- Prefect server database is stored in a Docker volume and persists across restarts
