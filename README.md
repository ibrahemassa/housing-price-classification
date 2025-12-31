## TODO:
### 2. Checkpoints  
### 3. FIX Docker  
### 6. CD  
### 7. (Maybe) Better Feature Selection  
### 8. (Maybe) Try Different Models

## Commands:
### 1. Run The API:
```bash
 uvicorn src.api:app --reload --port 8081 
```
### 2. Run Mlflow
```bash
mlflow server --host localhost --port 5000
```
### 3. Run Prefect server
```bash
prefect server start   
```
### 4. Run Main pipeline
```bash
python src/pipelines.py
```
### 5. Run Monitoring pipeline
```bash
python src/monitor_flow.py
```
### 6. Run Monitoring Dashboard
```bash
python run_dashboard.py
```
Or directly with Streamlit:
```bash
streamlit run src/monitoring/dashboard.py --server.port=8501
```

The dashboard will be available at `http://localhost:8501`
