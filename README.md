## TODO:
### 1. Finish the monitoring + Fix the plots  
### 2. Checkpoints  
### 3. Docker  
### 4. ~~Unit Tests~~  
### 5. Code Inspection and quality  
### 6. CI/CD  
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
