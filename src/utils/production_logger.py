import pandas as pd
from pathlib import Path
from datetime import datetime

LOG_DIR = Path("data/production")
LOG_DIR.mkdir(parents=True, exist_ok=True)

INPUT_LOG = LOG_DIR / "inputs.parquet"
PRED_LOG = LOG_DIR / "predictions.parquet"


def log_input(data: dict):
    df = pd.DataFrame([data])
    df["timestamp"] = datetime.utcnow()

    if INPUT_LOG.exists():
        df_old = pd.read_parquet(INPUT_LOG)
        df = pd.concat([df_old, df], ignore_index=True)

    df.to_parquet(INPUT_LOG)


def log_prediction(prediction: int):
    df = pd.DataFrame(
        [{"prediction": prediction, "timestamp": datetime.utcnow()}]
    )

    if PRED_LOG.exists():
        df_old = pd.read_parquet(PRED_LOG)
        df = pd.concat([df_old, df], ignore_index=True)

    df.to_parquet(PRED_LOG)

