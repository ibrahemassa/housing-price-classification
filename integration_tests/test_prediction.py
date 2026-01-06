import joblib
import numpy as np
from pathlib import Path


def test_model_can_make_prediction():
    processed_dir = Path("data/processed")

    X_test, _ = joblib.load(processed_dir / "test.pkl")
    model = joblib.load(processed_dir / "model.pkl")

    X_sample = X_test[0]

    prediction = model.predict(X_sample)

    assert prediction is not None
    assert prediction.shape == (1,)
    assert prediction[0] in [0, 1, 2]

