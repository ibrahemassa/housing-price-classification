import joblib
import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix

def test_model_can_make_prediction():
    processed_dir = Path("data/processed")
    model_dir = Path("models")

    X_test, _ = joblib.load(processed_dir / "test.pkl")
    model = joblib.load(model_dir / "model.pkl")

    if not hasattr(X_test, "__getitem__"):
        X_test = csr_matrix(X_test)

    X_sample = X_test[0:1]

    prediction = model.predict(X_sample)

    assert prediction is not None
    assert prediction.shape == (1,)
    assert prediction[0] in [0, 1, 2]
