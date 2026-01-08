import numpy as np
from scipy.sparse import csr_matrix

class DummyModel:
    def predict(self, X):
        return np.array([1])

def test_model_can_make_prediction():
    X_test = csr_matrix([[0, 1, 2]])

    model = DummyModel()
    prediction = model.predict(X_test)

    assert prediction.shape == (1,)
    assert prediction[0] in [0, 1, 2]
