import joblib
import pytest
from model import X_test, y_test

def test_model_accuracy():
    model = joblib.load("model.pkl")
    accuracy = model.score(X_test, y_test)
    assert accuracy > 0.8, "Model accuracy is too low!"
