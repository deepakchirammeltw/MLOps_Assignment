import joblib
import pytest
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
data = load_iris()
X, y = data.data, data.target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

@pytest.fixture
def trained_model():
    """Load the trained model from model2.joblib"""
    return joblib.load("model2.joblib")

def test_model_accuracy(trained_model):
    """Ensure the model achieves an acceptable accuracy"""
    y_pred = trained_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy > 0.8, f"Model accuracy is too low: {accuracy:.2f}"

def test_model_type(trained_model):
    """Ensure the loaded model is a RandomForestClassifier"""
    assert isinstance(trained_model, RandomForestClassifier), "Model is not a RandomForestClassifier"

def test_model_predictions(trained_model):
    """Ensure model produces valid class predictions"""
    y_pred = trained_model.predict(X_test)
    assert all(pred in [0, 1, 2] for pred in y_pred), "Model predicted an invalid class label"
