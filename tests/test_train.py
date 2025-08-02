# tests/test_train.py
import pytest
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib

# Add src to path to allow imports
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from utils import load_data
from train import train_model

# A fixture to run training once before tests
@pytest.fixture(scope="module")
def trained_model_and_data():
    """Fixture to run the training script and load artifacts."""
    # Ensure artifacts directory is clean before test
    if os.path.exists("artifacts"):
        for f in os.listdir("artifacts"):
            os.remove(os.path.join("artifacts", f))
            
    train_model()
    model = joblib.load("artifacts/model.joblib")
    X_test, y_test = joblib.load("artifacts/test_data.joblib")
    return model, X_test, y_test

def test_load_data():
    """Unit test for dataset loading."""
    X, y = load_data()
    assert X is not None, "Data loading failed for features (X)."
    assert y is not None, "Data loading failed for target (y)."
    assert X.shape[0] == y.shape[0], "Mismatch in number of samples."

def test_model_creation(trained_model_and_data):
    """Validate that the model is a LinearRegression instance."""
    model, _, _ = trained_model_and_data
    assert isinstance(model, LinearRegression), "Model is not a LinearRegression instance."

def test_model_trained(trained_model_and_data):
    """Check if the model has been trained (coefficients exist)."""
    model, _, _ = trained_model_and_data
    assert hasattr(model, 'coef_'), "Model does not have coefficients, it may not be trained."
    assert model.coef_ is not None

def test_r2_score_threshold(trained_model_and_data):
    """Ensure R2 score exceeds a minimum threshold."""
    model, X_test, y_test = trained_model_and_data
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)
    # A reasonable threshold for this model
    assert score > 0.5, f"R2 score {score:.2f} is below the threshold of 0.5."