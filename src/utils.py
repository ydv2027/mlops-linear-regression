# src/utils.py
from sklearn.datasets import fetch_california_housing
import joblib

def load_data():
    """Loads the California Housing dataset."""
    housing = fetch_california_housing()
    return housing.data, housing.target

def save_model(model, filepath):
    """Saves a model to a file using joblib."""
    joblib.dump(model, filepath)

def load_model(filepath):
    """Loads a model from a file using joblib."""
    return joblib.load(filepath)