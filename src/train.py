# src/train.py
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# Assuming utils.py contains the load_data function
from utils import load_data

def train_model():
    """
    Trains a Linear Regression model on the California Housing dataset.
    """
    # Create artifacts directory if it doesn't exist
    os.makedirs("artifacts", exist_ok=True)

    # Load and split data using the utility function
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions and evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print(f"Model: {type(model).__name__}")
    print(f"R2 Score: {r2:.4f}")
    print(f"Loss (MSE): {mse:.4f}")

    # Save the trained model
    model_path = "artifacts/model.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    # Save test data for prediction verification
    joblib.dump((X_test, y_test), "artifacts/test_data.joblib")
    print("Test data saved for prediction.")

if __name__ == "__main__":
    train_model()