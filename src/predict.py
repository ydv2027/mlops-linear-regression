# src/predict.py
import joblib
import numpy as np

def run_prediction():
    """Loads model and test data to run a sample prediction."""
    try:
        # Load the trained model and test data
        model = joblib.load("artifacts/model.joblib")
        X_test, _ = joblib.load("artifacts/test_data.joblib")

        # Take a sample from the test set for prediction
        sample = X_test[:5]
        predictions = model.predict(sample)

        print("--- Model Prediction Verification ---")
        print("Sample Input Features:")
        print(sample)
        print("\nSample Predictions:")
        print(predictions)
        print("\nPrediction script executed successfully.")

    except FileNotFoundError:
        print("Error: Model or test data not found. Ensure 'train.py' has been run.")
        exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)

if __name__ == "__main__":
    run_prediction()