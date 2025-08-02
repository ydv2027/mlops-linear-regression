# src/quantize.py
import joblib
import numpy as np

def quantize_params(value, bits=8):
    """
    Performs affine quantization on a numpy array.
    Formula: R = S * (Q - Z)
    where R is real value, S is scale, Q is quantized value, Z is zero-point.
    """
    # Calculate scale and zero-point
    min_val, max_val = np.min(value), np.max(value)
    q_min, q_max = 0, 2**bits - 1
    
    # Handle the case where max_val equals min_val
    if max_val == min_val:
        scale = 1.0 
        zero_point = 0
    else:
        scale = (max_val - min_val) / (q_max - q_min)
        zero_point = q_min - min_val / scale

    # Quantize
    quantized_value = np.round(value / scale + zero_point).astype(np.uint8)
    
    return quantized_value, scale, zero_point

def dequantize_params(quantized_value, scale, zero_point):
    """De-quantizes a value using its scale and zero-point."""
    return scale * (quantized_value.astype(np.float32) - zero_point)

def quantize_model():
    """
    Loads a trained model, quantizes its parameters, and verifies performance.
    """
    # Load the trained model
    model = joblib.load("artifacts/model.joblib")
    
    # Extract coefficients and intercept
    coef = model.coef_
    intercept = model.intercept_
    
    # Save unquantized (raw) parameters
    joblib.dump({'coef': coef, 'intercept': intercept}, "artifacts/unquant_params.joblib")
    print("Unquantized parameters saved.")

    # Quantize parameters
    q_coef, scale_coef, zp_coef = quantize_params(coef)
    q_intercept, scale_intercept, zp_intercept = quantize_params(np.array([intercept])) # intercept must be an array
    
    # Save quantized parameters and metadata
    quantized_params = {
        'q_coef': q_coef,
        'scale_coef': scale_coef,
        'zp_coef': zp_coef,
        'q_intercept': q_intercept,
        'scale_intercept': scale_intercept,
        'zp_intercept': zp_intercept
    }
    joblib.dump(quantized_params, "artifacts/quant_params.joblib")
    print("Quantized parameters saved.")

    # --- Verification Step ---
    print("\n--- Verifying De-quantized Inference ---")
    # Load test data
    X_test, y_test = joblib.load("artifacts/test_data.joblib")
    
    # De-quantize parameters
    dq_coef = dequantize_params(q_coef, scale_coef, zp_coef)
    dq_intercept = dequantize_params(q_intercept, scale_intercept, zp_intercept)
    
    # Perform inference with de-quantized parameters
    y_pred_dequantized = np.dot(X_test, dq_coef) + dq_intercept[0]

    # Compare with original model's prediction
    y_pred_original = model.predict(X_test)

    # Calculate and print metrics
    from sklearn.metrics import r2_score, mean_squared_error
    r2_original = r2_score(y_test, y_pred_original)
    r2_dequantized = r2_score(y_test, y_pred_dequantized)
    mse_original = mean_squared_error(y_test, y_pred_original)
    mse_dequantized = mean_squared_error(y_test, y_pred_dequantized)

    print(f"Original R2: {r2_original:.4f}, De-quantized R2: {r2_dequantized:.4f}")
    print(f"Original MSE: {mse_original:.4f}, De-quantized MSE: {mse_dequantized:.4f}")

if __name__ == "__main__":
    quantize_model()