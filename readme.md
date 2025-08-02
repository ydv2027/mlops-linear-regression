## Swatantra Yadav (G24AI1065)
https://github.com/ydv2027/mlops-linear-regression

Main Branch

##  workflow CI/CD Pipeline Workflow

The pipeline consists of three sequential jobs that ensure a robust validation and build process.

1.  **`test_suite`** 🧪
    * **Description**: Runs unit tests on the codebase using `pytest`.
    * **Trigger**: Must pass before any other job can start.

2.  **`train_and_quantize`** 🧠
    * **Description**: Executes the model training and quantization scripts. The resulting model artifacts are saved and uploaded.
    * **Depends On**: `test_suite`

3.  **`build_and_test_container`** 🐳
    * **Description**: Builds a Docker image, downloads the artifacts from the previous job, and runs the container to verify that the `predict.py` script executes successfully.
    * **Depends On**: `train_and_quantize`

---

## 📊 Model Performance & Results

The following table compares the performance of the original floating-point model with the manually quantized 8-bit integer model.

| Metric | Original Model (FP32) | Quantized Model (INT8 → FP32) |
| :--- | :---: | :---: |
| **R² Score** | `0.5758` | `-50330.6845` |
| **MSE Loss** | `0.5559` | `65955.0913` |

### Analysis of Quantization

As shown in the table, the **manual 8-bit quantization resulted in a catastrophic drop in performance**. The R² score plummeted, indicating that the de-quantized model's predictions are no better than random chance. This highlights a critical concept in model optimization: naive quantization without careful handling of scaling and zero-point can lead to significant precision loss, rendering the model unusable. This experiment successfully demonstrates the pipeline's mechanics but also serves as a cautionary example of the challenges in model compression.

---

## 🛠️ How to Run Locally

To run this project on your local machine, follow these steps.


├── .github/
│   └── workflows/
│       └── ci.yml
├── src/
│   ├── __init__.py
│   ├── train.py
│   ├── quantize.py
│   ├── predict.py
│   └── utils.py
├── tests/
│   ├── __init__.py
│   └── test_train.py
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/ydv2027/mlops-linear-regression.git](https://github.com/ydv2027/mlops-linear-regression.git)
    cd mlops-linear-regression
    ```

2.  **Set up a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the pipeline scripts:**
    ```bash
    # Train the model
    python src/train.py

    # Quantize the trained model
    python src/quantize.py

    # Run a sample prediction
    python src/predict.py
    ```
