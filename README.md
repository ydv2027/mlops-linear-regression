# MLOps Linear Regression Pipeline

A complete MLOps pipeline for Linear Regression using the California Housing dataset with training, testing, quantization, Dockerization, and CI/CD automation.

🔗 **GitHub Repo:** https://github.com/ydv2027/mlops-linear-regression

## Table of Contents
- [Overview](#overview)
- [Repository Setup](#repository-setup)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Docker Usage](#docker-usage)
- [CI/CD Pipeline](#cicd-pipeline)
- [Testing](#testing)
- [Quantization Details](#quantization-details)

## Overview

This project implements a complete MLOps pipeline for Linear Regression using the California Housing dataset. The pipeline includes:

- **Model Training**: Linear Regression using scikit-learn
- **Testing**: Comprehensive unit tests with pytest
- **Quantization**: Manual 8-bit quantization for model compression
- **Containerization**: Docker support for deployment
- **CI/CD**: Automated GitHub Actions workflow
- **Monitoring**: Performance tracking and comparison

## Repository Setup

This repository was created entirely using the Windows PowerShell command line tool. Here are the exact steps followed:

```powershell
python -m venv mlops_env
mlops_env\Scripts\Activate.ps1


New-Item -ItemType Directory -Name "mlops-linear-regression"
Set-Location "mlops-linear-regression"

New-Item -ItemType Directory -Path "src", "tests", ".github\workflows", "models" -Force

New-Item -ItemType File -Name "README.md", ".gitignore", "requirements.txt"
New-Item -ItemType File -Path "src\__init__.py", "src\train.py", "src\quantize.py", "src\predict.py", "src\utils.py"
New-Item -ItemType File -Path "tests\__init__.py", "tests\test_train.py"
New-Item -ItemType File -Path ".github\workflows\ci.yml"
New-Item -ItemType File -Name "Dockerfile"

@"
scikit-learn==1.3.0
numpy==1.24.3
joblib==1.3.2
pytest==7.4.0
pandas==2.0.3
"@ | Out-File -FilePath "requirements.txt" -Encoding UTF8

pip install -r requirements.txt

git init
git add .
git commit -m "Initial commit: MLOps Linear Regression Pipeline setup"
git branch -M master

git remote add origin https://github.com/sourin00/mlops-linear-regression.git
git push -u origin master

mlops-linear-regression/
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
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md


Model Performance
Metric	Original Model	Quantized Model	Difference
R² Score	0.5758	0.5758	0.0000
MSE	0.5559	0.5559	0.0000
Max Prediction Error	-	0.000002	+0.000002
Mean Prediction Error	-	0.000002	+0.000002
Model Size	1.2 KB	0.3 KB	-0.9 KB

Swatantra Yadav
