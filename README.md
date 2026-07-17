# Phishing URL Detector

A machine learning system to classify URLs as phishing or legitimate.

This project implements a complete pipeline including feature extraction, dataset collection, model training/evaluation, and a live CLI prediction tool.

## Key Features
- **Live Feature Extraction (`features.py`)**: Extracts 14 features from any URL string, including URL structure (length, dots, slashes), suspicious keywords, typosquatting Levenshtein distance, and real-time SSL certificate validity.
- **Balanced Datasets (`prepare_data.py`)**: Pulls active phishing URLs from **PhishTank** and legitimate domains from **Tranco**, generating realistic paths/subdomains to build a robust dataset.
- **Model Comparison (`train.py`)**: Evaluates Logistic Regression, Random Forest, and Gradient Boosting, focusing heavily on **Recall** for the phishing class.
- **CLI Prediction Tool (`predict.py`)**: Live URL classification showing phishing probability and detailed feature contributions.
- **Research Notebook (`phishing_detector.ipynb`)**: Complete walk-through of the ML process.

---

## How to Set Up and Run

### 1. Install Dependencies
This project uses the fast `uv` Python package manager. Set up the virtual environment and install all packages:
```bash
# Install dependencies
uv sync
```

### 2. Generate Dataset
Download datasets, extract features, and split the data into training and test sets:
```bash
uv run python prepare_data.py --sample-size 1000
```

### 3. Train Models
Train Logistic Regression, Random Forest, and Gradient Boosting, optimize the threshold, and save the best model:
```bash
uv run python train.py
```

### 4. Run Classification
Classify any URL and see a breakdown of the top contributing features:
```bash
uv run python predict.py "https://www.google.com"
uv run python predict.py "http://verify-secure-paypal.update-account.com/login"
```
