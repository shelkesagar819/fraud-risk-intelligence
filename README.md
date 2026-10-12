# Real-Time Fraud Detection & Risk Intelligence Platform

A machine learning project designed to identify potentially fraudulent credit card transactions and provide risk insights through an interactive dashboard and REST API.

## Project Overview

This project uses transaction data and a machine learning model to classify transactions as potentially fraudulent or legitimate. It includes a Streamlit dashboard for exploring transaction data and a FastAPI service for predictions.

## Features

- Fraud detection using machine learning
- Interactive Streamlit dashboard
- Transaction analysis and filtering
- Model performance metrics and confusion matrix
- REST API for fraud predictions
- Risk classification based on predicted probability

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Plotly
- FastAPI
- Uvicorn
- Joblib

## Project Structure

```text
fraud-risk-intelligence/
├── artifacts/
│   └── metrics.json
├── dashboard/
│   └── app.py
├── data/
│   └── raw/
├── src/
│   ├── api.py
│   └── train.py
├── tests/
├── .gitignore
└── requirements.txt
```

## Model Evaluation

The current saved evaluation results are:

| Metric | Score |
|---|---:|
| Precision | 0.061 |
| Recall | 0.918 |
| F1 Score | 0.114 |
| ROC-AUC | 0.972 |

These metrics are from the model's held-out evaluation. The low precision indicates that false-positive predictions remain an important limitation.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/shelkesagar819/fraud-risk-intelligence.git
cd fraud-risk-intelligence
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Prepare the dataset and model

Place the required credit card transaction dataset at:

`data/raw/creditcard.csv`

Train the model:

```bash
python -m src.train
```

### 5. Run the dashboard

```bash
streamlit run dashboard/app.py
```

### 6. Run the prediction API

Open a separate terminal and run:

```bash
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Open the interactive API documentation at:

`http://127.0.0.1:8000/docs`

## Important Notes

- The dataset and trained model files are not included in this repository by default.
- Model predictions are estimates and should not be treated as definitive proof of fraud.
- The evaluation results show a high recall but low precision, so further model tuning and threshold evaluation may be needed.

## Author

**Sagar Shelke**

GitHub: [@shelkesagar819](https://github.com/shelkesagar819)