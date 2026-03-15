# Explainable Fraud Detection Dashboard

An interactive dashboard for detecting and explaining credit card fraud predictions using machine learning and SHAP-based interpretability.

## Overview

This project builds a fraud detection model on real-world transaction data and provides transparent explanations for predictions, allowing users to understand **why** a transaction is flagged as fraudulent.

The system combines:
- **Machine Learning Classification** (Random Forest)
- **Explainable AI** (SHAP TreeExplainer)
- **Flask Web Application** for an interactive dashboard

## Dataset

**Credit Card Fraud Detection Dataset**
- Source: [Kaggle – MLG-ULB](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 284,807 transactions | 492 fraud cases (0.17%)
- 30 features: V1–V28 (PCA), Time, Amount
- Target: `Class` (1 = Fraud, 0 = Normal)

## Project Structure

```
explainable-fraud-dashboard/
├── data/
│   └── creditcard.csv          # Dataset
├── models/
│   ├── trained_model.pkl       # Saved Random Forest model
│   ├── scaler.pkl              # StandardScaler for Amount/Time
│   ├── feature_names.pkl       # Feature name list
│   ├── X_test.pkl              # Test features
│   ├── y_test.pkl              # Test labels
│   └── metrics.json            # Evaluation metrics
├── src/
│   ├── preprocessing.py        # Data loading & preprocessing
│   ├── train_model.py          # Model training & evaluation
│   └── shap_explainer.py       # SHAP explanation generator
├── templates/
│   ├── index.html              # Dashboard home page
│   └── result.html             # Prediction result page
├── static/
│   ├── style.css               # Dashboard styling
│   ├── confusion_matrix.png    # Generated confusion matrix
│   ├── shap_global.png         # Global SHAP summary plot
│   └── shap_local_*.png        # Per-transaction SHAP plots
├── app.py                      # Flask web application
├── requirements.txt
└── README.md
```

## Setup & Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd explainable-fraud-dashboard

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place the dataset
# Download creditcard.csv from Kaggle and place it in data/
```

## Usage

### Step 1: Train the Model

```bash
python src/train_model.py
```

This will:
- Load and preprocess the dataset
- Train a Random Forest classifier with balanced class weights
- Print evaluation metrics (Precision, Recall, F1, ROC-AUC)
- Save model artifacts to `models/`
- Save confusion matrix plot to `static/`

### Step 2: Generate SHAP Explanations

```bash
python src/shap_explainer.py
```

This generates the global SHAP feature importance plot saved to `static/shap_global.png`.

### Step 3: Run the Dashboard

```bash
python app.py
```

Open `http://localhost:5000` in your browser to:
- View model evaluation metrics
- See global feature importance (SHAP)
- Analyze individual transactions with local SHAP explanations
- Try random fraud transactions

## Model Evaluation Metrics

Because fraud datasets are highly imbalanced, accuracy alone is insufficient. The model is evaluated using:

| Metric | Description |
|--------|-------------|
| Precision | Of all predicted frauds, how many are actual frauds |
| Recall | Of all actual frauds, how many were detected |
| F1-Score | Harmonic mean of Precision and Recall |
| ROC-AUC | Area under the ROC curve |

## Technologies Used

| Component | Technology |
|-----------|------------|
| Language | Python |
| ML Framework | scikit-learn |
| Explainability | SHAP |
| Web Framework | Flask |
| Visualization | Matplotlib, SHAP plots |
| Data Handling | Pandas, NumPy |

## Key Features

- **Fraud prediction** using Random Forest with balanced class weights
- **Class imbalance handling** via stratified splitting and balanced weighting
- **Global explanations** — which features matter most across all transactions
- **Local explanations** — why a specific transaction was flagged
- **Interactive dashboard** for exploring predictions with SHAP visualizations
- **Real-time analysis** — enter any test transaction index for instant explanation
