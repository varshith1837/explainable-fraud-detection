"""
preprocessing.py
-----------------
Loads and preprocesses the Credit Card Fraud Detection dataset.
Handles feature scaling and stratified train-test splitting.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "creditcard.csv")
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_data(filepath: str = DATA_PATH) -> pd.DataFrame:
    """Load the credit-card transaction CSV."""
    df = pd.read_csv(filepath)
    print(f"[INFO] Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


def dataset_summary(df: pd.DataFrame) -> dict:
    """Return quick stats about fraud vs normal transactions."""
    total = len(df)
    fraud = int(df["Class"].sum())
    normal = total - fraud
    ratio = fraud / total * 100
    summary = {
        "total_transactions": total,
        "fraud_cases": fraud,
        "normal_cases": normal,
        "fraud_ratio_pct": round(ratio, 4),
    }
    print(f"[INFO] Fraud: {fraud} / {total} ({ratio:.4f}%)")
    return summary


def preprocess(df: pd.DataFrame):
    """
    1. Scale 'Amount' and 'Time' with StandardScaler.
    2. Split into stratified train / test sets.

    Returns
    -------
    X_train, X_test, y_train, y_test, scaler, feature_names
    """
    scaler = StandardScaler()
    df["Amount"] = scaler.fit_transform(df[["Amount"]])
    df["Time"] = scaler.fit_transform(df[["Time"]])

    feature_names = [c for c in df.columns if c != "Class"]
    X = df[feature_names].values
    y = df["Class"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"[INFO] Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")
    return X_train, X_test, y_train, y_test, scaler, feature_names


# ──────────────────────────────────────────────
# Quick sanity-check when run directly
# ──────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data()
    dataset_summary(df)
    X_train, X_test, y_train, y_test, scaler, feat = preprocess(df)
    print(f"[INFO] Features: {len(feat)}")
    print("[OK] Preprocessing complete.")
