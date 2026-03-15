"""
train_model.py
--------------
Trains a Random Forest classifier on the credit-card fraud dataset,
evaluates it, and saves the trained model + artefacts.
"""

import os
import sys
import json
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay,
)

# ── allow imports from project root ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocessing import load_data, dataset_summary, preprocess

# ----------------------------------------------
# Paths
# ----------------------------------------------
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODEL_DIR = os.path.join(BASE_DIR, "models")
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)


def train_model(X_train, y_train):
    """Train a Random Forest with balanced class weights."""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print("[INFO] Model training complete.")
    return model


def evaluate_model(model, X_test, y_test) -> dict:
    """Evaluate and print key metrics; return dict of metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
    }

    print("\n" + "=" * 50)
    print("  MODEL EVALUATION")
    print("=" * 50)
    print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))
    for k, v in metrics.items():
        print(f"  {k:>12}: {v}")
    print("=" * 50 + "\n")

    return metrics, y_pred


def save_confusion_matrix(y_test, y_pred):
    """Save confusion matrix plot to static/."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=["Normal", "Fraud"],
        cmap="Blues",
        ax=ax,
    )
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(STATIC_DIR, "confusion_matrix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Confusion matrix saved -> {path}")


def main():
    # 1. Load & preprocess
    df = load_data()
    summary = dataset_summary(df)
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess(df)

    # 2. Train
    model = train_model(X_train, y_train)

    # 3. Evaluate
    metrics, y_pred = evaluate_model(model, X_test, y_test)

    # 4. Save artefacts
    joblib.dump(model, os.path.join(MODEL_DIR, "trained_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(feature_names, os.path.join(MODEL_DIR, "feature_names.pkl"))
    joblib.dump(X_test, os.path.join(MODEL_DIR, "X_test.pkl"))
    joblib.dump(y_test, os.path.join(MODEL_DIR, "y_test.pkl"))
    print("[INFO] Model and artefacts saved to models/")

    # 5. Save metrics JSON (for Flask dashboard)
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump({**metrics, **summary}, f, indent=2)
    print("[INFO] Metrics saved -> models/metrics.json")

    # 6. Confusion matrix plot
    save_confusion_matrix(y_test, y_pred)

    print("[OK] Training pipeline finished.")


if __name__ == "__main__":
    main()
