"""
app.py
------
Flask web application for the Explainable Fraud Detection Dashboard.
Serves model predictions with SHAP-based explanations.
"""

import os
import json
import uuid
import joblib
import numpy as np
from flask import Flask, render_template, request, redirect, url_for

from src.shap_explainer import build_explainer, local_explanation

# ----------------------------------------------
# App setup
# ----------------------------------------------
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# ──────────────────────────────────────────────
# Load artefacts once at startup
# ──────────────────────────────────────────────
model = joblib.load(os.path.join(MODEL_DIR, "trained_model.pkl"))
X_test = joblib.load(os.path.join(MODEL_DIR, "X_test_sample.pkl"))
y_test = joblib.load(os.path.join(MODEL_DIR, "y_test_sample.pkl"))
feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))

with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
    metrics = json.load(f)

# Build SHAP explainer (background sample keeps it fast)
explainer = build_explainer(model, X_test)

print(f"[INFO] Dashboard ready  -  {X_test.shape[0]:,} test transactions available")


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────
@app.route("/")
def index():
    """Dashboard home page."""
    return render_template(
        "index.html",
        metrics=metrics,
        total_test=X_test.shape[0],
    )


@app.route("/predict", methods=["POST"])
def predict():
    """Run prediction + SHAP explanation for a chosen transaction."""
    try:
        idx = int(request.form.get("tx_index", 0))
        if idx < 0 or idx >= len(X_test):
            return render_template(
                "index.html",
                metrics=metrics,
                total_test=X_test.shape[0],
                error=f"Index must be between 0 and {len(X_test) - 1}",
            )
    except ValueError:
        return render_template(
            "index.html",
            metrics=metrics,
            total_test=X_test.shape[0],
            error="Please enter a valid integer index.",
        )

    # Prediction
    instance = X_test[idx]
    proba = model.predict_proba(instance.reshape(1, -1))[0]
    pred_label = int(np.argmax(proba))
    pred_class = "Fraud" if pred_label == 1 else "Normal"
    confidence = round(float(proba[pred_label]) * 100, 2)
    actual_label = "Fraud" if int(y_test[idx]) == 1 else "Normal"

    # SHAP local explanation
    plot_filename = f"shap_local_{uuid.uuid4().hex[:8]}.png"
    plot_path = os.path.join(STATIC_DIR, plot_filename)
    _, top_features, _ = local_explanation(
        explainer, instance, feature_names, save_path=plot_path
    )

    return render_template(
        "result.html",
        tx_index=idx,
        pred_class=pred_class,
        confidence=confidence,
        fraud_prob=round(float(proba[1]) * 100, 2),
        normal_prob=round(float(proba[0]) * 100, 2),
        actual_label=actual_label,
        top_features=top_features,
        shap_plot=plot_filename,
    )


@app.route("/random_fraud")
def random_fraud():
    """Pick a random fraud transaction and redirect to prediction."""
    fraud_indices = np.where(y_test == 1)[0]
    if len(fraud_indices) == 0:
        return redirect(url_for("index", prefill=0))
    chosen = int(np.random.choice(fraud_indices))
    # Simulate POST by going through predict logic directly
    return redirect(url_for("index", prefill=chosen))


# ──────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)
