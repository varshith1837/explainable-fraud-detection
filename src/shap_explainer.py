"""
shap_explainer.py
-----------------
Generates SHAP-based explanations for the trained fraud-detection model.
Provides global feature importance and local per-transaction explanations.
"""

import os
import sys
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

# ── allow imports from project root ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ----------------------------------------------
# Paths
# ----------------------------------------------
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODEL_DIR = os.path.join(BASE_DIR, "models")
STATIC_DIR = os.path.join(BASE_DIR, "static")


def load_artefacts():
    """Load the trained model, test data, and feature names."""
    model = joblib.load(os.path.join(MODEL_DIR, "trained_model.pkl"))
    X_test = joblib.load(os.path.join(MODEL_DIR, "X_test.pkl"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
    return model, X_test, feature_names


def build_explainer(model, X_background):
    """
    Create a SHAP TreeExplainer.
    Uses a small background sample to keep computation fast.
    """
    bg = shap.sample(X_background, min(100, len(X_background)))
    explainer = shap.TreeExplainer(model, data=bg)
    return explainer


def global_summary_plot(explainer, X_sample, feature_names):
    """Generate and save the global SHAP summary bar plot."""
    shap_values = explainer.shap_values(X_sample)

    # For binary classifier shap_values is a list [class-0, class-1]
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    fig, ax = plt.subplots(figsize=(8, 6))
    shap.summary_plot(
        sv, X_sample,
        feature_names=feature_names,
        plot_type="bar",
        max_display=15,
        show=False,
    )
    plt.title("Global Feature Importance (SHAP)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(STATIC_DIR, "shap_global.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"[INFO] Global SHAP plot saved -> {path}")
    return path


def local_explanation(explainer, X_instance, feature_names, save_path=None):
    """
    Generate a local SHAP waterfall plot for a single transaction.

    Returns
    -------
    shap_values_instance : array of SHAP values for class-1 (fraud)
    top_features : list of (feature_name, shap_value) sorted by abs contribution
    plot_path : path to the saved plot image
    """
    shap_values = explainer.shap_values(X_instance.reshape(1, -1))
    if isinstance(shap_values, list):
        sv = shap_values[1][0]
    elif len(shap_values.shape) == 3:
        sv = shap_values[0, :, 1]
    else:
        sv = shap_values[0]

    # Top contributing features
    abs_sv = np.abs(sv)
    sorted_idx = np.argsort(abs_sv)[::-1]
    top_features = [
        (str(feature_names[int(i)]), round(float(sv[int(i)]), 4)) 
        for i in sorted_idx[:10]
    ]

    # Waterfall-style horizontal bar plot
    top_n = 10
    indices = sorted_idx[:top_n][::-1]  # reverse for bottom-up bar chart
    names = [str(feature_names[int(i)]) for i in indices]
    values = [float(sv[int(i)]) for i in indices]
    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in values]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(names, values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("SHAP Value (impact on Fraud prediction)", fontsize=11)
    ax.set_title("Local Explanation — Top Feature Contributions", fontsize=13, fontweight="bold")
    ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(STATIC_DIR, "shap_local.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return sv, top_features, save_path


# ----------------------------------------------
# Standalone: generate global plot
# ----------------------------------------------
if __name__ == "__main__":
    model, X_test, feature_names = load_artefacts()
    explainer = build_explainer(model, X_test)

    # Global plot on a sample of test transactions
    sample = X_test[:500]
    global_summary_plot(explainer, sample, feature_names)

    # Demo local explanation on first test transaction
    sv, top_feats, path = local_explanation(explainer, X_test[0], feature_names)
    print("\nTop contributing features:")
    for name, val in top_feats:
        print(f"  {name:>20}  ->  {val:+.4f}")
    print(f"\n[OK] Local SHAP plot saved -> {path}")
