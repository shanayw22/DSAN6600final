"""
v2_shap_analysis.py
=========================================================
Compute SHAP feature importance for trained TQE models.

Outputs:
    - shap_summary_plot.png
    - shap_bar_plot.png
    - shap_values.npy
    - shap_feature_importance.csv

Models supported:
    - mlp
    - residual_mlp
    - transformer
=========================================================
"""

import os
import json
import pickle
import argparse
import logging
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import shap
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

# ============================================================================
# Confugiration
# ============================================================================
OUTPUT_DIR = '../v2/V2_Shap_Result/'

# -----------------------------------------------------------
# Load feature-engineered CSV + sample
# -----------------------------------------------------------
def load_fe_dataset(csv_path, sample_n):
    logging.info(f"Loading FE dataset: {csv_path}")
    df = pd.read_csv(csv_path)

    if sample_n < len(df):
        df = df.sample(sample_n, random_state=42).reset_index(drop=True)
    logging.info(f"Sampled {len(df)} rows for SHAP.")

    feature_cols = [c for c in df.columns if c != "ccmatrix_score"]
    X = df[feature_cols].values
    y = df["ccmatrix_score"].values

    return df, X, y, feature_cols


# -----------------------------------------------------------
# Load model + scalers
# -----------------------------------------------------------
def load_model_and_scalers(model_dir, model_name):
    logging.info(f"Loading model: {model_name}")

    model_path = os.path.join(model_dir, f"{model_name}.keras")
    model = tf.keras.models.load_model(model_path)

    with open(os.path.join(model_dir, "feature_scaler.pkl"), "rb") as f:
        feature_scaler = pickle.load(f)

    with open(os.path.join(model_dir, "target_scaler.pkl"), "rb") as f:
        target_scaler = pickle.load(f)

    return model, feature_scaler, target_scaler


# -----------------------------------------------------------
# Compute SHAP values
# -----------------------------------------------------------
def compute_shap(model, X_scaled, feature_names, output_dir, model_name):
    logging.info(f"Computing SHAP values for model: {model_name}")

    # KernelExplainer expects a prediction function
    predict_fn = lambda inputs: model.predict(inputs).reshape(-1)

    # Using 50 background samples for stable SHAP estimation
    background = X_scaled[np.random.choice(X_scaled.shape[0], 50, replace=False)]

    explainer = shap.KernelExplainer(predict_fn, background)

    logging.info("Running SHAP... (this may take several minutes)")
    shap_values = explainer.shap_values(X_scaled)

    shap_values = np.array(shap_values)  # ensure ndarray

    # Save raw SHAP
    np.save(os.path.join(output_dir, f"shap_values_{model_name}.npy"), shap_values)

    # SHAP summary plot
    shap.summary_plot(
        shap_values,
        X_scaled,
        feature_names=feature_names,
        show=False
    )
    plt.savefig(os.path.join(output_dir, f"shap_summary_{model_name}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # SHAP bar plot
    shap.summary_plot(
        shap_values,
        X_scaled,
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )
    plt.savefig(os.path.join(output_dir, f"shap_bar_{model_name}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Export importance as CSV
    importance = np.mean(np.abs(shap_values), axis=0)  # (features,)
    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    }).sort_values("importance", ascending=False)

    df_imp.to_csv(os.path.join(output_dir, f"shap_importance_{model_name}.csv"), index=False)

    logging.info(f"Saved SHAP results for {model_name}")

    return shap_values


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Compute SHAP importance")
    parser.add_argument("--csv", required=True, help="Feature-engineered CSV (e.g., 80K FE dataset)")
    parser.add_argument("--model_root", required=True, help="Folder containing mlp/residual_mlp/transformer folders")
    parser.add_argument("--model", required=True, choices=["mlp", "residual_mlp", "transformer"])
    parser.add_argument("--sample_n", type=int, default=200, help="Rows to compute SHAP on")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load FE dataset
    df, X, y, feature_cols = load_fe_dataset(args.csv, args.sample_n)

    # 2. Load model + scalers
    model_dir = os.path.join(args.model_root, args.model)
    model, feature_scaler, target_scaler = load_model_and_scalers(model_dir, args.model)

    # 3. Scale for the model
    X_scaled = feature_scaler.transform(X)

    # 4. Compute SHAP
    shap_values = compute_shap(
        model=model,
        X_scaled=X_scaled,
        feature_names=feature_cols,
        output_dir=OUTPUT_DIR,
        model_name=args.model
    )

    logging.info("SHAP analysis complete!")


if __name__ == "__main__":
    main()
