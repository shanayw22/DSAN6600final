"""
v2_evaluate_models.py
=========================================================
Evaluate three final trained models on 50 FE rows.

Steps:
  1. Load FE dataset (10000 rows)
  2. Load saved models + scalers
  3. Predict and compute RMSE/MAE
  4. Output CSV (all predictions) + metrics JSON
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
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)


# ============================================================
# Load FE dataset
# ============================================================
def load_dataset(csv_path):

    logging.info(f"Loading dataset from: {csv_path}")

    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c != "ccmatrix_score"]

    X = df[feature_cols].values
    y = df["ccmatrix_score"].values

    logging.info(f"Loaded {X.shape[0]:,} samples, {X.shape[1]} features.")
    return X, y, df, feature_cols


# ============================================================
# Predict with a trained model
# ============================================================
def predict_with_model(model_dir, model_type, X_raw):

    logging.info(f"Evaluating model: {model_type}")

    # Load model
    model_path = os.path.join(model_dir, f"{model_type}.keras")
    model = tf.keras.models.load_model(model_path)

    # Load scalers
    with open(os.path.join(model_dir, "feature_scaler.pkl"), "rb") as f:
        feature_scaler = pickle.load(f)

    with open(os.path.join(model_dir, "target_scaler.pkl"), "rb") as f:
        target_scaler = pickle.load(f)

    # Scale features
    X_scaled = feature_scaler.transform(X_raw)

    # Predict
    y_pred_scaled = model.predict(X_scaled).reshape(-1)
    y_pred = target_scaler.inverse_transform(
        y_pred_scaled.reshape(-1, 1)
    ).flatten()

    return y_pred


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Evaluate final trained models with FE dataset")
    parser.add_argument("--csv", required=True, help="Feature-engineered CSV file")
    parser.add_argument("--model_root", required=True, help="Directory storing mlp/residual/transformer folders")
    parser.add_argument("--output", default="eval_results.csv")
    args = parser.parse_args()

    # Load dataset
    X, y, df, feature_cols = load_dataset(args.csv)

    # Model types
    model_types = ["mlp", "residual_mlp", "transformer"]

    # Save metrics
    metrics_summary = {}

    # Predict for each model
    for m in model_types:
        model_dir = os.path.join(args.model_root, m)

        preds = predict_with_model(model_dir, m, X)

        # Add predictions to DataFrame
        df[f"pred_{m}"] = preds
        df[f"err_{m}"] = preds - y

        # Compute metrics
        rmse = np.sqrt(mean_squared_error(y, preds))
        mae = mean_absolute_error(y, preds)

        metrics_summary[m] = {"rmse": rmse, "mae": mae}

        logging.info(f"\n====== {m} ======")
        logging.info(f"RMSE = {rmse:.6f}")
        logging.info(f"MAE  = {mae:.6f}")


    # Save all columns (all model predictions)
    df.to_csv(args.output, index=False)
    logging.info(f"\nSaved evaluation results → {args.output}")

    # Save metrics JSON
    metrics_path = args.output.replace(".csv", "_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=4)

    logging.info(f"Saved metrics summary → {metrics_path}")


if __name__ == "__main__":
    main()
