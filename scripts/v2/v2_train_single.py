"""
v2_train_single.py
============================

Train final models (MLP, Residual MLP, Transformer)
on the entire dataset (e.g., 80,000 rows).

This script:
  - Loads feature-engineered CSV
  - Loads best hyperparameters from JSON
  - Trains final model on all data
  - Saves:
      • trained model (.keras)
      • feature scaler (.pkl)
      • target scaler (.pkl)
      • training history (.json)
============================
"""

import os
import json
import pickle
import logging
import warnings
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from v2_models import get_model_builder

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings('ignore')

# ============================================================================
# Confugiration
# ============================================================================
OUTPUT_DIR = '../v2/V2_Final_Models/'
PARAM_DIR = '../v2/V2_CV_Output/'

# ============================================================
# Load dataset
# ============================================================
def load_dataset(csv_path):

    logging.info(f"Loading dataset from: {csv_path}")

    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c != "ccmatrix_score"]

    X = df[feature_cols].values
    y = df["ccmatrix_score"].values

    logging.info(f"Loaded {X.shape[0]:,} samples, {X.shape[1]} features.")
    return X, y


# ============================================================
# Train final model
# ============================================================
def train_final_model(model_type, params, X, y, output_dir):
    
    logging.info(f"\n===== Training FINAL model: {model_type} =====")
    logging.info(f"Using params: {params}")

    # Create output directory
    model_dir = os.path.join(output_dir, model_type)
    os.makedirs(model_dir, exist_ok=True)

    # --------------------------------------------------------
    # Scaling
    # --------------------------------------------------------
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    X_scaled = feature_scaler.fit_transform(X)
    y_scaled = target_scaler.fit_transform(y.reshape(-1, 1)).flatten()

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------
    input_dim = X.shape[1]
    model_fn = get_model_builder(model_type)
    model = model_fn(input_dim=input_dim, **params)

    # --------------------------------------------------------
    # Train + EarlyStopping
    # --------------------------------------------------------
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=6,
        min_delta=1e-4,
        restore_best_weights=True
    )

    history = model.fit(
        X_scaled, y_scaled,
        validation_split=0.1,
        epochs=60,
        batch_size=128,
        callbacks=[early_stop],
        verbose=1
    )

    # --------------------------------------------------------
    # Save model + scalers + logs
    # --------------------------------------------------------
    model_path = os.path.join(model_dir, f"{model_type}.keras")
    model.save(model_path)
    logging.info(f"Saved model → {model_path}")

    # Save scalers
    with open(os.path.join(model_dir, "feature_scaler.pkl"), "wb") as f:
        pickle.dump(feature_scaler, f)

    with open(os.path.join(model_dir, "target_scaler.pkl"), "wb") as f:
        pickle.dump(target_scaler, f)

    # Save training history
    with open(os.path.join(model_dir, "training_history.json"), "w") as f:
        json.dump(history.history, f, indent=4)

    logging.info(f"Saved scalers + history to {model_dir}")


# ============================================================
# Load best params from JSON
# ============================================================
def load_best_params(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data["best_params"]


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Train final TQE models")
    parser.add_argument("--csv", required=True, help="Path to feature-engineered CSV")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load dataset
    X, y = load_dataset(args.csv)

    # List of models you want to train
    model_types = ["mlp", "residual_mlp", "transformer"]

    for m in model_types:
        json_path = os.path.join(PARAM_DIR, f"best_params_{m}.json")
        logging.info(f"\nLoading best params: {json_path}")

        params = load_best_params(json_path)
        train_final_model(m, params, X, y, OUTPUT_DIR)

if __name__ == "__main__":
    main()
