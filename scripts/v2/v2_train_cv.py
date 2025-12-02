"""
v2_train_cv.py
----------------------------------------------------
Cross-Validation + Hyperparameter Search module.

This script:
  Loads feature-engineered tabular data (CSV)
  Uses v2_models.py to construct models
  Runs K-Fold CV to evaluate model performance
    - Uses v2_models.py to construct models
    - scales data using StandardScaler
  Performs hyperparameter optimization
  Show results on test dataset with best hyperparameters

This module DOES NOT train the final model.
Use v2_train_single.py for final training.
----------------------------------------------------
"""

import os
import json
import argparse
import logging
import numpy as np
import pandas as pd
import warnings
from functools import partial
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from v2_models import get_model_builder
import tensorflow as tf

# Set up logging and warnings
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings('ignore')


# ============================================================
# Configuration
# ============================================================
OUTPUT_DIR = '../v2/V2_CV_Output/'

# ============================================================
# Load csv file
# ============================================================
def load_tabular_dataset(csv_path: str, test_ratio: float =0.1):
    """
    Load feature-engineered tabular dataset from CSV.
    Args:
        csv_path (str): path to feature-engineered
        test_ratio (float): fraction of data to reserve for test set
    Returns:
        X (np.ndarray): feature matrix
        y (np.ndarray): target array
        test_df (pd.DataFrame): test dataset saved separately
    """

    logging.info(f"Loading feature-engineered CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    
    logging.info("Split and saving test dataset with fraction of 10%...")
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    test_size = int(len(df) * test_ratio)
    test_df = df.iloc[:test_size].copy()
    trainval_df = df.iloc[test_size:].copy()

    feature_cols = [col for col in df.columns if col != "ccmatrix_score"]

    X = trainval_df[feature_cols].values
    y = trainval_df["ccmatrix_score"].values

    logging.info(f"Loaded {X.shape[0]:,} samples with {X.shape[1]} features.")

    return X, y, test_df

# ============================================================
# K-Fold Cross Validation
# ============================================================
def run_kfold_cv(X, y, model_fn, hyperparams, k=5, random_state=42):
    """
    Run K-Fold cross-validation for a given model & hyperparameters.
    Scaling is performed per-fold to prevent data leakage.
    """

    logging.info(f"Running {k}-Fold CV for hyperparams: {hyperparams}")

    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

    fold_rmses = []
    fold_maes = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        logging.info(f"  Fold {fold_idx + 1}/{k}")

        # Split
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Per-fold scaling 
        feature_scaler = StandardScaler()
        X_train_scaled = feature_scaler.fit_transform(X_train)
        X_val_scaled = feature_scaler.transform(X_val)

        target_scaler = StandardScaler()
        y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).flatten()
    
        # Build model for this fold
        model = model_fn()

        # Print Model Structure & Params
        logging.info(f"[Model Built for Fold {fold_idx+1}] Hyperparameters: {hyperparams}")

        try:
            model.summary(print_fn=lambda x: logging.info(x))
        except Exception as e:
            logging.info(f"(Model summary unavailable: {e})")

        # Train
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            min_delta=1e-4
        )

        model.fit(
            X_train_scaled, y_train_scaled,
            epochs=100,
            batch_size=128,
            validation_data=(X_val_scaled, y_val_scaled),
            callbacks=[early_stop],
            verbose=0
        )

        # Predict
        val_pred_scaled = model.predict(X_val_scaled).reshape(-1)

        # Unscale predictions
        val_pred = target_scaler.inverse_transform(
            val_pred_scaled.reshape(-1, 1)
        ).flatten()

        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        mae = mean_absolute_error(y_val, val_pred)

        fold_rmses.append(rmse)
        fold_maes.append(mae)

        logging.info(f"Fold RMSE={rmse:.4f}, MAE={mae:.4f}")

    avg_rmse = np.mean(fold_rmses)
    avg_mae = np.mean(fold_maes)

    logging.info(f"→ CV Results: Avg RMSE={avg_rmse:.4f}, Avg MAE={avg_mae:.4f}")

    return avg_rmse, avg_mae


# ============================================================
# Hyperparameter Search
# ============================================================
def hyperparameter_search(X, y, model_type, search_space, k=5):
    """
    Grid Search over hyperparameter combinations.
    """

    logging.info(f"=== Starting Hyperparameter Search for {model_type} ===")

    # Get model constructor, but still needs input_dim later
    raw_model_fn = get_model_builder(model_type)
    input_dim = X.shape[1]

    best_params = None
    best_score = 1e9

    import itertools
    keys = list(search_space.keys())
    values = list(search_space.values())

    for combination in itertools.product(*values):
        params = dict(zip(keys, combination))
        logging.info(f"\nEvaluating params: {params}")

        # Wrap model_fn to inject input_dim automatically
        model_fn = partial(raw_model_fn, input_dim=input_dim, **params)

        avg_rmse, _ = run_kfold_cv(
            X=X,
            y=y,
            model_fn=model_fn,
            hyperparams=params,
            k=k
        )

        if avg_rmse < best_score:
            best_score = avg_rmse
            best_params = params
            logging.info(f"New best params found! RMSE={best_score:.4f}")

    return best_params, best_score


# ============================================================
# main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Run K-Fold CV for TQE models")
    parser.add_argument("--csv", type=str, required=True, help="Path to feature-engineered CSV file")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Fraction of data to reserve for test set")
    parser.add_argument("--model", type=str, default="mlp", choices=["mlp", "residual_mlp", "transformer"], help="Which model type to evaluate")
    parser.add_argument("--kfolds", type=int, default=5, help="Number of CV folds")
    args = parser.parse_args()
  
    # ============================================================
    # Load dataset
    # ============================================================
    logging.info("Loading dataset...")
    X_raw, y_raw, test_df = load_tabular_dataset(
        csv_path=args.csv,
        test_ratio=args.test_ratio,
    )

    # ============================================================
    # save test dataset
    # ============================================================
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    test_csv_path = os.path.join(OUTPUT_DIR, "test_dataset.csv")
    test_df.to_csv(test_csv_path, index=False)
    logging.info(f"Saved test dataset to: {test_csv_path}")

    # ============================================================
    # Define search space
    # ============================================================
    if args.model == "mlp":
        search_space = {
            "hidden_dim": [64, 128, 256],
            "num_layers": [2, 3, 4],
            "dropout": [0.2, 0.3],
            "learning_rate": [1e-3, 5e-4]
        }

    elif args.model == "residual_mlp":
        search_space = {
            "hidden_dim": [64, 128, 256],
            "num_layers": [2, 3],
            "dropout": [0.2, 0.3],
            "learning_rate": [1e-3, 5e-4]
        }

    elif args.model == "transformer":
        search_space = {
            "d_model": [64, 128],
            "num_heads": [4, 8],
            "dropout": [0.1, 0.2],
            "learning_rate": [1e-3, 5e-4]
        }

    # ============================================================
    # Hyperparameter search via K-Fold CV
    # ============================================================
    best_params, best_rmse = hyperparameter_search(
        X=X_raw,
        y=y_raw,
        model_type=args.model,
        search_space=search_space,
        k=args.kfolds
    )

    logging.info("\n================ BEST RESULT ================")
    logging.info(f"Model:          {args.model}")
    logging.info(f"Best Params:    {best_params}")
    logging.info(f"Best CV RMSE:   {best_rmse:.4f}")
    logging.info("================================================")

    # Save best params
    best_json = os.path.join(OUTPUT_DIR, f"best_params_{args.model}.json")
    with open(best_json, "w") as f:
        json.dump({
            "model": args.model,
            "best_params": best_params,
            "best_cv_rmse": best_rmse
        }, f, indent=4)
    logging.info(f"Saved best hyperparameters to: {best_json}")

# ============================================================
if __name__ == "__main__":
    main()

