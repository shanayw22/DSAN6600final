"""
v2_split_scale.py
------------------

Module B — Data Split + Scaling

This module prepares the feature-engineered dataset for model training:
    ✓ Load FE CSV
    ✓ Split into Train / Validation / Test
    ✓ Fit StandardScaler on TRAIN only
    ✓ Transform all feature splits
    ✓ Standardize target variable (for neural networks)
    ✓ Save scalers for later inference

Output:
    X_train_scaled, X_val_scaled, X_test_scaled
    y_train, y_val, y_test
    scalers (feature_scaler, target_scaler)
"""

import os
import logging
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)


# ============================================================================
# 1. Load feature-engineered CSV
# ============================================================================
def load_fe_csv(path: str):
    """
    Load the feature-engineered CSV produced by v2_feature_engineer.py

    Args:
        path (str): Full file path to tqe_features.csv

    Returns:
        pd.DataFrame
    """
    logging.info(f"Loading feature-engineered CSV: {path}")
    df = pd.read_csv(path)
    logging.info(f"Loaded {len(df):,} rows and {df.shape[1]} columns.")
    return df


# ============================================================================
# 2. Split data into train / val / test
# ============================================================================
def split_dataset(df: pd.DataFrame, feature_cols, target_col='ccmatrix_score',
                  train_ratio=0.7, val_ratio=0.15, random_state=42):
    """
    Split the feature dataset into Train / Validation / Test.

    Args:
        df (pd.DataFrame): cleaned FE dataframe
        feature_cols (list): selected feature column names
        target_col (str): target column
        train_ratio (float): percent for train
        val_ratio (float): percent for validation

    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """

    logging.info("Splitting dataset into train/val/test...")

    X = df[feature_cols].values
    y = df[target_col].values

    # First split: train and temp
    test_ratio = 1 - train_ratio
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_ratio, random_state=random_state
    )

    # Second split: val & test
    val_share = val_ratio / (1 - train_ratio)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1 - val_share, random_state=random_state
    )

    logging.info(f"Train: {X_train.shape[0]:,} rows")
    logging.info(f"Val:   {X_val.shape[0]:,} rows")
    logging.info(f"Test:  {X_test.shape[0]:,} rows")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================================
# 3. Feature Scaling (fit on train only)
# ============================================================================
def scale_features(X_train, X_val, X_test):
    """
    Standardize features using StandardScaler fitted on TRAIN ONLY.

    Args:
        X_train, X_val, X_test (np.ndarray)

    Returns:
        X_train_scaled, X_val_scaled, X_test_scaled, feature_scaler
    """

    logging.info("Scaling features...")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    logging.info("Feature scaling complete.")

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


# ============================================================================
# 4. Target Scaling (helpful for neural networks)
# ============================================================================
def scale_target(y_train, y_val, y_test):
    """
    Standardize target (y). Neural nets benefit greatly from a normalized target.

    Args:
        y_train, y_val, y_test (np.ndarray)

    Returns:
        y_train_scaled, y_val_scaled, y_test_scaled, target_scaler
    """

    logging.info("Scaling target variable...")

    scaler = StandardScaler()

    y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = scaler.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = scaler.transform(y_test.reshape(-1, 1)).flatten()

    logging.info("Target scaling complete.")

    return y_train_scaled, y_val_scaled, y_test_scaled, scaler


# ============================================================================
# 5. Full Pipeline Function
# ============================================================================
def prepare_train_val_test(
        fe_csv_path,
        feature_columns,
        target_column='ccmatrix_score',
        save_dir=None):
    """
    Full data preparation pipeline:
        - Load FE CSV
        - Split into train/val/test
        - Scale features
        - Scale targets
        - Save scalers (optional)

    Args:
        fe_csv_path (str): path to tqe_features.csv
        feature_columns (list): selected feature columns
        target_column (str): regression target
        save_dir (str or None): directory to save scaler.pkl

    Returns:
        dict containing all arrays + scalers
    """

    df = load_fe_csv(fe_csv_path)

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        df, feature_columns, target_col=target_column
    )

    X_train_scaled, X_val_scaled, X_test_scaled, feature_scaler = scale_features(
        X_train, X_val, X_test
    )

    y_train_scaled, y_val_scaled, y_test_scaled, target_scaler = scale_target(
        y_train, y_val, y_test
    )

    # Save scalers
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        with open(os.path.join(save_dir, "feature_scaler.pkl"), "wb") as f:
            pickle.dump(feature_scaler, f)

        with open(os.path.join(save_dir, "target_scaler.pkl"), "wb") as f:
            pickle.dump(target_scaler, f)

        logging.info(f"Saved scalers to {save_dir}")

    return {
        "X_train": X_train_scaled,
        "X_val": X_val_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train_scaled,
        "y_val": y_val_scaled,
        "y_test": y_test_scaled,
        "y_train_orig": y_train,      # keep original for evaluation
        "y_val_orig": y_val,
        "y_test_orig": y_test,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
    }


# ============================================================================
# Main (optional for standalone testing)
# ============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Split + Scale module (v2)")
    parser.add_argument("--csv", type=str, required=True, help="Path to tqe_features.csv")
    parser.add_argument("--save_dir", type=str, default="./scalers", help="Where to save scalers")
    args = parser.parse_args()

    # Example usage — you would replace this with actual feature list
    from v2_feature_engineer import feature_columns

    prepare_train_val_test(args.csv, feature_columns, save_dir=args.save_dir)

    logging.info("Split + Scaling completed successfully.")
