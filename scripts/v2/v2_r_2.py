import pandas as pd
import numpy as np
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import argparse

def main():
    parser = argparse.ArgumentParser(description="Compute R2 for all models in eval_results.csv")
    parser.add_argument("--csv", required=True, help="eval_results.csv path")
    parser.add_argument("--output", default="model_metrics.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    # True label
    y_true = df["ccmatrix_score"].values

    # Auto-detect all prediction columns: pred_mlp, pred_residual_mlp, pred_transformer
    model_cols = [c for c in df.columns if c.startswith("pred_")]

    metrics = {}

    for col in model_cols:
        model_name = col.replace("pred_", "")
        y_pred = df[col].values

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        metrics[model_name] = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }

        print(f"\n=== {model_name.upper()} ===")
        print(f"RMSE: {rmse:.6f}")
        print(f"MAE:  {mae:.6f}")
        print(f"R²:   {r2:.6f}")

    # Save CSV
    out_df = pd.DataFrame([
        {"model": m, **scores} for m, scores in metrics.items()
    ])
    out_df.to_csv(args.output, index=False)

    # Save JSON
    json_path = args.output.replace(".csv", ".json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\nSaved model metrics → {args.output}")
    print(f"Saved metrics JSON  → {json_path}")


if __name__ == "__main__":
    main()
