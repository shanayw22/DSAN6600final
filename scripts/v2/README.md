# Translation Quality Estimation — v2 Pipeline (Cross Validation & SHAP Analysis)

This v2 version is a major upgrade over the original `train_model.py`, `predict_quality.py`, and `example_usage.py`. It introduces modular architecture, controlled sampling, improved feature engineering, and a full hyperparameter tuning workflow.

This version includes:

- **Modular Design**  
- **Cross-Validation–Driven Hyperparameter Tuning**
- **SHAP Analysis**

---

## V2 File Structure

```text
v2/
├── __pycache__/
│
├── V2_FE_Output/          # Outputs from feature engineering (CSV, logs)
├── V2_CV_Output/          # Cross-validation & hyperparameter search results
├── V2_Final_Models/       # Final trained models (full-data training)
├── V2_Evaluate_Result/    # Model evaluation metrics & visualizations
├── V2_Shap_Result/        # SHAP feature importance analysis outputs
│
├── README.md              # Project documentation (this file)
│
├── v2_feature_engineer.py # Feature engineering from raw HI–ZH sentence pairs
├── v2_models.py           # Unified model builders (LR, RF, XGB, MLP)
├── v2_train_cv.py         # K-Fold CV + hyperparameter optimization
├── v2_train_single.py     # Final model training on full dataset
├── v2_evaluate_models.py  # Evaluate trained models on test split
└── v2_shap_analysis.py    # SHAP-based model interpretability
```

---

## Module Explanations

**v2_feature_engineer.py**

Converts raw HI–ZH sentence pairs into a machine-learning-ready tabular dataset.

Computes:

- semantic similarity features

- perplexity-based fluency features

- length & statistical features

- interaction features

- Removes highly collinear features.

Saves engineered feature CSVs into V2_FE_Output/.

**v2_models.py**

Centralized model construction module.

Provides unified builders for:

- MLP

- Residual MLP  

- Transformer

Designed to integrate cleanly with both cross-validation and final training.

Called internally by `v2_train_cv.py` and `v2_train_single.py`.

**v2_train_cv.py**

Performs K-Fold Cross-Validation on multiple candidate models.

Runs hyperparameter search (grid search).

Logs performance metrics such as RMSE, MAE, and R² for each configuration.

Selects the best hyperparameters and stores results in V2_CV_Output/.

**v2_train_single.py**

Loads best hyperparameters from the CV module.

Trains the final version of the model using the full dataset.

Saves:

- final model file

- scaler

Outputs stored under V2_Final_Models/.

**v2_evaluate_models.py**

Evaluates trained models on the test split.

Computes:

- RMSE

- MAE

- R²

Generates residual plots and prediction-vs-truth csv.

Saves results to V2_Evaluate_Result/.

**v2_shap_analysis.py**

Runs KernelSHAP on the selected model to measure feature importance.

Allows sampling to reduce computation on large datasets.

Produces:

- SHAP summary plot

- SHAP beeswarm plot

- SHAP values array

Results stored inside V2_Shap_Result/.

---

## Pipeline Building and Usage

```text
Raw Data
   ↓
Feature Engineering (v2_feature_engineer.py)
   ↓
Split + Scale (optional)
   ↓
Cross-Validation + Hyperparameter Search (v2_train_cv.py)
   ↓
Full-Data Final Training (v2_train_single.py)
   ↓
Model Evaluation (v2_evaluate_models.py)
   ↓
SHAP Interpretation (v2_shap_analysis.py)
```

**Usage(example): v2_feature_engineer.py**
```bash
python v2_feature_engineer.py \
    --sample_size 200000 \
    --minilm_model paraphrase-multilingual-MiniLM-L12-v2 \
    --labse_model sentence-transformers/LaBSE \
    --colinearity_threshold 0.95
```

**Usage(example): v2_train_cv.py**
```bash
python v2_train_cv.py \
    --csv V2_FE_Output/tqe_features_800k.csv \
    --model mlp \
    --kfolds 5 \
    --test_ratio 0.1
```

**Usage(example): v2_train_single.py**
```bash
python v2_train_single.py \
    --csv V2_FE_Output/tqe_features_800k.csv
```

**Usage(example): v2_evaluate_models.py**
```bash
python v2_evaluate_models.py \
    --csv V2_FE_Output/tqe_features_800k.csv \
    --model_root V2_Final_Models/
```

**Usage(example): v2_shap_analysis.py**
```bash
python v2_shap_analysis.py \
    --csv V2_FE_Output/tqe_features_800k.csv \
    --model_root V2_Final_Models/ \
    --model mlp \
    --sample_n 200
```

---


