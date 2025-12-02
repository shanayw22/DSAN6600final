# Translation Quality Estimation — v2 Pipeline

This v2 version is a major upgrade over the original `train_model.py`, `predict_quality.py`, and `example_usage.py`.  
It introduces modular architecture, controlled sampling, improved feature engineering, and a full hyperparameter tuning workflow.

---

# 1. Overview of v2

This version includes:

- **Modular Design**  
- **Cross-Validation–Driven Hyperparameter Tuning**
- **Updated Feature Engineering Pipeline**

---

## Sample size - 80000

## for hyperparameter tuning 

v2 is a modification of the origin scripts include train_model.py; predict_quality.py and example_usage.py
however it's focus is on hyperparameter tuning with cross validation and feature engineering with sample size control.

## How to run the scripts under v2?

first build a conda environment and us pip to install the requirements.txt

under v2 directory

run python v2_feature_engineering.py --sample_size 80000

run python v2_train_cv.py \
    --csv ../../V2_FE_Output/tqe_features_80000.csv \
    --model transformer \ 

for --model you can choose from ['mlp', 'residual_mlp', 'transformer']
 
## Hyperparameter tuning results



### search space

### optimal parameters for three different types of models

### Model performance with optimal parameters
