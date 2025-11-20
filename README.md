# Translation Quality Estimation for Hindi-Chinese Pairs

A deep learning project to predict translation quality scores for Hindi-Chinese sentence pairs without reference translations.

## Project Overview

**Goal**: Predict CCMatrix alignment scores (continuous) for Hindi-Chinese translation pairs using sentence-level features.

**Approach**: Regression model that predicts continuous alignment scores based on:
- Length and statistical features
- Semantic similarity (multiple embedding models)
- Language model fluency (Chinese perplexity)
- Advanced interaction and polynomial features

**Current Status**: ✅ Deep learning models implemented and training

## Key Features

- **Regression Task**: Predicts continuous CCMatrix alignment scores (0.0-1.0 range)
- **Feature Engineering**: 30+ features including semantic similarities, perplexity, interactions
- **Colinearity Removal**: Automatically removes highly correlated features (correlation > 0.95)
- **Deep Learning Models**: 
  - Simple MLP with regularization
  - Deep MLP with Dropout & Batch Normalization
  - Residual MLP with skip connections
- **Target Transformation**: StandardScaler applied to targets for better neural network convergence

## Project Structure

```
DSAN6600final/
├── data/
│   └── hi-zh.txt/          # CCMatrix Hindi-Chinese parallel corpus
├── models/                  # Trained models and artifacts
├── notebooks/
│   └── notebook1.ipynb      # Data exploration and analysis
├── scripts/
│   ├── train_model.py       # Model training script
│   ├── predict_quality.py   # Prediction script
│   └── README.md            # Detailed script documentation
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Models

```bash
cd scripts
python train_model.py
```

This will:
- Load 200,000 sentence pairs (configurable)
- Compute embedding similarities and semantic features
- Extract 30+ features
- Remove colinear features
- Train 3 deep learning models (100 epochs each)
- Save best model to `../models/`

### 3. Predict Quality

```bash
python predict_quality.py --hindi "मैं आज स्कूल जा रहा हूँ।" --chinese "我今天要去学校。"
```

## Features Used

### Length Features (9)
- Character/word counts, ratios, differences, averages
- Normalized length ratios

### Semantic Features (11)
- `embedding_similarity`: Cosine similarity using `paraphrase-multilingual-MiniLM-L12-v2`
- `labse_similarity`: Cosine similarity using `sentence-transformers/LaBSE`
- `chinese_perplexity`: Language model perplexity (fluency indicator)
- `chinese_fluency`: Inverse of perplexity
- Interaction features (embedding × labse, averages, differences)
- Polynomial features (squared similarities)
- Ratio features (similarity ratios)

### Alignment Features (1)
- Character-level alignment heuristic

### Interaction Features (3)
- Length-similarity interactions
- Length-perplexity interactions
- Length-perplexity ratios

### Statistical Features (8)
- Punctuation counts and ratios
- Digit counts
- Vocabulary diversity

**Total**: ~32 features (reduced to ~25-30 after colinearity removal)

## Model Architecture

All models use:
- **Input**: Scaled features (StandardScaler)
- **Output**: Continuous CCMatrix score (inverse transformed from scaled space)
- **Loss**: Mean Squared Error (MSE)
- **Metrics**: MAE, RMSE, R², Pearson, Spearman correlation
- **Regularization**: L2 (0.01), Dropout (0.2-0.5), Batch Normalization
- **Training**: 100 epochs with early stopping (patience=10)

## Evaluation Metrics

- **MAE** (Mean Absolute Error): Average prediction error
- **RMSE** (Root Mean Squared Error): Penalizes larger errors
- **R²** (Coefficient of Determination): Proportion of variance explained
- **Pearson Correlation**: Linear relationship strength
- **Spearman Correlation**: Monotonic relationship strength

## Configuration

Edit `scripts/train_model.py` to customize:
- `SAMPLE_SIZE`: Number of pairs to use (default: 200000)
- `DATA_PATH`: Path to data directory
- `OUTPUT_DIR`: Where to save models
- `SKIP_DEEP_LEARNING`: Set to True to skip neural networks

## Output Files

Models are saved to `models/`:
- `quality_estimation_*.h5`: Best trained model (Keras format)
- `scaler.pkl`: Feature scaler
- `target_scaler.pkl`: Target scaler (for inverse transform)
- `feature_columns.pkl`: Feature names (after colinearity removal)
- `training_results.pkl`: Performance metrics
- `training_summary.txt`: Text summary

## TODO / Future Improvements

### High Priority
- [ ] **Hyperparameter Tuning**: Grid search or Bayesian optimization for learning rate, batch size, architecture
- [ ] **Feature Importance Analysis**: SHAP values or permutation importance to understand feature contributions
- [ ] **Error Analysis**: Examine high-error cases to identify patterns and improve features
- [ ] **Cross-Validation**: Implement k-fold cross-validation for more robust evaluation
- [ ] **Model Ensembling**: Combine predictions from multiple models (voting or stacking)

### Medium Priority
- [ ] **Additional Features**: 
  - POS tag distributions
  - Named entity overlap
  - Word alignment scores
  - Translation probability from small NMT model
- [ ] **Data Augmentation**: Synthesize training examples to improve generalization
- [ ] **Human Evaluation**: Create small human-annotated test set for validation
- [ ] **Visualization**: Add plots for predictions vs actual, feature importance, error distributions
- [ ] **Prediction Intervals**: Implement proper prediction intervals instead of heuristic confidence intervals

### Low Priority
- [ ] **Transformer Models**: Fine-tune mBERT/XLM-R directly for quality estimation
- [ ] **Siamese Networks**: Encode source/target separately and compare
- [ ] **Active Learning**: Select most informative examples for human annotation
- [ ] **Deployment**: Create API or web interface for easy use
- [ ] **Documentation**: Add more detailed docstrings and usage examples

## Research Questions

1. **Can we predict quality from source-target features alone?**
   - ✅ Yes, current models achieve reasonable correlation with CCMatrix scores

2. **Which features are most predictive?**
   - 🔄 In progress: Need feature importance analysis

3. **How well do predictions correlate with human judgments?**
   - ⏳ Pending: Requires human-annotated validation set

## Challenges and Mitigations

| Challenge | Mitigation |
|-----------|------------|
| Label quality (CCMatrix = alignment, not quality) | Use multiple quality proxies; focus on relative ranking |
| Feature engineering complexity | Automated colinearity removal; iterative feature selection |
| Computational resources | Batch processing; efficient models (sentence-transformers) |
| Model interpretability | Feature importance analysis; SHAP values (planned) |
| Evaluation without human labels | Multiple metrics; qualitative analysis |

## Success Criteria

### Minimum Viable ✅
- [x] Model outperforms simple baselines
- [x] Reasonable correlation with CCMatrix scores (Spearman > 0.5)
- [x] Identifies clear high/low quality examples

### Stretch Goals 🔄
- [ ] Strong correlation (Spearman > 0.7)
- [ ] Useful for filtering low-quality pairs
- [ ] Interpretable feature importance
- [ ] Generalizes to unseen data

## Technical Stack

- **Feature Engineering**: pandas, numpy, scikit-learn
- **Embeddings**: sentence-transformers, transformers (HuggingFace)
- **Models**: TensorFlow/Keras
- **Evaluation**: scikit-learn metrics, scipy (correlations)
- **Data**: CCMatrix Hindi-Chinese parallel corpus

## License

See individual data files for licensing information.

## Contact

For questions or issues, please refer to the project documentation in `scripts/README.md`.
