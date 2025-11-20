# Training Script

## train_model.py

Trains a quality estimation model to predict continuous CCMatrix alignment scores (regression).

### Usage

```bash
cd scripts
python train_model.py
```

### What it does:

1. **Loads data** from the Hindi-Chinese parallel corpus
2. **Computes embedding similarities** using multilingual sentence transformers
   - `embedding_similarity`: Using `paraphrase-multilingual-MiniLM-L12-v2`
   - `labse_similarity`: Using `sentence-transformers/LaBSE` (different model to avoid correlation)
3. **Computes Chinese perplexity** using `uer/gpt2-chinese-cluecorpussmall` (fluency indicator)
4. **Extracts features** from sentence pairs:
   - Length features (9 features)
   - Semantic features (11 features): embedding similarities, perplexity, interactions, polynomials
   - Alignment features (1 feature): character-level alignment
   - Interaction features (3 features): length-similarity, length-perplexity interactions
   - Statistical features (8 features): punctuation, digits, vocabulary diversity
5. **Removes colinear features** (correlation > 0.95) to reduce redundancy
6. **Trains deep learning models** (if TensorFlow available):
   - Simple MLP (Multi-Layer Perceptron) with regularization
   - Deep MLP with Dropout & Batch Normalization
   - Residual MLP (with skip connections)
7. **Evaluates** models on test set using MAE, RMSE, R², Pearson, Spearman
8. **Saves** best model, scalers, feature columns, and results

### Output

Saves to `../models/`:
- `quality_estimation_*.h5` - Trained Keras/TensorFlow model (best performing)
- `scaler.pkl` - Feature scaler (StandardScaler)
- `target_scaler.pkl` - Target scaler (StandardScaler for CCMatrix scores)
- `feature_columns.pkl` - List of feature names (after colinearity removal)
- `training_results.pkl` - Model performance metrics
- `training_summary.txt` - Text summary

### Configuration

Edit these variables at the top of the script:
- `SAMPLE_SIZE`: Number of pairs to use (None for full dataset, default: 200000)
- `DATA_PATH`: Path to data directory
- `OUTPUT_DIR`: Where to save models
- `SKIP_DEEP_LEARNING`: Set to True to skip neural network training

### Features Used

The model uses **sentence-level features** including:
- **Length features** (9): Character/word counts, ratios, averages
- **Semantic features** (11): 
  - `embedding_similarity` (paraphrase-multilingual-MiniLM-L12-v2)
  - `labse_similarity` (LaBSE model)
  - `chinese_perplexity` (language model fluency)
  - `chinese_fluency` (inverse of perplexity)
  - Interaction features (embedding × labse, averages, differences)
  - Polynomial features (squared similarities)
  - Ratio features (similarity ratios)
- **Alignment features** (1): Character-level alignment heuristic
- **Interaction features** (3): Length-similarity, length-perplexity interactions
- **Statistical features** (8): Punctuation, digits, vocabulary diversity

**Total**: ~32 features (after colinearity removal, typically reduces to ~25-30)

### Target Variable

- **CCMatrix Score** (continuous): Alignment/similarity score between Hindi and Chinese sentences
- Range: Typically 0.0 to 1.0 (depends on dataset)
- Higher scores indicate better alignment/similarity

### Model Training

- **Epochs**: 100 per model (with early stopping if validation loss doesn't improve)
- **Batch size**: 64
- **Learning rate**: 0.0001
- **Regularization**: L2 regularization (0.01), Dropout (0.2-0.5), Batch Normalization
- **Early stopping**: Patience of 10 epochs

---

## predict_quality.py

Use trained models to predict translation quality for any Hindi-Chinese sentence pair.

### Key Features

✅ **Works with ANY translation system**: NMT models, LLMs (ChatGPT, Claude), Google Translate, human translators, etc.  
✅ **No dependency on translation model**: Only needs source-target sentence pairs  
✅ **Fast predictions**: Extract features and predict in milliseconds  
✅ **Batch processing**: Evaluate multiple translations at once  
✅ **Regression output**: Predicts continuous CCMatrix alignment scores

### Usage

#### 1. Single Prediction (Command Line)

```bash
python predict_quality.py --hindi "मैं आज स्कूल जा रहा हूँ।" --chinese "我今天要去学校。"
```

#### 2. Batch Prediction from CSV

```bash
# CSV file should have 'hindi' and 'chinese' columns
python predict_quality.py --file translations.csv --output predictions.csv
```

#### 3. Interactive Mode

```bash
python predict_quality.py
# Then enter Hindi and Chinese sentences interactively
```

#### 4. Python API

```python
from predict_quality import predict_single, predict_batch

# Single prediction
score, confidence_interval = predict_single(
    hindi_sentence="मैं आज स्कूल जा रहा हूँ।",
    chinese_sentence="我今天要去学校。"
)
print(f"Predicted Score: {score:.4f}")
print(f"95% CI: [{confidence_interval[0]:.4f}, {confidence_interval[1]:.4f}]")

# Batch prediction
sentence_pairs = [
    ("नमस्ते", "你好"),
    ("धन्यवाद", "谢谢"),
]
results = predict_batch(sentence_pairs)
for r in results:
    print(f"Score: {r['predicted_score']:.4f}")
```

### Example: Evaluating LLM Translations

```python
from predict_quality import predict_single

# Get translation from ChatGPT/Claude/any LLM
hindi_source = "यह एक बहुत अच्छी किताब है।"
llm_translation = "这是一本非常好的书。"  # From your LLM

# Evaluate quality
score, ci = predict_single(hindi_source, llm_translation)
print(f"Translation Quality Score: {score:.4f}")
print(f"95% Confidence Interval: [{ci[0]:.4f}, {ci[1]:.4f}]")
```

### Example: Comparing Different Translation Systems

```python
from predict_quality import predict_single

hindi = "मुझे यह पसंद है।"

# Compare translations from different systems
systems = {
    "Google Translate": "我喜欢这个。",
    "ChatGPT": "我挺喜欢这个的。",
    "NMT Model": "这个我喜欢。",
}

for system_name, translation in systems.items():
    score, ci = predict_single(hindi, translation)
    print(f"{system_name}: {score:.4f} (CI: [{ci[0]:.4f}, {ci[1]:.4f}])")
```
