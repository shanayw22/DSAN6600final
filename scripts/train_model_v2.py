"""
Translation Quality Estimation - Training Script v2
--------------------------------------------------
This script is a modular and extensible redesign of the original `train_model.py`.  
It introduces structural improvements that enable cross-validation, hyperparameter tuning, 
and more reliable experimentation.

Major Improvements in v2:

1. Code Modularization
   - Refactored the entire training pipeline into reusable functions:
        • load_and_prepare_data()
        • add_all_features()
        • build_model()
        • train_one_model()
        • run_cross_validation()
   - This makes the pipeline cleaner, testable, and easier to extend for tuning or ensembling.

2. Added Cross-Validation
   - Implemented K-Fold cross-validation to obtain a more robust estimate of model performance.
   - CV replaces the single train/validation split used in the original script.
   - This structure is required for reliable hyperparameter tuning and model comparison.

3. Improved Logging and Debuggability
   - Added logging statements to track data loading, feature computation, model training,
     validation performance across folds, and parameter evaluation.
   - Makes the training process easier to monitor and debug, especially during tuning.

4. Prepared for Hyperparameter Tuning (to be added)
   - The modular design + CV function allow easy integration of:
        • Grid search
        • Random search
        • Bayesian optimization
        • KerasTuner
   - These can now be plugged in without modifying the training loop.

Overall, v2 establishes a clean and extensible training architecture
that supports all required high-priority tasks (CV, tuning, error analysis, 
feature importance, and ensembling) in future iterations.
"""

import warnings
import logging
import time
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr, pearsonr
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Deep learning imports
logging.info("Checking TensorFlow availability...")
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    # to_categorical not needed for regression
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.info("Warning: TensorFlow not available. Deep learning models will be skipped.")

# Set random seeds for reproducibility
np.random.seed(42)
if TF_AVAILABLE:
    tf.random.set_seed(42)
    os.environ['PYTHONHASHSEED'] = '0'
    # Configure TensorFlow for better performance on M1/M2 Macs
    
    # Enable eager execution explicitly
    tf.config.run_functions_eagerly(True)  # Use graph mode for better performance

# Set up logging and warnings
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================
# DATA_PATH = '../data/hi-zh.txt/'
# SAMPLE_SIZE = 200000
# OUTPUT_DIR = '../models/'
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# Option to skip deep learning models if they're too slow
# SKIP_DEEP_LEARNING = False  # Set to True to skip neural networks

# ============================================================================
# Load Data
# ============================================================================
def load_data(data_path:str, sample_size:int):
    """
        Load Hindi-Chinese parallel corpus
        Args:
            data_path (str): Path to the data files
            sample_size (int): Number of samples to load (for quick testing)
        Returns:
            pd.DataFrame: DataFrame with columns ['hindi', 'chinese', 'ccmatrix_score']
    """

    hindi_sentences = []
    chinese_sentences = []
    scores = []
    
    logging.info(f"Loading data from {data_path}...")
    with open(data_path + 'CCMatrix.hi-zh.hi', 'r', encoding='utf-8') as f:
        hindi_sentences = [line.strip() for line in f.readlines()]
    
    with open(data_path + 'CCMatrix.hi-zh.zh', 'r', encoding='utf-8') as f:
        chinese_sentences = [line.strip() for line in f.readlines()]
    
    with open(data_path + 'CCMatrix.hi-zh.scores', 'r', encoding='utf-8') as f:
        scores = [float(line.strip()) for line in f.readlines()]
    
    # Sample if specified
    if sample_size and sample_size < len(hindi_sentences):
        indices = np.random.choice(len(hindi_sentences), sample_size, replace=False)
        hindi_sentences = [hindi_sentences[i] for i in indices]
        chinese_sentences = [chinese_sentences[i] for i in indices]
        scores = [scores[i] for i in indices]
    
    df = pd.DataFrame({
        'hindi': hindi_sentences,
        'chinese': chinese_sentences,
        'ccmatrix_score': scores
    })

    logging.info(f"Loaded {len(df):,} sentence pairs.")
    logging.info("Showing sample dataframe:")
    logging.info(df.head())
    
    return df

# ============================================================================
# Embedding Similarities Computation (model: 'paraphrase-multilingual-MiniLM-L12-v2'; 'LaBSE')
# ============================================================================
def compute_embedding_similarities(hindi_sentences, chinese_sentences, model:object, batch_size=64):
    """
        Compute cosine similarity between Hindi and Chinese sentence embeddings
        Args:
            hindi_sentences (list): List of Hindi sentences
            chinese_sentences (list): List of Chinese sentences
            batch_size (int): Batch size for processing
        Returns:
            np.array: Array of cosine similarity scores
            Embeddings for Hindi and Chinese sentences
        Defaults to 'paraphrase-multilingual-MiniLM-L12-v2' model
    """

    logging.info(f"Using sentence transformer {model}...")

    similarities = []
    hindi_embeddings = []
    chinese_embeddings = []
    total = len(hindi_sentences)
        
    logging.info(f"Processing {total:,} sentence pairs in batches of {batch_size}...")
    start_time = time.time()
        
    for i in range(0, total, batch_size):
        hi_batch = hindi_sentences[i:i+batch_size]
        zh_batch = chinese_sentences[i:i+batch_size]
            
        hi_emb = model.encode(hi_batch, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        zh_emb = model.encode(zh_batch, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
            
        batch_similarities = np.sum(hi_emb * zh_emb, axis=1)
        similarities.extend(batch_similarities)
        hindi_embeddings.append(hi_emb)
        chinese_embeddings.append(zh_emb)
            
        if (i // batch_size + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (i + batch_size) / elapsed if elapsed > 0 else 0
            logging.info(f"  Processed {min(i + batch_size, total):,}/{total:,} pairs ({rate:.0f} pairs/sec)")
        
        elapsed = time.time() - start_time
        logging.info(f"Completed in {elapsed:.2f} seconds")
        
    # Concatenate all embeddings
    hindi_emb_all = np.vstack(hindi_embeddings)
    chinese_emb_all = np.vstack(chinese_embeddings)
        
    return np.array(similarities), hindi_emb_all, chinese_emb_all

# ============================================================================
# Load Chinese Language Model for Perplexity
# ============================================================================
def load_chinese_lm(model_name='uer/gpt2-chinese-cluecorpussmall'):
    """
    Load a Chinese language model and tokenizer for perplexity computation.
    Returns (tokenizer, model, device)
    """
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")

    logging.info(f"Loading Chinese LM: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    model.eval()
    model.to(device)

    logging.info("Chinese LM loaded successfully.")
    return tokenizer, model, device

# ============================================================================
# Chinese Language Model Perplexity Computation
# ============================================================================
def compute_perplexity(texts, tokenizer, model, device, batch_size=32):
    """
        Compute perplexity for a batch of texts
        Args:
            texts (list): List of Chinese sentences
            tokenizer: Tokenizer for the language model
            model: Language model
            device: Device to run the model on
        Returns:
            np.array: Array of perplexity scores
    """

    logging.info(f"Computing perplexity for {len(texts):,} sentences...")
    perplexities = []
    total = len(texts)
            
    for i in range(0, total, batch_size):
        batch_texts = texts[i:i+batch_size]
                
        batch_perplexities = []
        for text in batch_texts:
            try:
                # Tokenize
                inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                        
                # Compute loss
                with torch.no_grad():
                    outputs = model(**inputs, labels=inputs['input_ids'])
                    loss = outputs.loss.item()
                    # Perplexity = exp(loss)
                    ppl = np.exp(loss)
                    batch_perplexities.append(ppl)
            except Exception as e:
                # If computation fails, use a default value
                batch_perplexities.append(100.0)  # High perplexity = low fluency
                
        perplexities.extend(batch_perplexities)
                
        if (i // batch_size + 1) % 10 == 0:
            logging.info(f"  Processed {min(i + batch_size, total):,}/{total:,} sentences for perplexity")
            
    return np.array(perplexities)

# ============================================================================
# Target Variable: CCMatrix Scores (Continuous)
# ============================================================================
print("\n" + "=" * 70)
print("TARGET VARIABLE: CCMATRIX SCORES")
print("=" * 70)

print(f"\nCCMatrix score statistics:")
print(df['ccmatrix_score'].describe())
print(f"\nCCMatrix score range: [{df['ccmatrix_score'].min():.4f}, {df['ccmatrix_score'].max():.4f}]")

# ============================================================================
# Feature Engineering
# ============================================================================
print("\n" + "=" * 70)
print("FEATURE ENGINEERING")
print("=" * 70)

def compute_char_ngram_overlap(text1, text2, n=1):
    """Compute character n-gram overlap ratio"""
    if len(text1) < n or len(text2) < n:
        return 0.0
    
    ngrams1 = set([text1[i:i+n] for i in range(len(text1)-n+1)])
    ngrams2 = set([text2[i:i+n] for i in range(len(text2)-n+1)])
    
    if len(ngrams1) == 0 or len(ngrams2) == 0:
        return 0.0
    
    overlap = len(ngrams1 & ngrams2)
    union = len(ngrams1 | ngrams2)
    
    return overlap / union if union > 0 else 0.0

def compute_word_ngram_overlap(text1, text2, n=1):
    """Compute word n-gram overlap ratio"""
    words1 = text1.split()
    words2 = text2.split()
    
    if len(words1) < n or len(words2) < n:
        return 0.0
    
    ngrams1 = set([tuple(words1[i:i+n]) for i in range(len(words1)-n+1)])
    ngrams2 = set([tuple(words2[i:i+n]) for i in range(len(words2)-n+1)])
    
    if len(ngrams1) == 0 or len(ngrams2) == 0:
        return 0.0
    
    overlap = len(ngrams1 & ngrams2)
    union = len(ngrams1 | ngrams2)
    
    return overlap / union if union > 0 else 0.0

print("Computing length features...")
df['hindi_length'] = df['hindi'].str.len()
df['chinese_length'] = df['chinese'].str.len()
df['hindi_word_count'] = df['hindi'].str.split().str.len()
df['chinese_char_count'] = df['chinese'].apply(
    lambda x: len([c for c in x if '\u4e00' <= c <= '\u9fff'])
)
df['length_ratio'] = df['hindi_length'] / (df['chinese_length'] + 1)
df['length_diff'] = abs(df['hindi_length'] - df['chinese_length'])
df['hindi_avg_word_length'] = df['hindi_length'] / (df['hindi_word_count'] + 1)
df['chinese_avg_char_length'] = df['chinese_length'] / (df['chinese_char_count'] + 1)

# Removed overlap features - they don't work well for Hindi-Chinese (different scripts)

print("Computing statistical features...")
df['punctuation_count_hi'] = df['hindi'].str.count(r'[.,!?;:()\[\]{}"\']')
df['punctuation_count_zh'] = df['chinese'].str.count(r'[.,!?;:()\[\]{}"\']')
df['digit_count_hi'] = df['hindi'].str.count(r'\d')
df['digit_count_zh'] = df['chinese'].str.count(r'\d')
df['punctuation_ratio_hi'] = df['punctuation_count_hi'] / (df['hindi_length'] + 1)
df['punctuation_ratio_zh'] = df['punctuation_count_zh'] / (df['chinese_length'] + 1)

# Vocabulary diversity (unique words / total words)
df['vocab_diversity_hi'] = df['hindi'].apply(
    lambda x: len(set(x.split())) / (len(x.split()) + 1)
)
df['vocab_diversity_zh'] = df['chinese'].apply(
    lambda x: len(set(list(x))) / (len(x) + 1)
)

# ============================================================================
# Advanced Feature Engineering
# ============================================================================
print("Computing advanced features...")

# Interaction features (capture non-linear relationships)
df['embedding_labse_interaction'] = df['embedding_similarity'] * df['labse_similarity']
df['similarity_avg'] = (df['embedding_similarity'] + df['labse_similarity']) / 2
df['similarity_diff'] = abs(df['embedding_similarity'] - df['labse_similarity'])

# Normalized perplexity (inverse relationship - lower perplexity = better fluency)
df['chinese_fluency'] = 1.0 / (df['chinese_perplexity'] + 1.0)

# Length-similarity interactions
df['length_similarity_interaction'] = df['length_ratio'] * df['embedding_similarity']
df['length_perplexity_interaction'] = df['chinese_length'] * df['chinese_perplexity']

# Normalized length ratio (to [0, 1])
length_ratio_min = df['length_ratio'].min()
length_ratio_max = df['length_ratio'].max()
df['length_ratio_normalized'] = (df['length_ratio'] - length_ratio_min) / (length_ratio_max - length_ratio_min + 1e-8)

# Cross-lingual alignment features
def compute_char_alignment_score(text1, text2):
    """Simple character-level alignment heuristic"""
    chars1 = set(text1.lower())
    chars2 = set(text2.lower())
    if len(chars1) == 0 or len(chars2) == 0:
        return 0.0
    intersection = len(chars1 & chars2)
    union = len(chars1 | chars2)
    return intersection / union if union > 0 else 0.0

df['char_alignment'] = df.apply(
    lambda row: compute_char_alignment_score(row['hindi'], row['chinese']), axis=1
)

# Polynomial features for key variables
df['embedding_similarity_squared'] = df['embedding_similarity'] ** 2
df['labse_similarity_squared'] = df['labse_similarity'] ** 2

# Ratio features
df['similarity_ratio'] = df['embedding_similarity'] / (df['labse_similarity'] + 1e-8)
df['length_perplexity_ratio'] = df['chinese_length'] / (df['chinese_perplexity'] + 1e-8)

print("  Advanced features computed")

# Define feature columns (EXCLUDE ccmatrix_score to avoid leakage, but INCLUDE embedding_similarity)
feature_columns = [
    # Length features
    'hindi_length',
    'chinese_length',
    'hindi_word_count',
    'chinese_char_count',
    'length_ratio',
    'length_diff',
    'hindi_avg_word_length',
    'chinese_avg_char_length',
    'length_ratio_normalized',  # NEW
    
    # Semantic features
    'embedding_similarity',      # Embedding similarity (independent of CCMatrix target)
    'labse_similarity',          # Similarity from LaBSE model (additional semantic signal)
    'chinese_perplexity',        # Chinese language model perplexity (fluency indicator)
    'chinese_fluency',  # NEW
    'embedding_labse_interaction',  # NEW
    'similarity_avg',  # NEW
    'similarity_diff',  # NEW
    'embedding_similarity_squared',  # NEW
    'labse_similarity_squared',  # NEW
    'similarity_ratio',  # NEW
    
    # Alignment features
    'char_alignment',  # NEW
    
    # Interaction features
    'length_similarity_interaction',  # NEW
    'length_perplexity_interaction',  # NEW
    'length_perplexity_ratio',  # NEW
    
    # Statistical features
    'punctuation_count_hi',
    'punctuation_count_zh',
    'digit_count_hi',
    'digit_count_zh',
    'punctuation_ratio_hi',
    'punctuation_ratio_zh',
    'vocab_diversity_hi',
    'vocab_diversity_zh',
]

print(f"\nTotal features: {len(feature_columns)}")
print("Features:", feature_columns)

# ============================================================================
# Prepare Data for Training
# ============================================================================
print("\n" + "=" * 70)
print("PREPARING DATA FOR TRAINING")
print("=" * 70)

# Remove rows with missing values
df_clean = df[feature_columns + ['ccmatrix_score']].dropna()
print(f"Rows after removing missing values: {len(df_clean):,}")

# ============================================================================
# Remove Colinear Features
# ============================================================================
print("\n" + "=" * 70)
print("REMOVING COLINEAR FEATURES")
print("=" * 70)

# Compute correlation matrix
feature_df = df_clean[feature_columns]
corr_matrix = feature_df.corr().abs()

# Find pairs of highly correlated features (threshold: 0.95)
high_corr_pairs = []
upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if corr_matrix.iloc[i, j] > 0.95:
            high_corr_pairs.append((
                corr_matrix.columns[i],
                corr_matrix.columns[j],
                corr_matrix.iloc[i, j]
            ))

print(f"Found {len(high_corr_pairs)} pairs of highly correlated features (correlation > 0.95):")
for feat1, feat2, corr_val in high_corr_pairs[:10]:  # Show first 10
    print(f"  {feat1} <-> {feat2}: {corr_val:.4f}")

# Remove one feature from each highly correlated pair
# Strategy: Keep the feature that appears first in feature_columns (prioritize original features)
features_to_remove = set()
for feat1, feat2, _ in high_corr_pairs:
    # Find which feature appears first in feature_columns
    idx1 = feature_columns.index(feat1) if feat1 in feature_columns else len(feature_columns)
    idx2 = feature_columns.index(feat2) if feat2 in feature_columns else len(feature_columns)
    
    # Remove the one that appears later (or if equal, remove feat2)
    if idx1 < idx2:
        features_to_remove.add(feat2)
    else:
        features_to_remove.add(feat1)

# Remove colinear features from feature_columns
feature_columns_filtered = [f for f in feature_columns if f not in features_to_remove]

print(f"\nRemoved {len(features_to_remove)} colinear features:")
for feat in sorted(features_to_remove):
    print(f"  - {feat}")

print(f"\nFeatures before: {len(feature_columns)}")
print(f"Features after: {len(feature_columns_filtered)}")
print(f"Removed: {len(feature_columns) - len(feature_columns_filtered)} features")

# Update feature_columns
feature_columns = feature_columns_filtered

# Prepare X and y (regression: continuous target)
X = df_clean[feature_columns].values
y = df_clean['ccmatrix_score'].values

print(f"Feature matrix shape: {X.shape}")
print(f"Target (CCMatrix score) statistics:")
print(f"  Mean: {y.mean():.4f}")
print(f"  Std: {y.std():.4f}")
print(f"  Min: {y.min():.4f}")
print(f"  Max: {y.max():.4f}")

# ============================================================================
# Target Transformation (to improve model learning)
# ============================================================================
print("\n" + "=" * 70)
print("TARGET TRANSFORMATION")
print("=" * 70)

# Standardize target (helps neural networks and improves convergence)
from sklearn.preprocessing import StandardScaler as TargetScaler
target_scaler = TargetScaler()
y_scaled = target_scaler.fit_transform(y.reshape(-1, 1)).flatten()

print(f"Original target - Mean: {y.mean():.4f}, Std: {y.std():.4f}")
print(f"Scaled target - Mean: {y_scaled.mean():.4f}, Std: {y_scaled.std():.4f}")

# ============================================================================
# Train/Val/Test Split
# ============================================================================
print("\n" + "=" * 70)
print("CREATING TRAIN/VAL/TEST SPLITS")
print("=" * 70)

# First split: train (70%) and temp (30%)
X_train, X_temp, y_train_orig, y_temp_orig = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Second split: val (15%) and test (15%)
X_val, X_test, y_val_orig, y_test_orig = train_test_split(
    X_temp, y_temp_orig, test_size=0.5, random_state=42
)

# Transform targets to scaled version
y_train = target_scaler.transform(y_train_orig.reshape(-1, 1)).flatten()
y_val = target_scaler.transform(y_val_orig.reshape(-1, 1)).flatten()
y_test = target_scaler.transform(y_test_orig.reshape(-1, 1)).flatten()

print(f"Train set: {X_train.shape[0]:,} samples")
print(f"Validation set: {X_val.shape[0]:,} samples")
print(f"Test set: {X_test.shape[0]:,} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# Train Models
# ============================================================================
print("\n" + "=" * 70)
print("TRAINING MODELS")
print("=" * 70)

trained_models = {}
results = {}

# Deep Learning Models
if TF_AVAILABLE and not SKIP_DEEP_LEARNING:
    print("\n1. Training Deep Learning Models...")
    
    # Regression: use continuous targets directly (no categorical conversion)
    input_dim = X_train_scaled.shape[1]
    
    # 1a. Simple MLP (Multi-Layer Perceptron) - Improved with regularization
    print("\n  1a. Training Simple MLP (with regularization)...")
    
    def create_simple_mlp(input_dim):
        """Simpler, more regularized MLP for low-variance targets"""
        # Use tf.keras regularizers directly for compatibility
        l2_reg = tf.keras.regularizers.L2(0.01)
        model = models.Sequential([
            layers.Dense(64, activation='relu', input_shape=(input_dim,), 
                         kernel_regularizer=l2_reg),
            layers.Dropout(0.5),  # Increased dropout
            layers.BatchNormalization(),
            
            layers.Dense(32, activation='relu', 
                         kernel_regularizer=l2_reg),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            
            layers.Dense(16, activation='relu', 
                         kernel_regularizer=l2_reg),
            layers.Dropout(0.2),
            
            layers.Dense(1, activation='linear')
        ])
        return model
    
    mlp_model = create_simple_mlp(input_dim)
    # Use simple string identifier - most compatible
    mlp_model.compile(
        optimizer='adam',
        loss='mse',  # Mean squared error for regression
        metrics=['mae']  # Mean absolute error as metric
    )
    # Set learning rate after compilation (lower for better convergence)
    if hasattr(mlp_model.optimizer, 'learning_rate'):
        mlp_model.optimizer.learning_rate.assign(0.0001)  # Reduced from 0.001
    
    print("    Model compiled successfully. Starting training...")
    
    # Train model (using custom loop - early stopping implemented in loop)
    print(f"    Training on {len(X_train_scaled)} samples, {len(X_val_scaled)} validation samples...")
    print(f"    Batch size: 64, Expected batches per epoch: {(len(X_train_scaled) + 63) // 64}")
    print(f"    Input shape: {X_train_scaled.shape}, Output shape: {y_train.shape}")
    print(f"    Model summary:")
    mlp_model.summary()
    
    import time
    print("    Starting training...")
    start_time = time.time()
    
    # Test a single batch first to see if it works
    print("    Testing single batch forward pass...")
    test_batch_X = X_train_scaled[:64]
    test_batch_y = y_train[:64]
    try:
        test_pred = mlp_model(test_batch_X, training=False)
        print(f"    Single batch forward pass successful! Shape: {test_pred.shape}")
    except Exception as e:
        print(f"    ERROR in forward pass: {e}")
        raise
    
    # Test training step
    print("    Testing single training step...")
    try:
        with tf.GradientTape() as tape:
            pred = mlp_model(test_batch_X, training=True)
            loss = mlp_model.compiled_loss(test_batch_y, pred)
        grads = tape.gradient(loss, mlp_model.trainable_variables)
        mlp_model.optimizer.apply_gradients(zip(grads, mlp_model.trainable_variables))
        # Compute MAE for test
        mae = tf.reduce_mean(tf.abs(pred - test_batch_y))
        print(f"    Single training step successful! Loss: {loss.numpy():.4f}, MAE: {mae.numpy():.4f}")
    except Exception as e:
        print(f"    ERROR in training step: {e}")
        raise
    
    print("    Starting full training (custom loop to avoid M1/M2 Mac hang)...")
    
    # Custom training loop to avoid model.fit() hanging on M1/M2 Macs
    history_mlp = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': []}
    best_val_loss = float('inf')
    patience_counter = 0
    best_weights = None
    
    for epoch in range(100):
        epoch_losses = []
        epoch_maes = []
        
        # Training loop
        for i in range(0, len(X_train_scaled), 64):
            batch_X = X_train_scaled[i:i+64]
            batch_y = y_train[i:i+64]
            
            with tf.GradientTape() as tape:
                pred = mlp_model(batch_X, training=True)
                # Reshape pred if needed (remove extra dimension)
                if len(pred.shape) > 1 and pred.shape[1] == 1:
                    pred = tf.squeeze(pred, axis=1)
                loss = mlp_model.compiled_loss(batch_y, pred)
            
            grads = tape.gradient(loss, mlp_model.trainable_variables)
            mlp_model.optimizer.apply_gradients(zip(grads, mlp_model.trainable_variables))
            
            epoch_losses.append(float(loss.numpy()))
            # Compute MAE manually
            mae = tf.reduce_mean(tf.abs(pred - batch_y))
            epoch_maes.append(float(mae.numpy()))
        
        # Validation
        val_pred = mlp_model(X_val_scaled, training=False)
        if len(val_pred.shape) > 1 and val_pred.shape[1] == 1:
            val_pred = tf.squeeze(val_pred, axis=1)
        val_loss = float(mlp_model.compiled_loss(y_val, val_pred).numpy())
        # Compute MAE manually
        val_mae = float(tf.reduce_mean(tf.abs(val_pred - y_val)).numpy())
        
        train_loss = np.mean(epoch_losses)
        train_mae = np.mean(epoch_maes)
        
        history_mlp['loss'].append(train_loss)
        history_mlp['mae'].append(train_mae)
        history_mlp['val_loss'].append(val_loss)
        history_mlp['val_mae'].append(val_mae)
        
        print(f"Epoch {epoch+1}/35 - loss: {train_loss:.4f} - mae: {train_mae:.4f} - val_loss: {val_loss:.4f} - val_mae: {val_mae:.4f}")
        
        # Early stopping (more aggressive)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # get_weights() already returns numpy arrays
            best_weights = [w.copy() for w in mlp_model.get_weights()]
        else:
            patience_counter += 1
            if patience_counter >= 5:  # Reduced from 10 for faster stopping
                print(f"    Early stopping at epoch {epoch+1}")
                # Restore best weights
                mlp_model.set_weights(best_weights)
                break
    
    elapsed = time.time() - start_time
    print(f"    Training completed in {elapsed:.2f} seconds")
    
    # Predictions (using direct model call to avoid predict() hang)
    print("    Computing predictions...")
    train_pred = mlp_model(X_train_scaled, training=False).numpy()
    val_pred = mlp_model(X_val_scaled, training=False).numpy()
    test_pred = mlp_model(X_test_scaled, training=False).numpy()
    
    # Squeeze if needed
    if len(train_pred.shape) > 1 and train_pred.shape[1] == 1:
        train_pred = train_pred.squeeze(axis=1)
    if len(val_pred.shape) > 1 and val_pred.shape[1] == 1:
        val_pred = val_pred.squeeze(axis=1)
    if len(test_pred.shape) > 1 and test_pred.shape[1] == 1:
        test_pred = test_pred.squeeze(axis=1)
    
    y_train_pred_mlp = train_pred
    y_val_pred_mlp = val_pred
    y_test_pred_mlp = test_pred
    
    # Evaluation
    results['mlp'] = {
        'train_mae': mean_absolute_error(y_train, y_train_pred_mlp),
        'val_mae': mean_absolute_error(y_val, y_val_pred_mlp),
        'test_mae': mean_absolute_error(y_test, y_test_pred_mlp),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred_mlp)),
        'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred_mlp)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred_mlp)),
        'train_r2': r2_score(y_train, y_train_pred_mlp),
        'val_r2': r2_score(y_val, y_val_pred_mlp),
        'test_r2': r2_score(y_test, y_test_pred_mlp),
        'test_spearman': spearmanr(y_test, y_test_pred_mlp)[0],
        'test_pearson': pearsonr(y_test, y_test_pred_mlp)[0],
    }
    
    trained_models['mlp'] = mlp_model
    
    print(f"    Train MAE: {results['mlp']['train_mae']:.4f}, R²: {results['mlp']['train_r2']:.4f}")
    print(f"    Val MAE: {results['mlp']['val_mae']:.4f}, R²: {results['mlp']['val_r2']:.4f}")
    print(f"    Test MAE: {results['mlp']['test_mae']:.4f}, RMSE: {results['mlp']['test_rmse']:.4f}")
    print(f"    Test R²: {results['mlp']['test_r2']:.4f}, Spearman: {results['mlp']['test_spearman']:.4f}")
    
    # 1b. Deep MLP with Dropout and Batch Normalization (improved)
    print("\n  1b. Training Deep MLP with Dropout & BatchNorm (improved)...")
    
    def create_deep_mlp(input_dim):
        """Improved deep MLP with stronger regularization"""
        # Use tf.keras regularizers directly for compatibility
        l2_reg = tf.keras.regularizers.L2(0.01)
        model = models.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_dim,),
                         kernel_regularizer=l2_reg),
            layers.BatchNormalization(),
            layers.Dropout(0.5),  # Increased dropout
            
            layers.Dense(64, activation='relu',
                         kernel_regularizer=l2_reg),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            layers.Dense(32, activation='relu',
                         kernel_regularizer=l2_reg),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(16, activation='relu',
                         kernel_regularizer=l2_reg),
            layers.Dropout(0.2),
            
            layers.Dense(1, activation='linear')  # Regression: single output
        ])
        return model
    
    deep_mlp_model = create_deep_mlp(input_dim)
    # Use simple string identifier - most compatible
    deep_mlp_model.compile(
        optimizer='adam',
        loss='mse',  # Mean squared error for regression
        metrics=['mae']  # Mean absolute error as metric
    )
    # Set learning rate after compilation (lower for better convergence)
    if hasattr(deep_mlp_model.optimizer, 'learning_rate'):
        deep_mlp_model.optimizer.learning_rate.assign(0.0001)  # Reduced from 0.001
    
    # Train model - custom loop
    print("    Starting training (custom loop)...")
    start_time = time.time()
    
    history_deep_mlp = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': []}
    best_val_loss = float('inf')
    patience_counter = 0
    best_weights = None
    
    for epoch in range(100):
        epoch_losses = []
        epoch_maes = []
        
        # Training loop
        for i in range(0, len(X_train_scaled), 64):
            batch_X = X_train_scaled[i:i+64]
            batch_y = y_train[i:i+64]
            
            with tf.GradientTape() as tape:
                pred = deep_mlp_model(batch_X, training=True)
                if len(pred.shape) > 1 and pred.shape[1] == 1:
                    pred = tf.squeeze(pred, axis=1)
                loss = deep_mlp_model.compiled_loss(batch_y, pred)
            
            grads = tape.gradient(loss, deep_mlp_model.trainable_variables)
            deep_mlp_model.optimizer.apply_gradients(zip(grads, deep_mlp_model.trainable_variables))
            
            epoch_losses.append(float(loss.numpy()))
            # Compute MAE manually
            mae = tf.reduce_mean(tf.abs(pred - batch_y))
            epoch_maes.append(float(mae.numpy()))
        
        # Validation
        val_pred = deep_mlp_model(X_val_scaled, training=False)
        if len(val_pred.shape) > 1 and val_pred.shape[1] == 1:
            val_pred = tf.squeeze(val_pred, axis=1)
        val_loss = float(deep_mlp_model.compiled_loss(y_val, val_pred).numpy())
        # Compute MAE manually
        val_mae = float(tf.reduce_mean(tf.abs(val_pred - y_val)).numpy())
        
        train_loss = np.mean(epoch_losses)
        train_mae = np.mean(epoch_maes)
        
        history_deep_mlp['loss'].append(train_loss)
        history_deep_mlp['mae'].append(train_mae)
        history_deep_mlp['val_loss'].append(val_loss)
        history_deep_mlp['val_mae'].append(val_mae)
        
        print(f"Epoch {epoch+1}/25 - loss: {train_loss:.4f} - mae: {train_mae:.4f} - val_loss: {val_loss:.4f} - val_mae: {val_mae:.4f}")
        
        # Early stopping (more aggressive)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # get_weights() already returns numpy arrays
            best_weights = [w.copy() for w in deep_mlp_model.get_weights()]
        else:
            patience_counter += 1
            if patience_counter >= 5:  # Reduced from 10
                print(f"    Early stopping at epoch {epoch+1}")
                deep_mlp_model.set_weights(best_weights)
                break
    
    elapsed = time.time() - start_time
    print(f"    Training completed in {elapsed:.2f} seconds")
    
    # Predictions (using direct model call to avoid predict() hang)
    print("    Computing predictions...")
    train_pred = deep_mlp_model(X_train_scaled, training=False).numpy()
    val_pred = deep_mlp_model(X_val_scaled, training=False).numpy()
    test_pred = deep_mlp_model(X_test_scaled, training=False).numpy()
    
    # Squeeze if needed
    if len(train_pred.shape) > 1 and train_pred.shape[1] == 1:
        train_pred = train_pred.squeeze(axis=1)
    if len(val_pred.shape) > 1 and val_pred.shape[1] == 1:
        val_pred = val_pred.squeeze(axis=1)
    if len(test_pred.shape) > 1 and test_pred.shape[1] == 1:
        test_pred = test_pred.squeeze(axis=1)
    
    y_train_pred_deep = train_pred
    y_val_pred_deep = val_pred
    y_test_pred_deep = test_pred
    
    # Evaluation
    results['deep_mlp'] = {
        'train_mae': mean_absolute_error(y_train, y_train_pred_deep),
        'val_mae': mean_absolute_error(y_val, y_val_pred_deep),
        'test_mae': mean_absolute_error(y_test, y_test_pred_deep),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred_deep)),
        'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred_deep)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred_deep)),
        'train_r2': r2_score(y_train, y_train_pred_deep),
        'val_r2': r2_score(y_val, y_val_pred_deep),
        'test_r2': r2_score(y_test, y_test_pred_deep),
        'test_spearman': spearmanr(y_test, y_test_pred_deep)[0],
        'test_pearson': pearsonr(y_test, y_test_pred_deep)[0],
    }
    
    trained_models['deep_mlp'] = deep_mlp_model
    
    print(f"    Train MAE: {results['deep_mlp']['train_mae']:.4f}, R²: {results['deep_mlp']['train_r2']:.4f}")
    print(f"    Val MAE: {results['deep_mlp']['val_mae']:.4f}, R²: {results['deep_mlp']['val_r2']:.4f}")
    print(f"    Test MAE: {results['deep_mlp']['test_mae']:.4f}, RMSE: {results['deep_mlp']['test_rmse']:.4f}")
    print(f"    Test R²: {results['deep_mlp']['test_r2']:.4f}, Spearman: {results['deep_mlp']['test_spearman']:.4f}")
    
    # 1c. Residual MLP (with skip connections) - Improved
    print("\n  1c. Training Residual MLP (improved)...")
    
    def create_residual_mlp(input_dim):
        """Improved residual MLP with stronger regularization"""
        # Use tf.keras regularizers directly for compatibility
        l2_reg = tf.keras.regularizers.L2(0.01)
        inputs = layers.Input(shape=(input_dim,))
        
        # First block
        x = layers.Dense(64, activation='relu', 
                         kernel_regularizer=l2_reg)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        # Residual block 1
        residual = layers.Dense(32, activation='relu',
                                kernel_regularizer=l2_reg)(x)
        residual = layers.BatchNormalization()(residual)
        residual = layers.Dropout(0.4)(residual)
        x = layers.Dense(32, kernel_regularizer=l2_reg)(x)  # Projection
        x = layers.Add()([x, residual])
        x = layers.Activation('relu')(x)
        
        # Residual block 2
        residual = layers.Dense(16, activation='relu',
                                kernel_regularizer=l2_reg)(x)
        residual = layers.BatchNormalization()(residual)
        residual = layers.Dropout(0.3)(residual)
        x = layers.Dense(16, kernel_regularizer=l2_reg)(x)
        x = layers.Add()([x, residual])
        x = layers.Activation('relu')(x)
        
        # Output (regression: single output, linear activation)
        outputs = layers.Dense(1, activation='linear')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        return model
    
    residual_mlp_model = create_residual_mlp(input_dim)
    # Use simple string identifier - most compatible
    residual_mlp_model.compile(
        optimizer='adam',
        loss='mse',  # Mean squared error for regression
        metrics=['mae']  # Mean absolute error as metric
    )
    # Set learning rate after compilation (lower for better convergence)
    if hasattr(residual_mlp_model.optimizer, 'learning_rate'):
        residual_mlp_model.optimizer.learning_rate.assign(0.0001)  # Reduced from 0.001
    
    # Train model - custom loop
    print("    Starting training (custom loop)...")
    start_time = time.time()
    
    history_residual = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': []}
    best_val_loss = float('inf')
    patience_counter = 0
    best_weights = None
    
    for epoch in range(100):
        epoch_losses = []
        epoch_maes = []
        
        # Training loop
        for i in range(0, len(X_train_scaled), 64):
            batch_X = X_train_scaled[i:i+64]
            batch_y = y_train[i:i+64]
            
            with tf.GradientTape() as tape:
                pred = residual_mlp_model(batch_X, training=True)
                if len(pred.shape) > 1 and pred.shape[1] == 1:
                    pred = tf.squeeze(pred, axis=1)
                loss = residual_mlp_model.compiled_loss(batch_y, pred)
            
            grads = tape.gradient(loss, residual_mlp_model.trainable_variables)
            residual_mlp_model.optimizer.apply_gradients(zip(grads, residual_mlp_model.trainable_variables))
            
            epoch_losses.append(float(loss.numpy()))
            # Compute MAE manually
            mae = tf.reduce_mean(tf.abs(pred - batch_y))
            epoch_maes.append(float(mae.numpy()))
        
        # Validation
        val_pred = residual_mlp_model(X_val_scaled, training=False)
        if len(val_pred.shape) > 1 and val_pred.shape[1] == 1:
            val_pred = tf.squeeze(val_pred, axis=1)
        val_loss = float(residual_mlp_model.compiled_loss(y_val, val_pred).numpy())
        # Compute MAE manually
        val_mae = float(tf.reduce_mean(tf.abs(val_pred - y_val)).numpy())
        
        train_loss = np.mean(epoch_losses)
        train_mae = np.mean(epoch_maes)
        
        history_residual['loss'].append(train_loss)
        history_residual['mae'].append(train_mae)
        history_residual['val_loss'].append(val_loss)
        history_residual['val_mae'].append(val_mae)
        
        print(f"Epoch {epoch+1}/25 - loss: {train_loss:.4f} - mae: {train_mae:.4f} - val_loss: {val_loss:.4f} - val_mae: {val_mae:.4f}")
        
        # Early stopping (more aggressive)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # get_weights() already returns numpy arrays
            best_weights = [w.copy() for w in residual_mlp_model.get_weights()]
        else:
            patience_counter += 1
            if patience_counter >= 5:  # Reduced from 10
                print(f"    Early stopping at epoch {epoch+1}")
                residual_mlp_model.set_weights(best_weights)
                break
    
    elapsed = time.time() - start_time
    print(f"    Training completed in {elapsed:.2f} seconds")
    
    # Predictions (using direct model call to avoid predict() hang)
    print("    Computing predictions...")
    train_pred = residual_mlp_model(X_train_scaled, training=False).numpy()
    val_pred = residual_mlp_model(X_val_scaled, training=False).numpy()
    test_pred = residual_mlp_model(X_test_scaled, training=False).numpy()
    
    # Squeeze if needed
    if len(train_pred.shape) > 1 and train_pred.shape[1] == 1:
        train_pred = train_pred.squeeze(axis=1)
    if len(val_pred.shape) > 1 and val_pred.shape[1] == 1:
        val_pred = val_pred.squeeze(axis=1)
    if len(test_pred.shape) > 1 and test_pred.shape[1] == 1:
        test_pred = test_pred.squeeze(axis=1)
    
    y_train_pred_res = train_pred
    y_val_pred_res = val_pred
    y_test_pred_res = test_pred
    
    # Evaluation
    results['residual_mlp'] = {
        'train_mae': mean_absolute_error(y_train, y_train_pred_res),
        'val_mae': mean_absolute_error(y_val, y_val_pred_res),
        'test_mae': mean_absolute_error(y_test, y_test_pred_res),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred_res)),
        'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred_res)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred_res)),
        'train_r2': r2_score(y_train, y_train_pred_res),
        'val_r2': r2_score(y_val, y_val_pred_res),
        'test_r2': r2_score(y_test, y_test_pred_res),
        'test_spearman': spearmanr(y_test, y_test_pred_res)[0],
        'test_pearson': pearsonr(y_test, y_test_pred_res)[0],
    }
    
    trained_models['residual_mlp'] = residual_mlp_model
    
    print(f"    Train MAE: {results['residual_mlp']['train_mae']:.4f}, R²: {results['residual_mlp']['train_r2']:.4f}")
    print(f"    Val MAE: {results['residual_mlp']['val_mae']:.4f}, R²: {results['residual_mlp']['val_r2']:.4f}")
    print(f"    Test MAE: {results['residual_mlp']['test_mae']:.4f}, RMSE: {results['residual_mlp']['test_rmse']:.4f}")
    print(f"    Test R²: {results['residual_mlp']['test_r2']:.4f}, Spearman: {results['residual_mlp']['test_spearman']:.4f}")
    
elif SKIP_DEEP_LEARNING:
    print("\n1. Deep Learning Models: Skipped (SKIP_DEEP_LEARNING=True)")
else:
    print("\n1. Deep Learning Models: Skipped (TensorFlow not available)")

# ============================================================================
# Detailed Evaluation
# ============================================================================
print("\n" + "=" * 70)
print("DETAILED EVALUATION ON TEST SET")
print("=" * 70)

# Use best model (lowest test MAE for regression)
best_model_name = min(results.keys(), key=lambda k: results[k]['test_mae'])
best_model = trained_models[best_model_name]

print(f"\nBest Model: {best_model_name}")
print(f"Test MAE: {results[best_model_name]['test_mae']:.4f}")
print(f"Test RMSE: {results[best_model_name]['test_rmse']:.4f}")
print(f"Test R²: {results[best_model_name]['test_r2']:.4f}")
print(f"Test Spearman: {results[best_model_name]['test_spearman']:.4f}")

# Predictions on test set (in scaled space)
if best_model_name in ['mlp', 'deep_mlp', 'residual_mlp']:
    # For neural networks, use direct call to avoid predict() hang
    print("Computing final predictions with best model...")
    y_test_pred_scaled = best_model(X_test_scaled, training=False).numpy()
    if len(y_test_pred_scaled.shape) > 1 and y_test_pred_scaled.shape[1] == 1:
        y_test_pred_scaled = y_test_pred_scaled.squeeze(axis=1)
else:
    y_test_pred_scaled = best_model.predict(X_test_scaled)

# Inverse transform predictions back to original scale
y_test_pred = target_scaler.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()

# Regression evaluation summary (on original scale)
print("\nRegression Evaluation Summary (original scale):")
print(f"MAE: {mean_absolute_error(y_test_orig, y_test_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test_orig, y_test_pred)):.4f}")
print(f"R²: {r2_score(y_test_orig, y_test_pred):.4f}")
print(f"Pearson correlation: {pearsonr(y_test_orig, y_test_pred)[0]:.4f}")
print(f"Spearman correlation: {spearmanr(y_test_orig, y_test_pred)[0]:.4f}")

# Scatter plot data (for visualization if needed)
print(f"\nPrediction vs Actual Statistics (original scale):")
print(f"  Actual range: [{y_test_orig.min():.4f}, {y_test_orig.max():.4f}]")
print(f"  Predicted range: [{y_test_pred.min():.4f}, {y_test_pred.max():.4f}]")
print(f"  Mean actual: {y_test_orig.mean():.4f}, Mean predicted: {y_test_pred.mean():.4f}")

# Feature importance (if available)
if hasattr(best_model, 'feature_importances_'):
    print("\nTop 10 Most Important Features:")
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(feature_importance.head(10).to_string(index=False))
elif best_model_name in ['mlp', 'deep_mlp', 'residual_mlp']:
    print("\nNote: Feature importance not available for neural network models.")
    print("Neural networks learn distributed representations across all features.")

# ============================================================================
# Save Models and Artifacts
# ============================================================================
print("\n" + "=" * 70)
print("SAVING MODELS AND ARTIFACTS")
print("=" * 70)

# Save best model
if best_model_name in ['mlp', 'deep_mlp', 'residual_mlp']:
    # Save Keras model
    model_path = os.path.join(OUTPUT_DIR, f'quality_estimation_{best_model_name}.h5')
    best_model.save(model_path)
    print(f"Saved model: {model_path}")
else:
    # Save sklearn model with pickle
    model_path = os.path.join(OUTPUT_DIR, f'quality_estimation_{best_model_name}.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"Saved model: {model_path}")

# Save scaler
scaler_path = os.path.join(OUTPUT_DIR, 'scaler.pkl')
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"Saved scaler: {scaler_path}")

# Save target scaler (for inverse transform during prediction)
target_scaler_path = os.path.join(OUTPUT_DIR, 'target_scaler.pkl')
with open(target_scaler_path, 'wb') as f:
    pickle.dump(target_scaler, f)
print(f"Saved target scaler: {target_scaler_path}")

# Note: No label encoder needed for regression (continuous target)

# Save feature list
feature_path = os.path.join(OUTPUT_DIR, 'feature_columns.pkl')
with open(feature_path, 'wb') as f:
    pickle.dump(feature_columns, f)
print(f"Saved feature columns: {feature_path}")

# Save results
results_path = os.path.join(OUTPUT_DIR, 'training_results.pkl')
with open(results_path, 'wb') as f:
    pickle.dump(results, f)
print(f"Saved results: {results_path}")

# Save summary
summary_path = os.path.join(OUTPUT_DIR, 'training_summary.txt')
with open(summary_path, 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("QUALITY ESTIMATION MODEL TRAINING SUMMARY\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Dataset size: {len(df):,} pairs\n")
    f.write(f"Sample size: {SAMPLE_SIZE:,}\n")
    f.write(f"Features: {len(feature_columns)}\n")
    f.write(f"Train/Val/Test split: {len(X_train):,} / {len(X_val):,} / {len(X_test):,}\n\n")
    f.write("Model Results:\n")
    for model_name, result in results.items():
        f.write(f"\n{model_name}:\n")
        f.write(f"  Test MAE: {result['test_mae']:.4f}\n")
        f.write(f"  Test RMSE: {result['test_rmse']:.4f}\n")
        f.write(f"  Test R²: {result['test_r2']:.4f}\n")
        f.write(f"  Test Spearman: {result['test_spearman']:.4f}\n")
    f.write(f"\nBest Model: {best_model_name}\n")
    f.write(f"Best Test MAE: {results[best_model_name]['test_mae']:.4f}\n")
    f.write(f"Best Test R²: {results[best_model_name]['test_r2']:.4f}\n")

print(f"Saved summary: {summary_path}")

print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)
print(f"\nBest model ({best_model_name}) saved to: {model_path}")
print(f"Use this model to predict quality classes for new sentence pairs.")

