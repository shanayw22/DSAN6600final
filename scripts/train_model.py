"""
Translation Quality Estimation Model Training Script

This script trains a regression model to predict CCMatrix alignment scores
(continuous values) based on sentence-level features.

Target: CCMatrix scores (continuous) - alignment/similarity scores
Features: Sentence characteristics including embedding_similarity, but EXCLUDING ccmatrix_score to avoid leakage
"""

import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Progress bar support
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Warning: tqdm not available. Progress bars will be disabled.")
    # Create a dummy tqdm that does nothing
    def tqdm(iterable, *args, **kwargs):
        return iterable

# Deep learning imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    # to_categorical not needed for regression
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available. Deep learning models will be skipped.")

# Set random seeds for reproducibility
np.random.seed(42)
if TF_AVAILABLE:
    tf.random.set_seed(42)
    os.environ['PYTHONHASHSEED'] = '0'
    
    # ============================================================================
    # GPU Configuration for Optimal Performance
    # ============================================================================
    print("=" * 70)
    print("CONFIGURING GPU")
    print("=" * 70)
    
    # Check for GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ Found {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
            print("✓ GPU memory growth enabled")
            
            # Use GPU for computations (graph mode is faster on GPU)
            tf.config.run_functions_eagerly(False)
            print("✓ Using graph mode for GPU acceleration")
            
            # Enable mixed precision training for faster computation (FP16)
            try:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                print("✓ Mixed precision (FP16) enabled")
                print("  Note: Output layer will use float32 for numerical stability")
            except Exception as e:
                print(f"  Warning: Could not enable mixed precision: {e}")
                print("  Continuing with float32 precision")
            
            USE_GPU = True
            BATCH_SIZE = 256  # Larger batch size for GPU
        except RuntimeError as e:
            print(f"  Warning: GPU configuration error: {e}")
            USE_GPU = False
            BATCH_SIZE = 64
    else:
        print("  No GPU detected, using CPU")
        USE_GPU = False
        BATCH_SIZE = 64
        # For CPU/M1/M2 Macs, use eager execution to avoid hangs
        tf.config.run_functions_eagerly(True)
        print("  Using eager execution for CPU compatibility")
    
    print("=" * 70)
else:
    # TensorFlow not available
    USE_GPU = False
    BATCH_SIZE = 64

# ============================================================================
# Configuration
# ============================================================================
DATA_PATH = '../data/hi-zh.txt/'
SAMPLE_SIZE = 1000
OUTPUT_DIR = '../models/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Option to skip deep learning models if they're too slow
SKIP_DEEP_LEARNING = False  # Set to True to skip neural networks

# Batch size will be set based on GPU availability (see GPU configuration above)
# Default will be set in GPU config section

# ============================================================================
# Load Data
# ============================================================================
print("=" * 70)
print("LOADING DATA")
print("=" * 70)

def load_data(sample_size=None):
    """Load Hindi-Chinese parallel corpus"""
    print("📂 Loading data files...")
    hindi_sentences = []
    chinese_sentences = []
    scores = []
    
    # Load Hindi sentences
    print("  → Loading Hindi sentences...")
    with open(DATA_PATH + 'CCMatrix.hi-zh.hi', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if TQDM_AVAILABLE:
            lines = tqdm(lines, desc="    Reading Hindi", unit=" lines")
        hindi_sentences = [line.strip() for line in lines]
    print(f"    ✓ Loaded {len(hindi_sentences):,} Hindi sentences")
    
    # Load Chinese sentences
    print("  → Loading Chinese sentences...")
    with open(DATA_PATH + 'CCMatrix.hi-zh.zh', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if TQDM_AVAILABLE:
            lines = tqdm(lines, desc="    Reading Chinese", unit=" lines")
        chinese_sentences = [line.strip() for line in lines]
    print(f"    ✓ Loaded {len(chinese_sentences):,} Chinese sentences")
    
    # Load scores
    print("  → Loading alignment scores...")
    with open(DATA_PATH + 'CCMatrix.hi-zh.scores', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if TQDM_AVAILABLE:
            lines = tqdm(lines, desc="    Reading scores", unit=" lines")
        scores = [float(line.strip()) for line in lines]
    print(f"    ✓ Loaded {len(scores):,} scores")
    
    # Sample if specified
    if sample_size and sample_size < len(hindi_sentences):
        print(f"\n  🎲 Sampling {sample_size:,} pairs from {len(hindi_sentences):,} total pairs...")
        indices = np.random.choice(len(hindi_sentences), sample_size, replace=False)
        hindi_sentences = [hindi_sentences[i] for i in indices]
        chinese_sentences = [chinese_sentences[i] for i in indices]
        scores = [scores[i] for i in indices]
        print(f"    ✓ Sampling complete")
    
    print("  → Creating DataFrame...")
    df = pd.DataFrame({
        'hindi': hindi_sentences,
        'chinese': chinese_sentences,
        'ccmatrix_score': scores
    })
    
    return df

df = load_data(sample_size=SAMPLE_SIZE)
print(f"\n✅ Loaded {len(df):,} sentence pairs")
print(f"   Dataset shape: {df.shape}")

# ============================================================================
# Compute Embedding Similarities (if not already computed)
# ============================================================================
print("\n" + "=" * 70)
print("COMPUTING EMBEDDING SIMILARITIES")
print("=" * 70)

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import time
    
    print("🤖 Loading multilingual sentence transformer model...")
    print("   Model: paraphrase-multilingual-MiniLM-L12-v2")
    # Note: show_progress parameter is not available in all versions of sentence-transformers
    # We'll use the model without it and rely on tqdm for progress bars in our own loops
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    print("   ✓ Model loaded successfully!")
    
    def compute_embedding_similarities(hindi_sentences, chinese_sentences, model, batch_size=64):
        """Compute cosine similarity between Hindi and Chinese sentence embeddings
        Also returns embeddings for semantic feature extraction"""
        similarities = []
        hindi_embeddings = []
        chinese_embeddings = []
        total = len(hindi_sentences)
        num_batches = (total + batch_size - 1) // batch_size
        
        print(f"\n  📊 Processing {total:,} sentence pairs in {num_batches:,} batches (batch_size={batch_size})...")
        start_time = time.time()
        
        # Use tqdm for progress bar if available
        batch_range = range(0, total, batch_size)
        if TQDM_AVAILABLE:
            batch_range = tqdm(batch_range, desc="    Computing embeddings", 
                              unit=" batch", total=num_batches)
        
        for i in batch_range:
            hi_batch = hindi_sentences[i:i+batch_size]
            zh_batch = chinese_sentences[i:i+batch_size]
            
            hi_emb = model.encode(hi_batch, show_progress_bar=False, 
                                 convert_to_numpy=True, normalize_embeddings=True)
            zh_emb = model.encode(zh_batch, show_progress_bar=False, 
                                 convert_to_numpy=True, normalize_embeddings=True)
            
            batch_similarities = np.sum(hi_emb * zh_emb, axis=1)
            similarities.extend(batch_similarities)
            hindi_embeddings.append(hi_emb)
            chinese_embeddings.append(zh_emb)
        
        elapsed = time.time() - start_time
        rate = total / elapsed if elapsed > 0 else 0
        print(f"  ✅ Completed in {elapsed:.2f} seconds ({rate:.0f} pairs/sec)")
        
        # Concatenate all embeddings
        print("  → Concatenating embeddings...")
        hindi_emb_all = np.vstack(hindi_embeddings)
        chinese_emb_all = np.vstack(chinese_embeddings)
        print(f"    ✓ Hindi embeddings shape: {hindi_emb_all.shape}")
        print(f"    ✓ Chinese embeddings shape: {chinese_emb_all.shape}")
        
        return np.array(similarities), hindi_emb_all, chinese_emb_all
    
    print("\n🔄 Computing embedding similarities...")
    similarities, hindi_embeddings, chinese_embeddings = compute_embedding_similarities(
        df['hindi'].tolist(),
        df['chinese'].tolist(),
        model,
        batch_size=64
    )
    df['embedding_similarity'] = similarities
    print("  ✅ Embedding similarities computed and added to dataframe")
    
    # Compute semantic features using DIFFERENT embedding model (to avoid correlation with target)
    print("\n🔄 Computing semantic features from alternative embedding model (LaBSE)...")
    try:
        from sentence_transformers import SentenceTransformer as ST
        print("  🤖 Loading LaBSE model...")
        # Note: show_progress parameter is not available in all versions
        # We'll use the model without it and rely on tqdm for progress bars
        labse_model = ST('sentence-transformers/LaBSE')
        print("  ✓ LaBSE model loaded")
        
        def compute_labse_similarity(hindi_sentences, chinese_sentences, model, batch_size=64):
            """Compute cosine similarity using LaBSE embeddings"""
            similarities = []
            total = len(hindi_sentences)
            num_batches = (total + batch_size - 1) // batch_size
            
            print(f"  📊 Processing {total:,} pairs with LaBSE in {num_batches:,} batches...")
            start_time = time.time()
            
            batch_range = range(0, total, batch_size)
            if TQDM_AVAILABLE:
                batch_range = tqdm(batch_range, desc="    LaBSE embeddings", 
                                  unit=" batch", total=num_batches)
            
            for i in batch_range:
                hi_batch = hindi_sentences[i:i+batch_size]
                zh_batch = chinese_sentences[i:i+batch_size]
                
                hi_emb = model.encode(hi_batch, show_progress_bar=False, 
                                     convert_to_numpy=True, normalize_embeddings=True)
                zh_emb = model.encode(zh_batch, show_progress_bar=False, 
                                     convert_to_numpy=True, normalize_embeddings=True)
                
                batch_similarities = np.sum(hi_emb * zh_emb, axis=1)
                similarities.extend(batch_similarities)
                
            elapsed = time.time() - start_time
            rate = total / elapsed if elapsed > 0 else 0
            print(f"  ✅ LaBSE similarity computed in {elapsed:.2f} seconds ({rate:.0f} pairs/sec)")
            
            return np.array(similarities)
        
        df['labse_similarity'] = compute_labse_similarity(
            df['hindi'].tolist(),
            df['chinese'].tolist(),
            labse_model,
            batch_size=64
        )
        print("  ✅ LaBSE similarity added to dataframe")
    except Exception as e:
        print(f"  Warning: Could not compute LaBSE similarity: {e}")
        print("  Using dummy values for LaBSE similarity")
        df['labse_similarity'] = np.random.rand(len(df))
    
    # Compute Chinese language model perplexity (fluency indicator)
    print("\n🔄 Computing Chinese language model perplexity (fluency indicator)...")
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        # Use a Chinese language model for perplexity
        # Try to use GPU if available, otherwise CPU
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"  🖥️  Using device: {device}")
        
        # Load Chinese GPT model for perplexity
        perplexity_model_name = 'uer/gpt2-chinese-cluecorpussmall'
        print(f"  🤖 Loading Chinese language model: {perplexity_model_name}")
        # Note: progress parameter may not be available in all versions
        # We'll use the models without it and rely on tqdm for progress bars
        chinese_tokenizer = AutoTokenizer.from_pretrained(perplexity_model_name)
        chinese_lm = AutoModelForCausalLM.from_pretrained(perplexity_model_name)
        chinese_lm.eval()
        chinese_lm.to(device)
        print(f"  ✓ Model loaded and moved to {device}")
        
        def compute_perplexity(texts, tokenizer, model, device, batch_size=32):
            """Compute perplexity for a batch of texts"""
            perplexities = []
            total = len(texts)
            num_batches = (total + batch_size - 1) // batch_size
            
            print(f"  📊 Computing perplexity for {total:,} sentences in {num_batches:,} batches...")
            start_time = time.time()
            
            batch_range = range(0, total, batch_size)
            if TQDM_AVAILABLE:
                batch_range = tqdm(batch_range, desc="    Computing perplexity", 
                                  unit=" batch", total=num_batches)
            
            for i in batch_range:
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
                
            elapsed = time.time() - start_time
            rate = total / elapsed if elapsed > 0 else 0
            print(f"  ✅ Perplexity computed in {elapsed:.2f} seconds ({rate:.0f} sentences/sec)")
            
            return np.array(perplexities)
        
        df['chinese_perplexity'] = compute_perplexity(
            df['chinese'].tolist(),
            chinese_tokenizer,
            chinese_lm,
            device,
            batch_size=32
        )
        print("  ✅ Chinese perplexity added to dataframe")
        
        # Clean up
        del chinese_lm
        del chinese_tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    except Exception as e:
        print(f"  Warning: Could not compute Chinese perplexity: {e}")
        print("  Using dummy values for Chinese perplexity")
        df['chinese_perplexity'] = np.random.rand(len(df)) * 50.0 + 10.0  # Typical perplexity range
    
except ImportError:
    print("Warning: sentence-transformers not installed. Skipping embedding computation.")
    print("If you have pre-computed embeddings, add 'embedding_similarity' column to dataframe.")
    # Create dummy embedding similarity and semantic features for testing
    df['embedding_similarity'] = np.random.rand(len(df))
    df['labse_similarity'] = np.random.rand(len(df))
    df['chinese_perplexity'] = np.random.rand(len(df)) * 50.0 + 10.0  # Typical perplexity range

print(f"\nEmbedding similarity statistics:")
print(df['embedding_similarity'].describe())

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

print("🔧 Computing length features...")
if TQDM_AVAILABLE:
    tqdm.pandas(desc="    Computing length features")
    df['hindi_length'] = df['hindi'].str.len()
    df['chinese_length'] = df['chinese'].str.len()
    df['hindi_word_count'] = df['hindi'].str.split().str.len()
    df['chinese_char_count'] = df['chinese'].progress_apply(
        lambda x: len([c for c in x if '\u4e00' <= c <= '\u9fff'])
    )
else:
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
print("  ✅ Length features computed")

# Removed overlap features - they don't work well for Hindi-Chinese (different scripts)

print("\n🔧 Computing statistical features...")
df['punctuation_count_hi'] = df['hindi'].str.count(r'[.,!?;:()\[\]{}"\']')
df['punctuation_count_zh'] = df['chinese'].str.count(r'[.,!?;:()\[\]{}"\']')
df['digit_count_hi'] = df['hindi'].str.count(r'\d')
df['digit_count_zh'] = df['chinese'].str.count(r'\d')
df['punctuation_ratio_hi'] = df['punctuation_count_hi'] / (df['hindi_length'] + 1)
df['punctuation_ratio_zh'] = df['punctuation_count_zh'] / (df['chinese_length'] + 1)

# Vocabulary diversity (unique words / total words)
if TQDM_AVAILABLE:
    df['vocab_diversity_hi'] = df['hindi'].progress_apply(
        lambda x: len(set(x.split())) / (len(x.split()) + 1)
    )
    df['vocab_diversity_zh'] = df['chinese'].progress_apply(
        lambda x: len(set(list(x))) / (len(x) + 1)
    )
else:
    df['vocab_diversity_hi'] = df['hindi'].apply(
        lambda x: len(set(x.split())) / (len(x.split()) + 1)
    )
    df['vocab_diversity_zh'] = df['chinese'].apply(
        lambda x: len(set(list(x))) / (len(x) + 1)
    )
print("  ✅ Statistical features computed")

# ============================================================================
# Advanced Feature Engineering
# ============================================================================
print("\n🔧 Computing advanced features...")

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

if TQDM_AVAILABLE:
    df['char_alignment'] = df.progress_apply(
        lambda row: compute_char_alignment_score(row['hindi'], row['chinese']), axis=1
    )
else:
    df['char_alignment'] = df.apply(
        lambda row: compute_char_alignment_score(row['hindi'], row['chinese']), axis=1
    )

# Polynomial features for key variables
df['embedding_similarity_squared'] = df['embedding_similarity'] ** 2
df['labse_similarity_squared'] = df['labse_similarity'] ** 2

# Ratio features
df['similarity_ratio'] = df['embedding_similarity'] / (df['labse_similarity'] + 1e-8)
df['length_perplexity_ratio'] = df['chinese_length'] / (df['chinese_perplexity'] + 1e-8)

print("  ✅ Advanced features computed")

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
# Use indices to split embeddings consistently
indices = np.arange(len(X))
train_indices, temp_indices = train_test_split(
    indices, test_size=0.3, random_state=42
)

# Second split: val (15%) and test (15%)
val_indices, test_indices = train_test_split(
    temp_indices, test_size=0.5, random_state=42
)

# Split X and y using indices
X_train = X[train_indices]
X_val = X[val_indices]
X_test = X[test_indices]

y_train_orig = y[train_indices]
y_val_orig = y[val_indices]
y_test_orig = y[test_indices]

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
# Create tf.data.Dataset for GPU Optimization (if using GPU)
# ============================================================================
if TF_AVAILABLE and not SKIP_DEEP_LEARNING and USE_GPU:
    print("\n" + "=" * 70)
    print("CREATING TF.DATA DATASETS FOR GPU")
    print("=" * 70)
    
    # Convert to TensorFlow datasets for better GPU utilization
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (X_train_scaled.astype(np.float32), y_train.astype(np.float32))
    ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices(
        (X_val_scaled.astype(np.float32), y_val.astype(np.float32))
    ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (X_test_scaled.astype(np.float32), y_test.astype(np.float32))
    ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    print(f"✓ Created tf.data datasets with batch size: {BATCH_SIZE}")
    print(f"✓ Prefetching enabled for optimal GPU utilization")
    print("=" * 70)
else:
    train_dataset = None
    val_dataset = None
    test_dataset = None

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
    
    # Define optimized training functions if using GPU (shared across all models)
    # Note: We'll define these functions per-model after the optimizer is built
    # to avoid tf.function variable creation issues with mixed precision
    if USE_GPU and train_dataset is not None:
        # Placeholder - will be defined per model after optimizer is built
        train_step = None
        val_step = None
        predict_batch = None
    else:
        train_step = None
        val_step = None
        predict_batch = None
    
    # 1a. Simple MLP (Multi-Layer Perceptron) - Improved with regularization
    print("\n  1a. Training Simple MLP (with regularization)...")
    
    def create_simple_mlp(input_dim):
        """Simpler, more regularized MLP for low-variance targets"""
        # Use tf.keras regularizers directly for compatibility
        l2_reg = tf.keras.regularizers.L2(0.01)
        
        # For mixed precision, output layer must be float32
        dtype = 'float32' if USE_GPU and tf.keras.mixed_precision.global_policy().name == 'mixed_float16' else None
        
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
            
            layers.Dense(1, activation='linear', dtype=dtype)  # float32 for mixed precision
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
    
    # Train model (optimized for GPU if available)
    print(f"    Training on {len(X_train_scaled):,} samples, {len(X_val_scaled):,} validation samples...")
    print(f"    Batch size: {BATCH_SIZE}, Expected batches per epoch: {(len(X_train_scaled) + BATCH_SIZE - 1) // BATCH_SIZE}")
    print(f"    Input shape: {X_train_scaled.shape}, Output shape: {y_train.shape}")
    print(f"    Device: {'GPU' if USE_GPU else 'CPU'}")
    print(f"    Model summary:")
    mlp_model.summary()
    
    import time
    print("    Starting training...")
    start_time = time.time()
    
    # Optimized training: use tf.data if GPU available, otherwise custom loop
    if USE_GPU and train_dataset is not None:
        print("    Using optimized GPU training with tf.data...")
        
        # Build optimizer by running a dummy step (needed for mixed precision)
        print("    → Building optimizer (running dummy step)...")
        dummy_batch = next(iter(train_dataset))
        dummy_X, dummy_y = dummy_batch
        with tf.GradientTape() as tape:
            dummy_pred = mlp_model(dummy_X, training=True)
            if len(dummy_pred.shape) > 1 and dummy_pred.shape[1] == 1:
                dummy_pred = tf.squeeze(dummy_pred, axis=1)
            dummy_loss = mlp_model.compiled_loss(dummy_y, dummy_pred)
        dummy_grads = tape.gradient(dummy_loss, mlp_model.trainable_variables)
        mlp_model.optimizer.apply_gradients(zip(dummy_grads, mlp_model.trainable_variables))
        print("    ✓ Optimizer built successfully")
        
        # Now define tf.function decorated functions (optimizer is already built)
        @tf.function
        def train_step(model, x, y):
            with tf.GradientTape() as tape:
                pred = model(x, training=True)
                if len(pred.shape) > 1 and pred.shape[1] == 1:
                    pred = tf.squeeze(pred, axis=1)
                loss = model.compiled_loss(y, pred)
            grads = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
            mae = tf.reduce_mean(tf.abs(pred - y))
            return loss, mae
        
        @tf.function
        def val_step(model, x, y):
            pred = model(x, training=False)
            if len(pred.shape) > 1 and pred.shape[1] == 1:
                pred = tf.squeeze(pred, axis=1)
            loss = model.compiled_loss(y, pred)
            mae = tf.reduce_mean(tf.abs(pred - y))
            return loss, mae
        
        @tf.function
        def predict_batch(model, x):
            return model(x, training=False)
        
        history_mlp = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': []}
        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None
        
        num_epochs = 100
        print(f"\n  🚀 Starting training for up to {num_epochs} epochs...")
        print(f"  📊 Training batches: {(len(X_train_scaled) + BATCH_SIZE - 1) // BATCH_SIZE}")
        print(f"  📊 Validation batches: {(len(X_val_scaled) + BATCH_SIZE - 1) // BATCH_SIZE}")
        
        epoch_range = range(num_epochs)
        if TQDM_AVAILABLE:
            epoch_range = tqdm(epoch_range, desc="    Training epochs", unit=" epoch")
        
        for epoch in epoch_range:
            epoch_losses = []
            epoch_maes = []
            
            # Training loop with tf.data
            for batch_X, batch_y in train_dataset:
                loss, mae = train_step(mlp_model, batch_X, batch_y)
                epoch_losses.append(float(loss))
                epoch_maes.append(float(mae))
            
            # Validation with tf.data
            val_losses = []
            val_maes = []
            for batch_X, batch_y in val_dataset:
                v_loss, v_mae = val_step(mlp_model, batch_X, batch_y)
                val_losses.append(float(v_loss))
                val_maes.append(float(v_mae))
            
            train_loss = np.mean(epoch_losses)
            train_mae = np.mean(epoch_maes)
            val_loss = np.mean(val_losses)
            val_mae = np.mean(val_maes)
            
            history_mlp['loss'].append(train_loss)
            history_mlp['mae'].append(train_mae)
            history_mlp['val_loss'].append(val_loss)
            history_mlp['val_mae'].append(val_mae)
            
            print(f"Epoch {epoch+1}/100 - loss: {train_loss:.4f} - mae: {train_mae:.4f} - val_loss: {val_loss:.4f} - val_mae: {val_mae:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_weights = [w.copy() for w in mlp_model.get_weights()]
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    print(f"    Early stopping at epoch {epoch+1}")
                    mlp_model.set_weights(best_weights)
                    break
    else:
        # Fallback to custom loop for CPU/M1/M2 Macs
        print("    Using custom training loop (CPU compatibility mode)...")
        
        # Test a single batch first
        print("    Testing single batch forward pass...")
        test_batch_X = X_train_scaled[:BATCH_SIZE]
        test_batch_y = y_train[:BATCH_SIZE]
        try:
            test_pred = mlp_model(test_batch_X, training=False)
            print(f"    Single batch forward pass successful! Shape: {test_pred.shape}")
        except Exception as e:
            print(f"    ERROR in forward pass: {e}")
            raise
        
        history_mlp = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': []}
        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None
        
        for epoch in range(100):
            epoch_losses = []
            epoch_maes = []
            
            # Training loop
            for i in range(0, len(X_train_scaled), BATCH_SIZE):
                batch_X = X_train_scaled[i:i+BATCH_SIZE]
                batch_y = y_train[i:i+BATCH_SIZE]
                
                with tf.GradientTape() as tape:
                    pred = mlp_model(batch_X, training=True)
                    if len(pred.shape) > 1 and pred.shape[1] == 1:
                        pred = tf.squeeze(pred, axis=1)
                    loss = mlp_model.compiled_loss(batch_y, pred)
                
                grads = tape.gradient(loss, mlp_model.trainable_variables)
                mlp_model.optimizer.apply_gradients(zip(grads, mlp_model.trainable_variables))
                
                epoch_losses.append(float(loss.numpy()))
                mae = tf.reduce_mean(tf.abs(pred - batch_y))
                epoch_maes.append(float(mae.numpy()))
            
            # Validation
            val_pred = mlp_model(X_val_scaled, training=False)
            if len(val_pred.shape) > 1 and val_pred.shape[1] == 1:
                val_pred = tf.squeeze(val_pred, axis=1)
            val_loss = float(mlp_model.compiled_loss(y_val, val_pred).numpy())
            val_mae = float(tf.reduce_mean(tf.abs(val_pred - y_val)).numpy())
            
            train_loss = np.mean(epoch_losses)
            train_mae = np.mean(epoch_maes)
            
            history_mlp['loss'].append(train_loss)
            history_mlp['mae'].append(train_mae)
            history_mlp['val_loss'].append(val_loss)
            history_mlp['val_mae'].append(val_mae)
            
            print(f"Epoch {epoch+1}/100 - loss: {train_loss:.4f} - mae: {train_mae:.4f} - val_loss: {val_loss:.4f} - val_mae: {val_mae:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_weights = [w.copy() for w in mlp_model.get_weights()]
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    print(f"    Early stopping at epoch {epoch+1}")
                    mlp_model.set_weights(best_weights)
                    break
    
    elapsed = time.time() - start_time
    print(f"    Training completed in {elapsed:.2f} seconds")
    
    # Predictions (optimized for GPU if available)
    print("\n    📊 Computing predictions...")
    if USE_GPU and test_dataset is not None:
        # Use tf.data for faster GPU prediction
        @tf.function
        def predict_batch(model, x):
            return model(x, training=False)
        
        print("      → Training set predictions...")
        train_pred_list = []
        train_batches = list(train_dataset)
        if TQDM_AVAILABLE:
            train_batches = tqdm(train_batches, desc="        Training", unit=" batch")
        for batch_X, _ in train_batches:
            pred = predict_batch(mlp_model, batch_X)
            if len(pred.shape) > 1 and pred.shape[1] == 1:
                pred = tf.squeeze(pred, axis=1)
            train_pred_list.append(pred.numpy())
        train_pred = np.concatenate(train_pred_list)
        
        print("      → Validation set predictions...")
        val_pred_list = []
        val_batches = list(val_dataset)
        if TQDM_AVAILABLE:
            val_batches = tqdm(val_batches, desc="        Validation", unit=" batch")
        for batch_X, _ in val_batches:
            pred = predict_batch(mlp_model, batch_X)
            if len(pred.shape) > 1 and pred.shape[1] == 1:
                pred = tf.squeeze(pred, axis=1)
            val_pred_list.append(pred.numpy())
        val_pred = np.concatenate(val_pred_list)
        
        print("      → Test set predictions...")
        test_pred_list = []
        test_batches = list(test_dataset)
        if TQDM_AVAILABLE:
            test_batches = tqdm(test_batches, desc="        Test", unit=" batch")
        for batch_X, _ in test_batches:
            pred = predict_batch(mlp_model, batch_X)
            if len(pred.shape) > 1 and pred.shape[1] == 1:
                pred = tf.squeeze(pred, axis=1)
            test_pred_list.append(pred.numpy())
        test_pred = np.concatenate(test_pred_list)
        print("      ✅ All predictions computed")
    else:
        # Fallback for CPU
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
        
        # For mixed precision, output layer must be float32
        dtype = 'float32' if USE_GPU and tf.keras.mixed_precision.global_policy().name == 'mixed_float16' else None
        
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
            
            layers.Dense(1, activation='linear', dtype=dtype)  # Regression: single output
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
    
    # Train model - optimized for GPU
    print(f"    Training on {len(X_train_scaled):,} samples, {len(X_val_scaled):,} validation samples...")
    print(f"    Batch size: {BATCH_SIZE}, Device: {'GPU' if USE_GPU else 'CPU'}")
    start_time = time.time()
    
    if USE_GPU and train_dataset is not None:
        # Build optimizer by running a dummy step (needed for mixed precision)
        print("    → Building optimizer (running dummy step)...")
        dummy_batch = next(iter(train_dataset))
        dummy_X, dummy_y = dummy_batch
        with tf.GradientTape() as tape:
            dummy_pred = deep_mlp_model(dummy_X, training=True)
            if len(dummy_pred.shape) > 1 and dummy_pred.shape[1] == 1:
                dummy_pred = tf.squeeze(dummy_pred, axis=1)
            dummy_loss = deep_mlp_model.compiled_loss(dummy_y, dummy_pred)
        dummy_grads = tape.gradient(dummy_loss, deep_mlp_model.trainable_variables)
        deep_mlp_model.optimizer.apply_gradients(zip(dummy_grads, deep_mlp_model.trainable_variables))
        print("    ✓ Optimizer built successfully")
        
        # Now define tf.function decorated functions (optimizer is already built)
        @tf.function
        def train_step(model, x, y):
            with tf.GradientTape() as tape:
                pred = model(x, training=True)
                if len(pred.shape) > 1 and pred.shape[1] == 1:
                    pred = tf.squeeze(pred, axis=1)
                loss = model.compiled_loss(y, pred)
            grads = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
            mae = tf.reduce_mean(tf.abs(pred - y))
            return loss, mae
        
        @tf.function
        def val_step(model, x, y):
            pred = model(x, training=False)
            if len(pred.shape) > 1 and pred.shape[1] == 1:
                pred = tf.squeeze(pred, axis=1)
            loss = model.compiled_loss(y, pred)
            mae = tf.reduce_mean(tf.abs(pred - y))
            return loss, mae
        
        @tf.function
        def predict_batch(model, x):
            return model(x, training=False)
        
        history_deep_mlp = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': []}
        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None
        
        num_epochs = 100
        print(f"\n  🚀 Starting training for up to {num_epochs} epochs...")
        
        epoch_range = range(num_epochs)
        if TQDM_AVAILABLE:
            epoch_range = tqdm(epoch_range, desc="    Training epochs", unit=" epoch")
        
        for epoch in epoch_range:
            epoch_losses = []
            epoch_maes = []
            
            for batch_X, batch_y in train_dataset:
                loss, mae = train_step(deep_mlp_model, batch_X, batch_y)
                epoch_losses.append(float(loss))
                epoch_maes.append(float(mae))
            
            val_losses = []
            val_maes = []
            for batch_X, batch_y in val_dataset:
                v_loss, v_mae = val_step(deep_mlp_model, batch_X, batch_y)
                val_losses.append(float(v_loss))
                val_maes.append(float(v_mae))
            
            train_loss = np.mean(epoch_losses)
            train_mae = np.mean(epoch_maes)
            val_loss = np.mean(val_losses)
            val_mae = np.mean(val_maes)
            
            history_deep_mlp['loss'].append(train_loss)
            history_deep_mlp['mae'].append(train_mae)
            history_deep_mlp['val_loss'].append(val_loss)
            history_deep_mlp['val_mae'].append(val_mae)
            
            print(f"Epoch {epoch+1}/100 - loss: {train_loss:.4f} - mae: {train_mae:.4f} - val_loss: {val_loss:.4f} - val_mae: {val_mae:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_weights = [w.copy() for w in deep_mlp_model.get_weights()]
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    print(f"    Early stopping at epoch {epoch+1}")
                    deep_mlp_model.set_weights(best_weights)
                    break
    else:
        # Fallback to custom loop
        history_deep_mlp = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': []}
        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None
        
        for epoch in range(100):
            epoch_losses = []
            epoch_maes = []
            
            for i in range(0, len(X_train_scaled), BATCH_SIZE):
                batch_X = X_train_scaled[i:i+BATCH_SIZE]
                batch_y = y_train[i:i+BATCH_SIZE]
                
                with tf.GradientTape() as tape:
                    pred = deep_mlp_model(batch_X, training=True)
                    if len(pred.shape) > 1 and pred.shape[1] == 1:
                        pred = tf.squeeze(pred, axis=1)
                    loss = deep_mlp_model.compiled_loss(batch_y, pred)
                
                grads = tape.gradient(loss, deep_mlp_model.trainable_variables)
                deep_mlp_model.optimizer.apply_gradients(zip(grads, deep_mlp_model.trainable_variables))
                
                epoch_losses.append(float(loss.numpy()))
                mae = tf.reduce_mean(tf.abs(pred - batch_y))
                epoch_maes.append(float(mae.numpy()))
            
            val_pred = deep_mlp_model(X_val_scaled, training=False)
            if len(val_pred.shape) > 1 and val_pred.shape[1] == 1:
                val_pred = tf.squeeze(val_pred, axis=1)
            val_loss = float(deep_mlp_model.compiled_loss(y_val, val_pred).numpy())
            val_mae = float(tf.reduce_mean(tf.abs(val_pred - y_val)).numpy())
            
            train_loss = np.mean(epoch_losses)
            train_mae = np.mean(epoch_maes)
            
            history_deep_mlp['loss'].append(train_loss)
            history_deep_mlp['mae'].append(train_mae)
            history_deep_mlp['val_loss'].append(val_loss)
            history_deep_mlp['val_mae'].append(val_mae)
            
            print(f"Epoch {epoch+1}/100 - loss: {train_loss:.4f} - mae: {train_mae:.4f} - val_loss: {val_loss:.4f} - val_mae: {val_mae:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_weights = [w.copy() for w in deep_mlp_model.get_weights()]
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    print(f"    Early stopping at epoch {epoch+1}")
                    deep_mlp_model.set_weights(best_weights)
                    break
    
    elapsed = time.time() - start_time
    print(f"    Training completed in {elapsed:.2f} seconds")
    
    # Predictions (optimized for GPU)
    print("    Computing predictions...")
    if USE_GPU and test_dataset is not None:
        train_pred_list = []
        for batch_X, _ in train_dataset:
            pred = predict_batch(deep_mlp_model, batch_X)
            if len(pred.shape) > 1 and pred.shape[1] == 1:
                pred = tf.squeeze(pred, axis=1)
            train_pred_list.append(pred.numpy())
        train_pred = np.concatenate(train_pred_list)
        
        print("      → Validation set predictions...")
        val_pred_list = []
        val_batches = list(val_dataset)
        if TQDM_AVAILABLE:
            val_batches = tqdm(val_batches, desc="        Validation", unit=" batch")
        for batch_X, _ in val_batches:
            pred = predict_batch(deep_mlp_model, batch_X)
            if len(pred.shape) > 1 and pred.shape[1] == 1:
                pred = tf.squeeze(pred, axis=1)
            val_pred_list.append(pred.numpy())
        val_pred = np.concatenate(val_pred_list)
        
        print("      → Test set predictions...")
        test_pred_list = []
        test_batches = list(test_dataset)
        if TQDM_AVAILABLE:
            test_batches = tqdm(test_batches, desc="        Test", unit=" batch")
        for batch_X, _ in test_batches:
            pred = predict_batch(deep_mlp_model, batch_X)
            if len(pred.shape) > 1 and pred.shape[1] == 1:
                pred = tf.squeeze(pred, axis=1)
            test_pred_list.append(pred.numpy())
        test_pred = np.concatenate(test_pred_list)
        print("      ✅ All predictions computed")
    else:
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
        
        # For mixed precision, output layer must be float32
        dtype = 'float32' if USE_GPU and tf.keras.mixed_precision.global_policy().name == 'mixed_float16' else None
        
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
        outputs = layers.Dense(1, activation='linear', dtype=dtype)(x)
        
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
    
    # Train model - optimized for GPU
    print(f"    Training on {len(X_train_scaled):,} samples, {len(X_val_scaled):,} validation samples...")
    print(f"    Batch size: {BATCH_SIZE}, Device: {'GPU' if USE_GPU else 'CPU'}")
    start_time = time.time()
    
    if USE_GPU and train_dataset is not None:
        # Build optimizer by running a dummy step (needed for mixed precision)
        print("    → Building optimizer (running dummy step)...")
        dummy_batch = next(iter(train_dataset))
        dummy_X, dummy_y = dummy_batch
        with tf.GradientTape() as tape:
            dummy_pred = residual_mlp_model(dummy_X, training=True)
            if len(dummy_pred.shape) > 1 and dummy_pred.shape[1] == 1:
                dummy_pred = tf.squeeze(dummy_pred, axis=1)
            dummy_loss = residual_mlp_model.compiled_loss(dummy_y, dummy_pred)
        dummy_grads = tape.gradient(dummy_loss, residual_mlp_model.trainable_variables)
        residual_mlp_model.optimizer.apply_gradients(zip(dummy_grads, residual_mlp_model.trainable_variables))
        print("    ✓ Optimizer built successfully")
        
        # Now define tf.function decorated functions (optimizer is already built)
        @tf.function
        def train_step(model, x, y):
            with tf.GradientTape() as tape:
                pred = model(x, training=True)
                if len(pred.shape) > 1 and pred.shape[1] == 1:
                    pred = tf.squeeze(pred, axis=1)
                loss = model.compiled_loss(y, pred)
            grads = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
            mae = tf.reduce_mean(tf.abs(pred - y))
            return loss, mae
        
        @tf.function
        def val_step(model, x, y):
            pred = model(x, training=False)
            if len(pred.shape) > 1 and pred.shape[1] == 1:
                pred = tf.squeeze(pred, axis=1)
            loss = model.compiled_loss(y, pred)
            mae = tf.reduce_mean(tf.abs(pred - y))
            return loss, mae
        
        @tf.function
        def predict_batch(model, x):
            return model(x, training=False)
        
        history_residual = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': []}
        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None
        
        num_epochs = 100
        print(f"\n  🚀 Starting training for up to {num_epochs} epochs...")
        epoch_range = range(num_epochs)
        if TQDM_AVAILABLE:
            epoch_range = tqdm(epoch_range, desc="    Training epochs", unit=" epoch")
        
        for epoch in epoch_range:
            epoch_losses = []
            epoch_maes = []
            
            for batch_X, batch_y in train_dataset:
                loss, mae = train_step(residual_mlp_model, batch_X, batch_y)
                epoch_losses.append(float(loss))
                epoch_maes.append(float(mae))
            
            val_losses = []
            val_maes = []
            for batch_X, batch_y in val_dataset:
                v_loss, v_mae = val_step(residual_mlp_model, batch_X, batch_y)
                val_losses.append(float(v_loss))
                val_maes.append(float(v_mae))
            
            train_loss = np.mean(epoch_losses)
            train_mae = np.mean(epoch_maes)
            val_loss = np.mean(val_losses)
            val_mae = np.mean(val_maes)
            
            history_residual['loss'].append(train_loss)
            history_residual['mae'].append(train_mae)
            history_residual['val_loss'].append(val_loss)
            history_residual['val_mae'].append(val_mae)
            
            if TQDM_AVAILABLE and hasattr(epoch_range, 'set_description'):
                epoch_range.set_description(
                    f"Epoch {epoch+1} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f}"
                )
            else:
                print(f"Epoch {epoch+1}/100 - loss: {train_loss:.4f} - mae: {train_mae:.4f} - val_loss: {val_loss:.4f} - val_mae: {val_mae:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_weights = [w.copy() for w in residual_mlp_model.get_weights()]
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    print(f"    Early stopping at epoch {epoch+1}")
                    residual_mlp_model.set_weights(best_weights)
                    break
    else:
        # Fallback to custom loop
        history_residual = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': []}
        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None
        
        for epoch in range(100):
            epoch_losses = []
            epoch_maes = []
            
            for i in range(0, len(X_train_scaled), BATCH_SIZE):
                batch_X = X_train_scaled[i:i+BATCH_SIZE]
                batch_y = y_train[i:i+BATCH_SIZE]
                
                with tf.GradientTape() as tape:
                    pred = residual_mlp_model(batch_X, training=True)
                    if len(pred.shape) > 1 and pred.shape[1] == 1:
                        pred = tf.squeeze(pred, axis=1)
                    loss = residual_mlp_model.compiled_loss(batch_y, pred)
                
                grads = tape.gradient(loss, residual_mlp_model.trainable_variables)
                residual_mlp_model.optimizer.apply_gradients(zip(grads, residual_mlp_model.trainable_variables))
                
                epoch_losses.append(float(loss.numpy()))
                mae = tf.reduce_mean(tf.abs(pred - batch_y))
                epoch_maes.append(float(mae.numpy()))
            
            val_pred = residual_mlp_model(X_val_scaled, training=False)
            if len(val_pred.shape) > 1 and val_pred.shape[1] == 1:
                val_pred = tf.squeeze(val_pred, axis=1)
            val_loss = float(residual_mlp_model.compiled_loss(y_val, val_pred).numpy())
            val_mae = float(tf.reduce_mean(tf.abs(val_pred - y_val)).numpy())
            
            train_loss = np.mean(epoch_losses)
            train_mae = np.mean(epoch_maes)
            
            history_residual['loss'].append(train_loss)
            history_residual['mae'].append(train_mae)
            history_residual['val_loss'].append(val_loss)
            history_residual['val_mae'].append(val_mae)
            
            print(f"Epoch {epoch+1}/100 - loss: {train_loss:.4f} - mae: {train_mae:.4f} - val_loss: {val_loss:.4f} - val_mae: {val_mae:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_weights = [w.copy() for w in residual_mlp_model.get_weights()]
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    print(f"    Early stopping at epoch {epoch+1}")
                    residual_mlp_model.set_weights(best_weights)
                    break
    
    elapsed = time.time() - start_time
    print(f"    Training completed in {elapsed:.2f} seconds")
    
    # Predictions (optimized for GPU)
    print("\n    📊 Computing predictions...")
    if USE_GPU and test_dataset is not None:
        print("      → Training set predictions...")
        train_pred_list = []
        train_batches = list(train_dataset)
        if TQDM_AVAILABLE:
            train_batches = tqdm(train_batches, desc="        Training", unit=" batch")
        for batch_X, _ in train_batches:
            pred = predict_batch(residual_mlp_model, batch_X)
            if len(pred.shape) > 1 and pred.shape[1] == 1:
                pred = tf.squeeze(pred, axis=1)
            train_pred_list.append(pred.numpy())
        train_pred = np.concatenate(train_pred_list)
        
        print("      → Validation set predictions...")
        val_pred_list = []
        val_batches = list(val_dataset)
        if TQDM_AVAILABLE:
            val_batches = tqdm(val_batches, desc="        Validation", unit=" batch")
        for batch_X, _ in val_batches:
            pred = predict_batch(residual_mlp_model, batch_X)
            if len(pred.shape) > 1 and pred.shape[1] == 1:
                pred = tf.squeeze(pred, axis=1)
            val_pred_list.append(pred.numpy())
        val_pred = np.concatenate(val_pred_list)
        
        print("      → Test set predictions...")
        test_pred_list = []
        test_batches = list(test_dataset)
        if TQDM_AVAILABLE:
            test_batches = tqdm(test_batches, desc="        Test", unit=" batch")
        for batch_X, _ in test_batches:
            pred = predict_batch(residual_mlp_model, batch_X)
            if len(pred.shape) > 1 and pred.shape[1] == 1:
                pred = tf.squeeze(pred, axis=1)
            test_pred_list.append(pred.numpy())
        test_pred = np.concatenate(test_pred_list)
        print("      ✅ All predictions computed")
    else:
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
    
    # 1d. Transformer with Cross-Attention (Advanced Model)
    print("\n  1d. Training Transformer with Cross-Attention (advanced)...")
    
    # Check if we have embeddings available for cross-attention
    use_cross_attention = 'hindi_embeddings' in locals() and 'chinese_embeddings' in locals()
    
    if use_cross_attention and len(hindi_embeddings) > 0:
        # Create cross-attention model using Hindi and Chinese embeddings directly
        print("    → Using cross-attention with Hindi-Chinese embeddings...")
        
        def create_cross_attention_transformer(embedding_dim=384, num_heads=8, num_layers=2, d_model=256):
            """Transformer with cross-attention between Hindi and Chinese embeddings"""
            l2_reg = tf.keras.regularizers.L2(0.01)
            dtype = 'float32' if USE_GPU and tf.keras.mixed_precision.global_policy().name == 'mixed_float16' else None
            
            # Two inputs: Hindi embeddings and Chinese embeddings
            hindi_input = layers.Input(shape=(embedding_dim,), name='hindi_emb')
            chinese_input = layers.Input(shape=(embedding_dim,), name='chinese_emb')
            
            # Project embeddings to d_model
            hindi_proj = layers.Dense(d_model, activation='relu', kernel_regularizer=l2_reg)(hindi_input)
            hindi_proj = layers.BatchNormalization()(hindi_proj)
            hindi_proj = layers.Dropout(0.3)(hindi_proj)
            hindi_proj = layers.Reshape((1, d_model))(hindi_proj)
            
            chinese_proj = layers.Dense(d_model, activation='relu', kernel_regularizer=l2_reg)(chinese_input)
            chinese_proj = layers.BatchNormalization()(chinese_proj)
            chinese_proj = layers.Dropout(0.3)(chinese_proj)
            chinese_proj = layers.Reshape((1, d_model))(chinese_proj)
            
            # Cross-attention layers: Hindi attends to Chinese, and vice versa
            for i in range(num_layers):
                # Hindi-to-Chinese cross-attention
                hi_to_zh = layers.MultiHeadAttention(
                    num_heads=num_heads,
                    key_dim=d_model // num_heads,
                    dropout=0.1,
                    kernel_regularizer=l2_reg
                )(hindi_proj, chinese_proj)  # Query from Hindi, Key/Value from Chinese
                hi_to_zh = layers.Dropout(0.2)(hi_to_zh)
                hindi_proj = layers.Add()([hindi_proj, hi_to_zh])
                hindi_proj = layers.LayerNormalization()(hindi_proj)
                
                # Chinese-to-Hindi cross-attention
                zh_to_hi = layers.MultiHeadAttention(
                    num_heads=num_heads,
                    key_dim=d_model // num_heads,
                    dropout=0.1,
                    kernel_regularizer=l2_reg
                )(chinese_proj, hindi_proj)  # Query from Chinese, Key/Value from Hindi
                zh_to_hi = layers.Dropout(0.2)(zh_to_hi)
                chinese_proj = layers.Add()([chinese_proj, zh_to_hi])
                chinese_proj = layers.LayerNormalization()(chinese_proj)
                
                # Feed-forward for Hindi
                hi_ffn = layers.Dense(d_model * 2, activation='relu', kernel_regularizer=l2_reg)(hindi_proj)
                hi_ffn = layers.Dropout(0.2)(hi_ffn)
                hi_ffn = layers.Dense(d_model, kernel_regularizer=l2_reg)(hi_ffn)
                hindi_proj = layers.Add()([hindi_proj, hi_ffn])
                hindi_proj = layers.LayerNormalization()(hindi_proj)
                
                # Feed-forward for Chinese
                zh_ffn = layers.Dense(d_model * 2, activation='relu', kernel_regularizer=l2_reg)(chinese_proj)
                zh_ffn = layers.Dropout(0.2)(zh_ffn)
                zh_ffn = layers.Dense(d_model, kernel_regularizer=l2_reg)(zh_ffn)
                chinese_proj = layers.Add()([chinese_proj, zh_ffn])
                chinese_proj = layers.LayerNormalization()(chinese_proj)
            
            # Combine Hindi and Chinese representations
            combined = layers.Concatenate(axis=-1)([layers.Flatten()(hindi_proj), layers.Flatten()(chinese_proj)])
            
            # Final dense layers
            x = layers.Dense(256, activation='relu', kernel_regularizer=l2_reg)(combined)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.4)(x)
            
            x = layers.Dense(128, activation='relu', kernel_regularizer=l2_reg)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)
            
            x = layers.Dense(64, activation='relu', kernel_regularizer=l2_reg)(x)
            x = layers.Dropout(0.2)(x)
            
            outputs = layers.Dense(1, activation='linear', dtype=dtype)(x)
            
            model = models.Model(inputs=[hindi_input, chinese_input], outputs=outputs)
            return model
        
        # Prepare embedding inputs
        embedding_dim = hindi_embeddings.shape[1] if len(hindi_embeddings.shape) > 1 else 384
        
        # Split embeddings using the same indices as features
        # train_indices, val_indices, test_indices are defined in the train/test split section
        hindi_emb_train = hindi_embeddings[train_indices]
        hindi_emb_val = hindi_embeddings[val_indices]
        hindi_emb_test = hindi_embeddings[test_indices]
        
        chinese_emb_train = chinese_embeddings[train_indices]
        chinese_emb_val = chinese_embeddings[val_indices]
        chinese_emb_test = chinese_embeddings[test_indices]
        
        print(f"    ✓ Split embeddings: Train={len(hindi_emb_train)}, Val={len(hindi_emb_val)}, Test={len(hindi_emb_test)}")
        
        transformer_model = create_cross_attention_transformer(embedding_dim=embedding_dim, num_heads=8, num_layers=2, d_model=256)
        transformer_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        if hasattr(transformer_model.optimizer, 'learning_rate'):
            transformer_model.optimizer.learning_rate.assign(0.0001)
        
        print(f"    Training on {len(X_train_scaled):,} samples with cross-attention...")
        print(f"    Embedding dimension: {embedding_dim}, Attention heads: 8, Layers: 2")
        start_time = time.time()
        
        if USE_GPU:
            # Create datasets with embeddings
            train_emb_dataset = tf.data.Dataset.from_tensor_slices(
                ((hindi_emb_train.astype(np.float32), chinese_emb_train.astype(np.float32)), 
                 y_train.astype(np.float32))
            ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
            
            val_emb_dataset = tf.data.Dataset.from_tensor_slices(
                ((hindi_emb_val.astype(np.float32), chinese_emb_val.astype(np.float32)), 
                 y_val.astype(np.float32))
            ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
            
            test_emb_dataset = tf.data.Dataset.from_tensor_slices(
                ((hindi_emb_test.astype(np.float32), chinese_emb_test.astype(np.float32)), 
                 y_test.astype(np.float32))
            ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
            
            # Build optimizer
            print("    → Building optimizer...")
            dummy_batch = next(iter(train_emb_dataset))
            (dummy_hi, dummy_zh), dummy_y = dummy_batch
            with tf.GradientTape() as tape:
                dummy_pred = transformer_model([dummy_hi, dummy_zh], training=True)
                if len(dummy_pred.shape) > 1 and dummy_pred.shape[1] == 1:
                    dummy_pred = tf.squeeze(dummy_pred, axis=1)
                dummy_loss = transformer_model.compiled_loss(dummy_y, dummy_pred)
            dummy_grads = tape.gradient(dummy_loss, transformer_model.trainable_variables)
            transformer_model.optimizer.apply_gradients(zip(dummy_grads, transformer_model.trainable_variables))
            print("    ✓ Optimizer built")
            
            @tf.function
            def train_step(model, hi_emb, zh_emb, y):
                with tf.GradientTape() as tape:
                    pred = model([hi_emb, zh_emb], training=True)
                    if len(pred.shape) > 1 and pred.shape[1] == 1:
                        pred = tf.squeeze(pred, axis=1)
                    loss = model.compiled_loss(y, pred)
                grads = tape.gradient(loss, model.trainable_variables)
                model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
                mae = tf.reduce_mean(tf.abs(pred - y))
                return loss, mae
            
            @tf.function
            def val_step(model, hi_emb, zh_emb, y):
                pred = model([hi_emb, zh_emb], training=False)
                if len(pred.shape) > 1 and pred.shape[1] == 1:
                    pred = tf.squeeze(pred, axis=1)
                loss = model.compiled_loss(y, pred)
                mae = tf.reduce_mean(tf.abs(pred - y))
                return loss, mae
            
            history_transformer = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': []}
            best_val_loss = float('inf')
            patience_counter = 0
            best_weights = None
            
            num_epochs = 100
            print(f"\n  🚀 Starting training for up to {num_epochs} epochs...")
            epoch_range = range(num_epochs)
            if TQDM_AVAILABLE:
                epoch_range = tqdm(epoch_range, desc="    Training epochs", unit=" epoch")
            
            for epoch in epoch_range:
                epoch_losses = []
                epoch_maes = []
                
                for (hi_emb, zh_emb), batch_y in train_emb_dataset:
                    loss, mae = train_step(transformer_model, hi_emb, zh_emb, batch_y)
                    epoch_losses.append(float(loss))
                    epoch_maes.append(float(mae))
                
                val_losses = []
                val_maes = []
                for (hi_emb, zh_emb), batch_y in val_emb_dataset:
                    v_loss, v_mae = val_step(transformer_model, hi_emb, zh_emb, batch_y)
                    val_losses.append(float(v_loss))
                    val_maes.append(float(v_mae))
                
                train_loss = np.mean(epoch_losses)
                train_mae = np.mean(epoch_maes)
                val_loss = np.mean(val_losses)
                val_mae = np.mean(val_maes)
                
                history_transformer['loss'].append(train_loss)
                history_transformer['mae'].append(train_mae)
                history_transformer['val_loss'].append(val_loss)
                history_transformer['val_mae'].append(val_mae)
                
                if TQDM_AVAILABLE and hasattr(epoch_range, 'set_description'):
                    epoch_range.set_description(f"Epoch {epoch+1} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")
                else:
                    print(f"Epoch {epoch+1}/100 - loss: {train_loss:.4f} - mae: {train_mae:.4f} - val_loss: {val_loss:.4f} - val_mae: {val_mae:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_weights = [w.copy() for w in transformer_model.get_weights()]
                else:
                    patience_counter += 1
                    if patience_counter >= 5:
                        print(f"    ⏹️  Early stopping at epoch {epoch+1}")
                        transformer_model.set_weights(best_weights)
                        break
            
            elapsed = time.time() - start_time
            print(f"    Training completed in {elapsed:.2f} seconds")
            
            # Predictions
            print("\n    📊 Computing predictions...")
            train_pred_list = []
            for (hi_emb, zh_emb), _ in train_emb_dataset:
                pred = transformer_model([hi_emb, zh_emb], training=False).numpy()
                if len(pred.shape) > 1 and pred.shape[1] == 1:
                    pred = tf.squeeze(pred, axis=1)
                train_pred_list.append(pred)
            train_pred = np.concatenate(train_pred_list)
            
            val_pred_list = []
            for (hi_emb, zh_emb), _ in val_emb_dataset:
                pred = transformer_model([hi_emb, zh_emb], training=False).numpy()
                if len(pred.shape) > 1 and pred.shape[1] == 1:
                    pred = tf.squeeze(pred, axis=1)
                val_pred_list.append(pred)
            val_pred = np.concatenate(val_pred_list)
            
            test_pred_list = []
            for (hi_emb, zh_emb), _ in test_emb_dataset:
                pred = transformer_model([hi_emb, zh_emb], training=False).numpy()
                if len(pred.shape) > 1 and pred.shape[1] == 1:
                    pred = tf.squeeze(pred, axis=1)
                test_pred_list.append(pred)
            test_pred = np.concatenate(test_pred_list)
            print("      ✅ All predictions computed")
        else:
            # CPU fallback
            history_transformer = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': []}
            best_val_loss = float('inf')
            patience_counter = 0
            best_weights = None
            
            for epoch in range(100):
                epoch_losses = []
                epoch_maes = []
                
                for i in range(0, len(hindi_emb_train), BATCH_SIZE):
                    hi_batch = hindi_emb_train[i:i+BATCH_SIZE]
                    zh_batch = chinese_emb_train[i:i+BATCH_SIZE]
                    y_batch = y_train[i:i+BATCH_SIZE]
                    
                    with tf.GradientTape() as tape:
                        pred = transformer_model([hi_batch, zh_batch], training=True)
                        if len(pred.shape) > 1 and pred.shape[1] == 1:
                            pred = tf.squeeze(pred, axis=1)
                        loss = transformer_model.compiled_loss(y_batch, pred)
                    
                    grads = tape.gradient(loss, transformer_model.trainable_variables)
                    transformer_model.optimizer.apply_gradients(zip(grads, transformer_model.trainable_variables))
                    
                    epoch_losses.append(float(loss.numpy()))
                    mae = tf.reduce_mean(tf.abs(pred - y_batch))
                    epoch_maes.append(float(mae.numpy()))
                
                val_pred = transformer_model([hindi_emb_val, chinese_emb_val], training=False)
                if len(val_pred.shape) > 1 and val_pred.shape[1] == 1:
                    val_pred = tf.squeeze(val_pred, axis=1)
                val_loss = float(transformer_model.compiled_loss(y_val, val_pred).numpy())
                val_mae = float(tf.reduce_mean(tf.abs(val_pred - y_val)).numpy())
                
                train_loss = np.mean(epoch_losses)
                train_mae = np.mean(epoch_maes)
                
                history_transformer['loss'].append(train_loss)
                history_transformer['mae'].append(train_mae)
                history_transformer['val_loss'].append(val_loss)
                history_transformer['val_mae'].append(val_mae)
                
                print(f"Epoch {epoch+1}/100 - loss: {train_loss:.4f} - mae: {train_mae:.4f} - val_loss: {val_loss:.4f} - val_mae: {val_mae:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_weights = [w.copy() for w in transformer_model.get_weights()]
                else:
                    patience_counter += 1
                    if patience_counter >= 5:
                        print(f"    ⏹️  Early stopping at epoch {epoch+1}")
                        transformer_model.set_weights(best_weights)
                        break
            
            elapsed = time.time() - start_time
            print(f"    Training completed in {elapsed:.2f} seconds")
            
            train_pred = transformer_model([hindi_emb_train, chinese_emb_train], training=False).numpy()
            val_pred = transformer_model([hindi_emb_val, chinese_emb_val], training=False).numpy()
            test_pred = transformer_model([hindi_emb_test, chinese_emb_test], training=False).numpy()
            
            if len(train_pred.shape) > 1 and train_pred.shape[1] == 1:
                train_pred = train_pred.squeeze(axis=1)
            if len(val_pred.shape) > 1 and val_pred.shape[1] == 1:
                val_pred = val_pred.squeeze(axis=1)
            if len(test_pred.shape) > 1 and test_pred.shape[1] == 1:
                test_pred = test_pred.squeeze(axis=1)
        
        # Evaluation
        results['transformer'] = {
            'train_mae': mean_absolute_error(y_train, train_pred),
            'val_mae': mean_absolute_error(y_val, val_pred),
            'test_mae': mean_absolute_error(y_test, test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'train_r2': r2_score(y_train, train_pred),
            'val_r2': r2_score(y_val, val_pred),
            'test_r2': r2_score(y_test, test_pred),
            'test_spearman': spearmanr(y_test, test_pred)[0],
            'test_pearson': pearsonr(y_test, test_pred)[0],
        }
        
        trained_models['transformer'] = transformer_model
        
        print(f"    Train MAE: {results['transformer']['train_mae']:.4f}, R²: {results['transformer']['train_r2']:.4f}")
        print(f"    Val MAE: {results['transformer']['val_mae']:.4f}, R²: {results['transformer']['val_r2']:.4f}")
        print(f"    Test MAE: {results['transformer']['test_mae']:.4f}, RMSE: {results['transformer']['test_rmse']:.4f}")
        print(f"    Test R²: {results['transformer']['test_r2']:.4f}, Spearman: {results['transformer']['test_spearman']:.4f}")
    else:
        # Fallback to feature-based transformer if embeddings not available
        print("    → Using feature-based transformer (embeddings not available)...")
        
        def create_transformer_model(input_dim, num_heads=8, num_layers=2, d_model=128):
            """
            Transformer model with multi-head self-attention for feature interactions.
            Uses attention to capture complex feature relationships.
            """
            l2_reg = tf.keras.regularizers.L2(0.01)
            dtype = 'float32' if USE_GPU and tf.keras.mixed_precision.global_policy().name == 'mixed_float16' else None
            
            inputs = layers.Input(shape=(input_dim,))
            
            # Project input to d_model dimensions and add positional encoding
            x = layers.Dense(d_model, activation='relu', kernel_regularizer=l2_reg)(inputs)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)
            
            # Reshape for attention: treat features as sequence (batch, seq_len=1, d_model)
            # We'll expand to multiple "tokens" by splitting features into chunks for better attention
            # Alternative: use self-attention on the feature vector itself
            x_expanded = layers.Reshape((1, d_model))(x)
            
            # Multi-head self-attention layers with residual connections
            for i in range(num_layers):
                # Self-attention (query, key, value all from same input)
                attn_output = layers.MultiHeadAttention(
                    num_heads=num_heads, 
                    key_dim=d_model // num_heads,
                    dropout=0.1,
                    kernel_regularizer=l2_reg
                )(x_expanded, x_expanded)
                attn_output = layers.Dropout(0.2)(attn_output)
                
                # Residual connection & Layer Normalization
                x_expanded = layers.Add()([x_expanded, attn_output])
                x_expanded = layers.LayerNormalization()(x_expanded)
                
                # Feed-forward network (2-layer MLP)
                ffn = layers.Dense(d_model * 2, activation='relu', kernel_regularizer=l2_reg)(x_expanded)
                ffn = layers.Dropout(0.2)(ffn)
                ffn = layers.Dense(d_model, kernel_regularizer=l2_reg)(ffn)
                ffn = layers.Dropout(0.2)(ffn)
                
                # Residual connection & Layer Normalization
                x_expanded = layers.Add()([x_expanded, ffn])
                x_expanded = layers.LayerNormalization()(x_expanded)
            
            # Flatten back to (batch, d_model)
            x = layers.Flatten()(x_expanded)
            
            # Final classification head with residual connections
            # First dense block
            x1 = layers.Dense(128, activation='relu', kernel_regularizer=l2_reg)(x)
            x1 = layers.BatchNormalization()(x1)
            x1 = layers.Dropout(0.4)(x1)
            
            # Projection for residual
            x_proj = layers.Dense(128, kernel_regularizer=l2_reg)(x)
            x = layers.Add()([x1, x_proj])
            x = layers.Activation('relu')(x)
            x = layers.Dropout(0.3)(x)
            
            # Second dense block
            x2 = layers.Dense(64, activation='relu', kernel_regularizer=l2_reg)(x)
            x2 = layers.BatchNormalization()(x2)
            x2 = layers.Dropout(0.3)(x2)
            
            x_proj2 = layers.Dense(64, kernel_regularizer=l2_reg)(x)
            x = layers.Add()([x2, x_proj2])
            x = layers.Activation('relu')(x)
            x = layers.Dropout(0.2)(x)
            
            # Final layer
            x = layers.Dense(32, activation='relu', kernel_regularizer=l2_reg)(x)
            x = layers.Dropout(0.2)(x)
            
            # Output layer
            outputs = layers.Dense(1, activation='linear', dtype=dtype)(x)
            
            model = models.Model(inputs=inputs, outputs=outputs)
            return model
        
        transformer_model = create_transformer_model(input_dim, num_heads=8, num_layers=2, d_model=128)
        transformer_model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        if hasattr(transformer_model.optimizer, 'learning_rate'):
            transformer_model.optimizer.learning_rate.assign(0.0001)
        
        print(f"    Training on {len(X_train_scaled):,} samples, {len(X_val_scaled):,} validation samples...")
        print(f"    Batch size: {BATCH_SIZE}, Device: {'GPU' if USE_GPU else 'CPU'}")
        print(f"    Model: Transformer with {8} attention heads, {2} layers")
        start_time = time.time()
        
        if USE_GPU and train_dataset is not None:
            # Build optimizer
            print("    → Building optimizer (running dummy step)...")
            dummy_batch = next(iter(train_dataset))
            dummy_X, dummy_y = dummy_batch
            with tf.GradientTape() as tape:
                dummy_pred = transformer_model(dummy_X, training=True)
                if len(dummy_pred.shape) > 1 and dummy_pred.shape[1] == 1:
                    dummy_pred = tf.squeeze(dummy_pred, axis=1)
                dummy_loss = transformer_model.compiled_loss(dummy_y, dummy_pred)
            dummy_grads = tape.gradient(dummy_loss, transformer_model.trainable_variables)
            transformer_model.optimizer.apply_gradients(zip(dummy_grads, transformer_model.trainable_variables))
            print("    ✓ Optimizer built successfully")
            
            # Define tf.function decorated functions
            @tf.function
            def train_step(model, x, y):
                with tf.GradientTape() as tape:
                    pred = model(x, training=True)
                    if len(pred.shape) > 1 and pred.shape[1] == 1:
                        pred = tf.squeeze(pred, axis=1)
                    loss = model.compiled_loss(y, pred)
                grads = tape.gradient(loss, model.trainable_variables)
                model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
                mae = tf.reduce_mean(tf.abs(pred - y))
                return loss, mae
            
            @tf.function
            def val_step(model, x, y):
                pred = model(x, training=False)
                if len(pred.shape) > 1 and pred.shape[1] == 1:
                    pred = tf.squeeze(pred, axis=1)
                loss = model.compiled_loss(y, pred)
                mae = tf.reduce_mean(tf.abs(pred - y))
                return loss, mae
            
            @tf.function
            def predict_batch(model, x):
                return model(x, training=False)
            
            history_transformer = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': []}
            best_val_loss = float('inf')
            patience_counter = 0
            best_weights = None
            
            num_epochs = 100
            print(f"\n  🚀 Starting training for up to {num_epochs} epochs...")
            epoch_range = range(num_epochs)
            if TQDM_AVAILABLE:
                epoch_range = tqdm(epoch_range, desc="    Training epochs", unit=" epoch")
            
            for epoch in epoch_range:
                epoch_losses = []
                epoch_maes = []
                
                for batch_X, batch_y in train_dataset:
                    loss, mae = train_step(transformer_model, batch_X, batch_y)
                    epoch_losses.append(float(loss))
                    epoch_maes.append(float(mae))
                
                val_losses = []
                val_maes = []
                for batch_X, batch_y in val_dataset:
                    v_loss, v_mae = val_step(transformer_model, batch_X, batch_y)
                    val_losses.append(float(v_loss))
                    val_maes.append(float(v_mae))
                
                train_loss = np.mean(epoch_losses)
                train_mae = np.mean(epoch_maes)
                val_loss = np.mean(val_losses)
                val_mae = np.mean(val_maes)
                
                history_transformer['loss'].append(train_loss)
                history_transformer['mae'].append(train_mae)
                history_transformer['val_loss'].append(val_loss)
                history_transformer['val_mae'].append(val_mae)
                
                if TQDM_AVAILABLE and hasattr(epoch_range, 'set_description'):
                    epoch_range.set_description(
                        f"Epoch {epoch+1} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f}"
                    )
                else:
                    print(f"Epoch {epoch+1}/100 - loss: {train_loss:.4f} - mae: {train_mae:.4f} - val_loss: {val_loss:.4f} - val_mae: {val_mae:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_weights = [w.copy() for w in transformer_model.get_weights()]
                else:
                    patience_counter += 1
                    if patience_counter >= 5:
                        print(f"    ⏹️  Early stopping at epoch {epoch+1}")
                        transformer_model.set_weights(best_weights)
                        break
            
            elapsed = time.time() - start_time
            print(f"    Training completed in {elapsed:.2f} seconds")
            
            # Predictions
            print("\n    📊 Computing predictions...")
            print("      → Training set predictions...")
            train_pred_list = []
            train_batches = list(train_dataset)
            if TQDM_AVAILABLE:
                train_batches = tqdm(train_batches, desc="        Training", unit=" batch")
            for batch_X, _ in train_batches:
                pred = predict_batch(transformer_model, batch_X)
                if len(pred.shape) > 1 and pred.shape[1] == 1:
                    pred = tf.squeeze(pred, axis=1)
                train_pred_list.append(pred.numpy())
            train_pred = np.concatenate(train_pred_list)
            
            print("      → Validation set predictions...")
            val_pred_list = []
            val_batches = list(val_dataset)
            if TQDM_AVAILABLE:
                val_batches = tqdm(val_batches, desc="        Validation", unit=" batch")
            for batch_X, _ in val_batches:
                pred = predict_batch(transformer_model, batch_X)
                if len(pred.shape) > 1 and pred.shape[1] == 1:
                    pred = tf.squeeze(pred, axis=1)
                val_pred_list.append(pred.numpy())
            val_pred = np.concatenate(val_pred_list)
            
            print("      → Test set predictions...")
            test_pred_list = []
            test_batches = list(test_dataset)
            if TQDM_AVAILABLE:
                test_batches = tqdm(test_batches, desc="        Test", unit=" batch")
            for batch_X, _ in test_batches:
                pred = predict_batch(transformer_model, batch_X)
                if len(pred.shape) > 1 and pred.shape[1] == 1:
                    pred = tf.squeeze(pred, axis=1)
                test_pred_list.append(pred.numpy())
            test_pred = np.concatenate(test_pred_list)
            print("      ✅ All predictions computed")
        else:
            # Fallback to custom loop
            history_transformer = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': []}
            best_val_loss = float('inf')
            patience_counter = 0
            best_weights = None
            
            for epoch in range(100):
                epoch_losses = []
                epoch_maes = []
                
                for i in range(0, len(X_train_scaled), BATCH_SIZE):
                    batch_X = X_train_scaled[i:i+BATCH_SIZE]
                    batch_y = y_train[i:i+BATCH_SIZE]
                    
                    with tf.GradientTape() as tape:
                        pred = transformer_model(batch_X, training=True)
                        if len(pred.shape) > 1 and pred.shape[1] == 1:
                            pred = tf.squeeze(pred, axis=1)
                        loss = transformer_model.compiled_loss(batch_y, pred)
                    
                    grads = tape.gradient(loss, transformer_model.trainable_variables)
                    transformer_model.optimizer.apply_gradients(zip(grads, transformer_model.trainable_variables))
                    
                    epoch_losses.append(float(loss.numpy()))
                    mae = tf.reduce_mean(tf.abs(pred - batch_y))
                    epoch_maes.append(float(mae.numpy()))
                
                val_pred = transformer_model(X_val_scaled, training=False)
                if len(val_pred.shape) > 1 and val_pred.shape[1] == 1:
                    val_pred = tf.squeeze(val_pred, axis=1)
                val_loss = float(transformer_model.compiled_loss(y_val, val_pred).numpy())
                val_mae = float(tf.reduce_mean(tf.abs(val_pred - y_val)).numpy())
                
                train_loss = np.mean(epoch_losses)
                train_mae = np.mean(epoch_maes)
                
                history_transformer['loss'].append(train_loss)
                history_transformer['mae'].append(train_mae)
                history_transformer['val_loss'].append(val_loss)
                history_transformer['val_mae'].append(val_mae)
                
                print(f"Epoch {epoch+1}/100 - loss: {train_loss:.4f} - mae: {train_mae:.4f} - val_loss: {val_loss:.4f} - val_mae: {val_mae:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_weights = [w.copy() for w in transformer_model.get_weights()]
                else:
                    patience_counter += 1
                    if patience_counter >= 5:
                        print(f"    ⏹️  Early stopping at epoch {epoch+1}")
                        transformer_model.set_weights(best_weights)
                        break
            
            elapsed = time.time() - start_time
            print(f"    Training completed in {elapsed:.2f} seconds")
            
            train_pred = transformer_model(X_train_scaled, training=False).numpy()
            val_pred = transformer_model(X_val_scaled, training=False).numpy()
            test_pred = transformer_model(X_test_scaled, training=False).numpy()
            
            # Squeeze if needed
            if len(train_pred.shape) > 1 and train_pred.shape[1] == 1:
                train_pred = train_pred.squeeze(axis=1)
            if len(val_pred.shape) > 1 and val_pred.shape[1] == 1:
                val_pred = val_pred.squeeze(axis=1)
            if len(test_pred.shape) > 1 and test_pred.shape[1] == 1:
                test_pred = test_pred.squeeze(axis=1)
        
        y_train_pred_trans = train_pred
        y_val_pred_trans = val_pred
        y_test_pred_trans = test_pred
        
        # Evaluation
        results['transformer'] = {
            'train_mae': mean_absolute_error(y_train, y_train_pred_trans),
            'val_mae': mean_absolute_error(y_val, y_val_pred_trans),
            'test_mae': mean_absolute_error(y_test, y_test_pred_trans),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred_trans)),
            'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred_trans)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred_trans)),
            'train_r2': r2_score(y_train, y_train_pred_trans),
            'val_r2': r2_score(y_val, y_val_pred_trans),
            'test_r2': r2_score(y_test, y_test_pred_trans),
            'test_spearman': spearmanr(y_test, y_test_pred_trans)[0],
            'test_pearson': pearsonr(y_test, y_test_pred_trans)[0],
        }
        
        trained_models['transformer'] = transformer_model
        
        print(f"    Train MAE: {results['transformer']['train_mae']:.4f}, R²: {results['transformer']['train_r2']:.4f}")
        print(f"    Val MAE: {results['transformer']['val_mae']:.4f}, R²: {results['transformer']['val_r2']:.4f}")
        print(f"    Test MAE: {results['transformer']['test_mae']:.4f}, RMSE: {results['transformer']['test_rmse']:.4f}")
        print(f"    Test R²: {results['transformer']['test_r2']:.4f}, Spearman: {results['transformer']['test_spearman']:.4f}")
    
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
    # For neural networks, use optimized prediction
    print("Computing final predictions with best model...")
    if TF_AVAILABLE and USE_GPU and test_dataset is not None:
        # Use tf.data for faster GPU prediction
        test_pred_list = []
        for batch_X, _ in test_dataset:
            pred = predict_batch(best_model, batch_X)
            if len(pred.shape) > 1 and pred.shape[1] == 1:
                pred = tf.squeeze(pred, axis=1)
            test_pred_list.append(pred.numpy())
        y_test_pred_scaled = np.concatenate(test_pred_list)
    else:
        # Fallback for CPU
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
print("💾 SAVING MODELS AND ARTIFACTS")
print("=" * 70)

# Save best model
print("  → Saving best model...")
if best_model_name in ['mlp', 'deep_mlp', 'residual_mlp', 'transformer']:
    # Save Keras model
    # Save weights and architecture separately to avoid loss/metric deserialization issues
    model_path = os.path.join(OUTPUT_DIR, f'quality_estimation_{best_model_name}.h5')
    
    # Save model architecture as JSON
    model_json = best_model.to_json()
    model_json_path = os.path.join(OUTPUT_DIR, f'quality_estimation_{best_model_name}_architecture.json')
    with open(model_json_path, 'w') as f:
        f.write(model_json)
    
    # Save weights only (Keras requires filename to end in .weights.h5)
    weights_path = os.path.join(OUTPUT_DIR, f'quality_estimation_{best_model_name}.weights.h5')
    best_model.save_weights(weights_path)
    
    # Also save full model for backward compatibility (but this may have issues)
    try:
        best_model.save(model_path, save_format='h5')
        print(f"    ✅ Saved model (full): {model_path}")
    except Exception as e:
        print(f"    ⚠️  Could not save full model (will use weights + architecture): {e}")
    
    print(f"    ✅ Saved architecture: {model_json_path}")
    print(f"    ✅ Saved weights: {weights_path}")
else:
    # Save sklearn model with pickle
    model_path = os.path.join(OUTPUT_DIR, f'quality_estimation_{best_model_name}.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"    ✅ Saved model: {model_path}")

# Save scaler
print("  → Saving scalers...")
scaler_path = os.path.join(OUTPUT_DIR, 'scaler.pkl')
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"    ✅ Saved scaler: {scaler_path}")

# Save target scaler (for inverse transform during prediction)
target_scaler_path = os.path.join(OUTPUT_DIR, 'target_scaler.pkl')
with open(target_scaler_path, 'wb') as f:
    pickle.dump(target_scaler, f)
print(f"    ✅ Saved target scaler: {target_scaler_path}")

# Note: No label encoder needed for regression (continuous target)

# Save feature list
print("  → Saving feature list...")
feature_path = os.path.join(OUTPUT_DIR, 'feature_columns.pkl')
with open(feature_path, 'wb') as f:
    pickle.dump(feature_columns, f)
print(f"    ✅ Saved feature columns: {feature_path}")

# Save results
print("  → Saving training results...")
results_path = os.path.join(OUTPUT_DIR, 'training_results.pkl')
with open(results_path, 'wb') as f:
    pickle.dump(results, f)
print(f"    ✅ Saved results: {results_path}")

# Save summary
print("  → Saving training summary...")
summary_path = os.path.join(OUTPUT_DIR, 'training_summary.txt')
with open(summary_path, 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("QUALITY ESTIMATION MODEL TRAINING SUMMARY\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Dataset size: {len(df):,} pairs\n")
    f.write(f"Sample size: {SAMPLE_SIZE if SAMPLE_SIZE else 'Full dataset'}\n")
    f.write(f"Features: {len(feature_columns)}\n")
    f.write(f"Train/Val/Test split: {len(X_train):,} / {len(X_val):,} / {len(X_test):,}\n\n")
    f.write("Model Results:\n")
    for model_name, result in results.items():
        f.write(f"\n{model_name}:\n")
        f.write(f"  Test MAE: {result['test_mae']:.4f}\n")
        f.write(f"  Test RMSE: {result['test_rmse']:.4f}\n")
        f.write(f"  Test R²: {result['test_r2']:.4f}\n")
        # Handle potential None values in test_spearman
        spearman_val = result.get('test_spearman', 'N/A')
        if spearman_val is not None:
            f.write(f"  Test Spearman: {spearman_val:.4f}\n")
        else:
            f.write(f"  Test Spearman: N/A\n")
    f.write(f"\nBest Model: {best_model_name}\n")
    f.write(f"Best Test MAE: {results[best_model_name]['test_mae']:.4f}\n")
    f.write(f"Best Test R²: {results[best_model_name]['test_r2']:.4f}\n")

print(f"Saved summary: {summary_path}")

print("\n" + "=" * 70)
print("🎉 TRAINING COMPLETE!")
print("=" * 70)
print(f"\n✅ Best model ({best_model_name}) saved to: {model_path}")
print(f"📊 Use this model to predict quality scores for new sentence pairs.")
print(f"📁 All artifacts saved to: {OUTPUT_DIR}")

