"""
Translation Quality Estimation — Feature Engineering Module (v2)
---------------------------------------------------------------

This script provides a complete, modular, and reusable feature-engineering pipeline
for the Hindi→Chinese Translation Quality Estimation (TQE) project.

Major components provided in this module:

1. Data Loading
   - load_data(): load CCMatrix Hindi-Chinese parallel corpus.

2. Semantic Embedding Features
   - compute_embedding_similarities(): compute MiniLM embedding similarity.
   - LaBSE optional similarity is computed with the same function.

3. Fluency Features (Chinese perplexity)
   - load_chinese_lm(): load GPT2-Chinese model.
   - compute_perplexity(): compute LM-based fluency score.

4. Core Feature Engineering
   - add_length_feature(): length-based signals.
   - add_statitical_feature(): punctuation, digits, vocabulary diversity.
   - add_advanced_features(): interaction, polynomial, alignment, ratios.

5. Feature Selection Utilities
   - remove_colinear_features(): drop highly correlated features.
   - prepare_training_dataframe(): keep only selected columns and filter NA.

This file ONLY performs feature engineering — it does NOT train any model.
Training logic (splits, cross-validation, hyperparameter tuning) is handled in:
    • v2_model_train.py under the directory scripts/v2/
"""

import os
import torch
import warnings
import logging
import time
import pandas as pd
import numpy as np
import argparse

# Set up logging and warnings
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings('ignore')

# ============================================================================
# Confugiration
# ============================================================================
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

DATA_PATH = '../../data/hi-zh.txt/'
OUTPUT_DIR = '../../V2_FE_Output/'

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
def compute_embedding_similarities(hindi_sentences, chinese_sentences, model_name:str, batch_size=64):
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

    from sentence_transformers import SentenceTransformer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logging.info(f"Embedding model running on device: {device}")

    model = SentenceTransformer(model_name, device=device)

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

    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
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
def compute_perplexity(texts, tokenizer, model, device, batch_size=8):
    """
        Compute perplexity for a batch of Chinese sentences using GPU acceleration.
        Combines:
        - Batch GPU forward pass
        - Original progress logging
    """

    logging.info(f"Computing perplexity for {len(texts):,} sentences...")
    perplexities = []
    total = len(texts)
    start_time = time.time()

    for i in range(0, total, batch_size):
        batch_texts = texts[i:i+batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=256
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss                     # batch mean loss
            ppl_batch = torch.exp(loss).item()      # scalar perplexity

        perplexities.extend([ppl_batch] * len(batch_texts))

        if device == "mps" and (i % 1000 == 0) and i > 0:
            torch.mps.empty_cache()
            logging.info("    [MPS] Cleared cache to prevent memory fragmentation")

        if (i // batch_size + 1) % 10 == 0:
            elapsed = time.time() - start_time
            logging.info(
                f"  Processed {min(i + batch_size, total):,}/{total:,} "
                f"sentences ({elapsed:.1f}s elapsed)"
            )

    return np.array(perplexities)

# ============================================================================
# Feature Engineering
# ============================================================================
def add_length_feature(df:pd.DataFrame):
    """
        Add length-based features to the dataframe
        Args:
            df (pd.DataFrame): DataFrame with 'hindi' and 'chinese' columns
        Returns:
            pd.DataFrame: DataFrame with added length features
    """

    logging.info("Adding length-based features...")

    df['hindi_length'] = df['hindi'].str.len()
    df['chinese_length'] = df['chinese'].str.len()
    df['hindi_word_count'] = df['hindi'].str.split().str.len()
    df['chinese_char_count'] = df['chinese'].apply(lambda x: len([c for c in x if '\u4e00' <= c <= '\u9fff']))
    df['length_ratio'] = df['hindi_length'] / (df['chinese_length'] + 1)
    df['length_diff'] = abs(df['hindi_length'] - df['chinese_length'])
    df['hindi_avg_word_length'] = df['hindi_length'] / (df['hindi_word_count'] + 1)
    df['chinese_avg_char_length'] = df['chinese_length'] / (df['chinese_char_count'] + 1)

    logging.info("Length features added")

    return df

def add_statistical_feature(df:pd.DataFrame):
    """
        Add statistical features to the dataframe
        Args:
            df (pd.DataFrame): DataFrame with 'hindi' and 'chinese' columns
        Returns:
            pd.DataFrame: DataFrame with added statistical features
    """

    logging.info("Adding statistical features...")

    # Punctuation counts
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

    logging.info("Statistical features added")

    return df

# ============================================================================
# Advanced Feature Engineering
# ============================================================================
def add_advanced_features(df):
    """
    Add advanced semantic, interaction, and non-linear features to the DataFrame.

    Advanced feature categories included:
    ------------------------------------
    1. Embedding-Based Interaction Features:
        - embedding_labse_interaction: product of MiniLM & LaBSE similarities
        - similarity_avg: average semantic similarity
        - similarity_diff: disagreement magnitude between the two models

    2. Fluency Transformation Features:
        - chinese_fluency: inverse perplexity score (higher = more fluent)

    3. Length-Interaction Features:
        - length_similarity_interaction: semantic similarity × length ratio
        - length_perplexity_interaction: Chinese length × Chinese perplexity

    4. Normalized Length Metrics:
        - length_ratio_normalized: min-max scaled length ratio

    5. Character-Level Cross-Lingual Alignment:
        - char_alignment: crude alignment score using character set overlap
          (useful for digits, English words, symbols, etc.)

    6. Polynomial Features (Non-linear Expansion):
        - embedding_similarity_squared
        - labse_similarity_squared

    7. Ratio Features:
        - similarity_ratio: relative agreement between MiniLM and LaBSE
        - length_perplexity_ratio: fluency normalized by length

    Args:
        df (pd.DataFrame):
            Must include the following columns computed previously:
            - 'embedding_similarity'
            - 'labse_similarity'
            - 'chinese_perplexity'
            - 'length_ratio'
            - 'hindi'
            - 'chinese'
            - 'chinese_length'

    Returns:
        pd.DataFrame:
            The original DataFrame with added advanced feature columns.
    """

    logging.info("Adding advanced semantic & interaction features...")

    # Embedding interaction features
    df['embedding_labse_interaction'] = df['embedding_similarity'] * df['labse_similarity']
    df['similarity_avg'] = (df['embedding_similarity'] + df['labse_similarity']) / 2
    df['similarity_diff'] = abs(df['embedding_similarity'] - df['labse_similarity'])

    # Fluency transformation
    df['chinese_fluency'] = 1.0 / (df['chinese_perplexity'] + 1.0)

    # Length interaction
    df['length_similarity_interaction'] = df['length_ratio'] * df['embedding_similarity']
    df['length_perplexity_interaction'] = df['chinese_length'] * df['chinese_perplexity']

    # Normalized length ratio
    ratio_min = df['length_ratio'].min()
    ratio_max = df['length_ratio'].max()
    df['length_ratio_normalized'] = (df['length_ratio'] - ratio_min) / (ratio_max - ratio_min + 1e-8)

    # Character alignment feature
    def _alignment_score(a, b):
        set1, set2 = set(a.lower()), set(b.lower())
        inter = len(set1 & set2)
        union = len(set1 | set2)
        return inter / union if union > 0 else 0

    df['char_alignment'] = df.apply(lambda r: _alignment_score(r['hindi'], r['chinese']), axis=1)

    # Polynomial non-linear expansion
    df['embedding_similarity_squared'] = df['embedding_similarity'] ** 2
    df['labse_similarity_squared'] = df['labse_similarity'] ** 2

    # Ratio features
    df['similarity_ratio'] = df['embedding_similarity'] / (df['labse_similarity'] + 1e-8)
    df['length_perplexity_ratio'] = df['chinese_length'] / (df['chinese_perplexity'] + 1e-8)

    logging.info("Advanced features added successfully.")
    return df

# ============================================================================
# Filter out needed columns and remove nulls
# ============================================================================
def prepare_training_dataframe(df, feature_columns, target_column='ccmatrix_score'):
    """
    Select required feature columns, filter invalid rows, 
    and remove missing values for model training.

    Args:
        df (pd.DataFrame):
            Original DataFrame containing all raw & engineered features.

        feature_columns (list):
            List of feature names returned by define_feature_columns(),
            describing which columns to include as model inputs.

        target_column (str):
            Name of the target variable. Default: 'ccmatrix_score'.

    Returns:
        pd.DataFrame:
            A cleaned DataFrame containing only the selected features 
            + target column, with all NA rows removed.
    """

    logging.info("Preparing cleaned training DataFrame...")

    # Ensure all columns exist
    required_columns = feature_columns + [target_column]
    missing_cols = [col for col in required_columns if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns in DataFrame: {missing_cols}")

    # Select ONLY feature columns + target
    df_selected = df[required_columns].copy()

    # Remove rows with missing values
    df_clean = df_selected.dropna()

    logging.info(
        f"Training DataFrame prepared — {len(df_clean):,} rows "
        f"(removed {len(df_selected) - len(df_clean):,} rows with missing values)"
    )

    return df_clean

# ============================================================================
# Feature colinearity removal
# ============================================================================
def remove_colinear_features(df, feature_columns, threshold=0.95):
    """
    Automatically detect and remove highly colinear features.

    Steps:
    -------
    1. Compute correlation matrix of selected features.
    2. Identify feature pairs with correlation > threshold.
    3. Remove the feature that appears later in feature_columns.
    4. Return cleaned feature column list.

    Args:
        df (pd.DataFrame):
            DataFrame containing all features.

        feature_columns (list):
            List of feature names to check for colinearity.

        threshold (float):
            Correlation threshold for considering two features colinear.

    Returns:
        list:
            Filtered list of feature columns with colinear features removed.
    """

    logging.info("Removing colinear features...")

    feature_df = df[feature_columns]
    corr_matrix = feature_df.corr().abs()

    high_corr_pairs = []
    num_features = len(feature_columns)

    for i in range(num_features):
        for j in range(i + 1, num_features):
            if corr_matrix.iloc[i, j] > threshold:
                high_corr_pairs.append((feature_columns[i], feature_columns[j], corr_matrix.iloc[i, j]))

    logging.info(f"Found {len(high_corr_pairs)} high-correlation pairs.")

    features_to_remove = set()

    for feat1, feat2, _ in high_corr_pairs:
        idx1 = feature_columns.index(feat1)
        idx2 = feature_columns.index(feat2)
        # Remove the later one
        features_to_remove.add(feat2 if idx1 < idx2 else feat1)

    feature_columns_filtered = [f for f in feature_columns if f not in features_to_remove]

    logging.info(f"Removed {len(features_to_remove)} colinear features.")
    logging.info(f"Remaining features: {len(feature_columns_filtered)}")

    return feature_columns_filtered

# ============================================================================
# Main
# ============================================================================
def main():

    # Argument parsing
    parser = argparse.ArgumentParser(description="Feature Engineering Pipeline for Hindi→Chinese Translation Quality Estimation (v2)")
    parser.add_argument("--sample_size", type=int, default=None, help="Optional: load a subset of the corpus for quick testing")
    parser.add_argument("--minilm_model", type=str, default="paraphrase-multilingual-MiniLM-L12-v2", help="SentenceTransformer model name for embedding_similarity")
    parser.add_argument("--labse_model", type=str, default="sentence-transformers/LaBSE", help="Model name for LaBSE similarity computation")
    parser.add_argument("--colinearity_threshold", type=float, default=0.95, help="Correlation threshold for colinearity removal")
    args = parser.parse_args()

    # Pipeline 
    df = load_data(data_path=DATA_PATH, sample_size=args.sample_size)

    df["embedding_similarity"], _, _ = compute_embedding_similarities(df["hindi"].tolist(),df["chinese"].tolist(),args.minilm_model)
    df["labse_similarity"], _, _ = compute_embedding_similarities(df["hindi"].tolist(),df["chinese"].tolist(),args.labse_model)

    tokenizer, lm_model, device = load_chinese_lm()
    df["chinese_perplexity"] = compute_perplexity(df["chinese"].tolist(),tokenizer, lm_model, device)

    df = add_length_feature(df)
    df = add_statistical_feature(df)
    df = add_advanced_features(df)

    df_clean = prepare_training_dataframe(df, feature_columns)
    filtered_feature_cols = remove_colinear_features(df_clean, feature_columns, threshold=args.colinearity_threshold)

    logging.info(f"Final selected features ({len(filtered_feature_cols)}): {filtered_feature_cols}")
    logging.info("Preparing final feature-engineered dataframe...")
    df_final = df_clean[filtered_feature_cols + ['ccmatrix_score']]

    # Make output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save FE dataframe
    fe_csv_path = os.path.join(OUTPUT_DIR, f"tqe_features_{args.sample_size}.csv")
    df_final.to_csv(fe_csv_path, index=False)
    logging.info(f"Saved feature-engineered dataframe to {fe_csv_path}")
    logging.info("Feature Engineering Pipeline completed successfully!")

# ============================================================================
if __name__ == "__main__":
    main()