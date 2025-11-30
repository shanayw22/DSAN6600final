"""
Translation Quality Estimation - Prediction Script

Use trained models to predict translation quality for any Hindi-Chinese sentence pair.
Works with translations from any source: NMT models, LLMs, human translators, etc.

Predicts continuous CCMatrix alignment scores (regression).
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
    # Import loss and metric classes for custom_objects
    from tensorflow.keras.losses import MeanSquaredError
    from tensorflow.keras.metrics import MeanAbsoluteError
except ImportError:
    TF_AVAILABLE = False
    MeanSquaredError = None
    MeanAbsoluteError = None

# ============================================================================
# Configuration
# ============================================================================
# MODELS_DIR will be set based on current working directory
# Try relative path first, then absolute
import os
if os.path.exists('models/'):
    MODELS_DIR = 'models/'
elif os.path.exists('../models/'):
    MODELS_DIR = '../models/'
else:
    # Try to find models directory
    current_dir = os.getcwd()
    if 'scripts' in current_dir:
        MODELS_DIR = '../models/'
    else:
        MODELS_DIR = 'models/'

# ============================================================================
# Feature Extraction Functions (same as training)
# ============================================================================

def compute_char_alignment_score(text1, text2):
    """Simple character-level alignment heuristic"""
    chars1 = set(text1.lower())
    chars2 = set(text2.lower())
    if len(chars1) == 0 or len(chars2) == 0:
        return 0.0
    intersection = len(chars1 & chars2)
    union = len(chars1 | chars2)
    return intersection / union if union > 0 else 0.0

def extract_features(hindi_sentence, chinese_sentence):
    """
    Extract features from a Hindi-Chinese sentence pair.
    Same features used during training.
    """
    features = {}
    
    # Length features
    features['hindi_length'] = len(hindi_sentence)
    features['chinese_length'] = len(chinese_sentence)
    features['hindi_word_count'] = len(hindi_sentence.split())
    features['chinese_char_count'] = len([c for c in chinese_sentence if '\u4e00' <= c <= '\u9fff'])
    features['length_ratio'] = features['hindi_length'] / (features['chinese_length'] + 1)
    features['length_diff'] = abs(features['hindi_length'] - features['chinese_length'])
    features['hindi_avg_word_length'] = features['hindi_length'] / (features['hindi_word_count'] + 1)
    features['chinese_avg_char_length'] = features['chinese_length'] / (features['chinese_char_count'] + 1)
    
    # Semantic features
    # Feature 1: Embedding similarity (paraphrase-multilingual-MiniLM-L12-v2)
    try:
        from sentence_transformers import SentenceTransformer
        if not hasattr(extract_features, '_embedding_model'):
            extract_features._embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        model = extract_features._embedding_model
        hi_emb = model.encode([hindi_sentence], convert_to_numpy=True, normalize_embeddings=True)[0]
        zh_emb = model.encode([chinese_sentence], convert_to_numpy=True, normalize_embeddings=True)[0]
        features['embedding_similarity'] = np.sum(hi_emb * zh_emb)
    except Exception as e:
        print(f"Warning: Could not compute embedding similarity: {e}")
        features['embedding_similarity'] = 0.5  # Default value
    
    # Feature 2: LaBSE similarity (different embedding model)
    try:
        from sentence_transformers import SentenceTransformer as ST
        if not hasattr(extract_features, '_labse_model'):
            extract_features._labse_model = ST('sentence-transformers/LaBSE')
        
        labse_model = extract_features._labse_model
        hi_emb = labse_model.encode([hindi_sentence], convert_to_numpy=True, normalize_embeddings=True)[0]
        zh_emb = labse_model.encode([chinese_sentence], convert_to_numpy=True, normalize_embeddings=True)[0]
        features['labse_similarity'] = np.sum(hi_emb * zh_emb)
    except Exception as e:
        print(f"Warning: Could not compute LaBSE similarity: {e}")
        features['labse_similarity'] = 0.5  # Default value
    
    # Feature 3: Chinese language model perplexity (fluency indicator)
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        perplexity_model_name = 'uer/gpt2-chinese-cluecorpussmall'
        
        # Load model (cache it in module-level variable to avoid reloading)
        if not hasattr(extract_features, '_chinese_tokenizer'):
            extract_features._chinese_tokenizer = AutoTokenizer.from_pretrained(perplexity_model_name)
            extract_features._chinese_lm = AutoModelForCausalLM.from_pretrained(perplexity_model_name)
            extract_features._chinese_lm.eval()
            extract_features._chinese_lm.to(device)
        
        tokenizer = extract_features._chinese_tokenizer
        model = extract_features._chinese_lm
        
        # Compute perplexity
        inputs = tokenizer(chinese_sentence, return_tensors='pt', truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss.item()
            ppl = np.exp(loss)
            features['chinese_perplexity'] = ppl
    except Exception as e:
        print(f"Warning: Could not compute Chinese perplexity: {e}")
        features['chinese_perplexity'] = 30.0  # Default perplexity value
    
    # Advanced features (computed from base features)
    # These need to be computed after we have the base features
    # We'll compute them after creating the features dict
    
    # Statistical features
    features['punctuation_count_hi'] = len([c for c in hindi_sentence if c in '.,!?;:()[]{}"\''])
    features['punctuation_count_zh'] = len([c for c in chinese_sentence if c in '.,!?;:()[]{}"\''])
    features['digit_count_hi'] = sum(c.isdigit() for c in hindi_sentence)
    features['digit_count_zh'] = sum(c.isdigit() for c in chinese_sentence)
    features['punctuation_ratio_hi'] = features['punctuation_count_hi'] / (features['hindi_length'] + 1)
    features['punctuation_ratio_zh'] = features['punctuation_count_zh'] / (features['chinese_length'] + 1)
    
    # Vocabulary diversity
    features['vocab_diversity_hi'] = len(set(hindi_sentence.split())) / (features['hindi_word_count'] + 1)
    features['vocab_diversity_zh'] = len(set(list(chinese_sentence))) / (features['chinese_length'] + 1)
    
    # Advanced features (interactions, polynomials, etc.)
    # Note: length_ratio_normalized requires min/max from training data, so we'll use a default
    # This should ideally be loaded from training statistics, but for simplicity we'll compute it
    features['chinese_fluency'] = 1.0 / (features['chinese_perplexity'] + 1.0)
    features['embedding_labse_interaction'] = features['embedding_similarity'] * features['labse_similarity']
    features['similarity_avg'] = (features['embedding_similarity'] + features['labse_similarity']) / 2
    features['similarity_diff'] = abs(features['embedding_similarity'] - features['labse_similarity'])
    features['embedding_similarity_squared'] = features['embedding_similarity'] ** 2
    features['labse_similarity_squared'] = features['labse_similarity'] ** 2
    features['similarity_ratio'] = features['embedding_similarity'] / (features['labse_similarity'] + 1e-8)
    features['char_alignment'] = compute_char_alignment_score(hindi_sentence, chinese_sentence)
    features['length_similarity_interaction'] = features['length_ratio'] * features['embedding_similarity']
    features['length_perplexity_interaction'] = features['chinese_length'] * features['chinese_perplexity']
    features['length_perplexity_ratio'] = features['chinese_length'] / (features['chinese_perplexity'] + 1e-8)
    
    # Note: length_ratio_normalized requires training data statistics
    # We'll use a simple normalization based on typical ranges
    # In production, this should be loaded from training statistics
    typical_min = 0.1
    typical_max = 10.0
    features['length_ratio_normalized'] = (features['length_ratio'] - typical_min) / (typical_max - typical_min + 1e-8)
    features['length_ratio_normalized'] = max(0.0, min(1.0, features['length_ratio_normalized']))  # Clip to [0, 1]
    
    return features

# ============================================================================
# Load Model and Artifacts
# ============================================================================

def load_model_and_artifacts():
    """Load the trained model, scaler, target_scaler, and feature columns"""
    
    # Find model file
    model_files = []
    for file in os.listdir(MODELS_DIR):
        if file.startswith('quality_estimation_') and (file.endswith('.pkl') or file.endswith('.h5')):
            model_files.append(file)
    
    if not model_files:
        raise FileNotFoundError(f"No trained model found in {MODELS_DIR}. Please train a model first.")
    
    # Use the first model found (or you could select based on name)
    model_file = model_files[0]
    model_path = os.path.join(MODELS_DIR, model_file)
    
    print(f"Loading model: {model_file}")
    
    # Load model
    if model_file.endswith('.h5'):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required to load .h5 model. Install with: pip install tensorflow")
        # The issue: Keras tries to deserialize loss/metric during load, even with compile=False
        # Solution: Apply monkey-patch FIRST, then load model architecture and weights separately
        import h5py
        import json
        
        # CRITICAL: Apply monkey-patch BEFORE any Keras operations
        # This fixes the bug where Keras looks for 'mse' in keras.metrics instead of keras.losses
        import keras.metrics as km
        import keras.losses as kl
        original_km_mse = getattr(km, 'mse', None)
        original_kl_mse = getattr(kl, 'mse', None)
        
        # Add mse to metrics (Keras bug looks in wrong place)
        km.mse = tf.keras.losses.mean_squared_error
        if not hasattr(kl, 'mse'):
            kl.mse = tf.keras.losses.mean_squared_error
        
        # Also patch serialization registry if available
        try:
            from keras.src.saving import serialization_lib
            if hasattr(serialization_lib, '_GLOBAL_CUSTOM_OBJECTS'):
                serialization_lib._GLOBAL_CUSTOM_OBJECTS['mse'] = tf.keras.losses.mean_squared_error
        except:
            pass
        
        try:
            # Try loading from separate architecture + weights files first (new format)
            model_name_base = model_file.replace('quality_estimation_', '').replace('.h5', '')
            architecture_path = os.path.join(MODELS_DIR, f'quality_estimation_{model_name_base}_architecture.json')
            weights_path = os.path.join(MODELS_DIR, f'quality_estimation_{model_name_base}.weights.h5')
            
            if os.path.exists(architecture_path) and os.path.exists(weights_path):
                # Load architecture from JSON
                with open(architecture_path, 'r') as f:
                    model_config = f.read()
                model = tf.keras.models.model_from_json(model_config)
                
                # Load weights
                model.load_weights(weights_path)
                
                # Compile with the same settings used during training
                model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                print(f"    ✓ Loaded model from architecture + weights files")
            else:
                # Fallback: Try loading from H5 file by extracting config
                # Open H5 file and extract model config
                with h5py.File(model_path, 'r') as f:
                    if 'model_config' in f.attrs:
                        model_config_json = f.attrs['model_config']
                        if isinstance(model_config_json, bytes):
                            model_config_json = model_config_json.decode('utf-8')
                        model_config = json.loads(model_config_json)
                    else:
                        raise ValueError("Model config not found in H5 file")
                
                # Reconstruct model from config (this bypasses loss/metric deserialization)
                model = tf.keras.models.model_from_json(model_config)
                
                # Load weights separately
                model.load_weights(model_path, by_name=False, skip_mismatch=False)
                
                # Compile with the same settings used during training
                model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                print(f"    ✓ Loaded model from H5 file (extracted config)")
                
        except Exception as e:
            # Restore original state on error
            if original_km_mse is not None:
                km.mse = original_km_mse
            elif hasattr(km, 'mse'):
                delattr(km, 'mse')
            if original_kl_mse is not None:
                kl.mse = original_kl_mse
            elif hasattr(kl, 'mse') and kl.mse == tf.keras.losses.mean_squared_error:
                delattr(kl, 'mse')
            
            raise RuntimeError(
                f"Could not load model:\n"
                f"Error: {e}\n\n"
                f"SOLUTION: Please retrain the model. The new training will save models in a compatible format."
            )
        finally:
            # Clean up monkey-patch (optional - we can leave it for future loads)
            # Uncomment if you want to restore original state
            # if original_km_mse is not None:
            #     km.mse = original_km_mse
            # elif hasattr(km, 'mse'):
            #     delattr(km, 'mse')
            pass
        model_type = 'keras'
    else:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        model_type = 'sklearn'
    
    # Load feature scaler
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Load target scaler (for regression)
    target_scaler_path = os.path.join(MODELS_DIR, 'target_scaler.pkl')
    if not os.path.exists(target_scaler_path):
        raise FileNotFoundError(f"Target scaler not found: {target_scaler_path}")
    with open(target_scaler_path, 'rb') as f:
        target_scaler = pickle.load(f)
    
    # Load feature columns
    feature_path = os.path.join(MODELS_DIR, 'feature_columns.pkl')
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"Feature columns not found: {feature_path}")
    with open(feature_path, 'rb') as f:
        feature_columns = pickle.load(f)
    
    return model, scaler, target_scaler, feature_columns, model_type

# ============================================================================
# Prediction Functions
# ============================================================================

def predict_quality(hindi_sentence, chinese_sentence, model, scaler, target_scaler, feature_columns, model_type):
    """
    Predict CCMatrix alignment score for a Hindi-Chinese sentence pair (regression).
    
    Args:
        hindi_sentence: Source sentence in Hindi
        chinese_sentence: Target sentence in Chinese (translation)
        model: Trained quality estimation model
        scaler: Feature scaler
        target_scaler: Target scaler (for inverse transform)
        feature_columns: List of feature names
        model_type: 'keras' or 'sklearn'
    
    Returns:
        predicted_score: Continuous CCMatrix alignment score
        confidence_interval: Tuple of (lower_bound, upper_bound) - approximate 95% CI
    """
    # Extract features
    features_dict = extract_features(hindi_sentence, chinese_sentence)
    
    # Convert to array in correct order
    features_array = np.array([features_dict[col] for col in feature_columns]).reshape(1, -1)
    
    # Scale features
    features_scaled = scaler.transform(features_array)
    
    # Predict (in scaled space)
    if model_type == 'keras':
        # Use direct model call to avoid predict() hang on M1/M2 Macs
        prediction_scaled = model(features_scaled, training=False).numpy()
        if len(prediction_scaled.shape) > 1 and prediction_scaled.shape[1] == 1:
            prediction_scaled = prediction_scaled.squeeze(axis=1)
        prediction_scaled = prediction_scaled[0]
    else:
        prediction_scaled = model.predict(features_scaled)[0]
    
    # Inverse transform to original scale
    predicted_score = target_scaler.inverse_transform([[prediction_scaled]])[0][0]
    
    # Approximate confidence interval (using a simple heuristic)
    # In practice, you might want to use prediction intervals from the model
    std_estimate = 0.1 * abs(predicted_score)  # Rough estimate: 10% of prediction
    confidence_interval = (predicted_score - 1.96 * std_estimate, predicted_score + 1.96 * std_estimate)
    
    return predicted_score, confidence_interval

# ============================================================================
# Main Functions
# ============================================================================

def predict_single(hindi_sentence, chinese_sentence):
    """Predict quality score for a single sentence pair"""
    model, scaler, target_scaler, feature_columns, model_type = load_model_and_artifacts()
    
    score, confidence_interval = predict_quality(
        hindi_sentence, chinese_sentence, 
        model, scaler, target_scaler, feature_columns, model_type
    )
    
    return score, confidence_interval

def predict_batch(sentence_pairs):
    """
    Predict quality scores for multiple sentence pairs.
    
    Args:
        sentence_pairs: List of tuples [(hindi1, chinese1), (hindi2, chinese2), ...]
    
    Returns:
        List of predictions with scores and confidence intervals
    """
    model, scaler, target_scaler, feature_columns, model_type = load_model_and_artifacts()
    
    results = []
    for hindi, chinese in sentence_pairs:
        score, confidence_interval = predict_quality(
            hindi, chinese,
            model, scaler, target_scaler, feature_columns, model_type
        )
        results.append({
            'hindi': hindi,
            'chinese': chinese,
            'predicted_score': score,
            'confidence_lower': confidence_interval[0],
            'confidence_upper': confidence_interval[1],
        })
    
    return results

# ============================================================================
# Command Line Interface
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict translation quality for Hindi-Chinese pairs')
    parser.add_argument('--hindi', type=str, help='Hindi source sentence')
    parser.add_argument('--chinese', type=str, help='Chinese translation')
    parser.add_argument('--file', type=str, help='CSV file with hindi and chinese columns')
    parser.add_argument('--output', type=str, help='Output file for predictions')
    
    args = parser.parse_args()
    
    if args.file:
        # Batch prediction from file
        print(f"Loading sentences from {args.file}...")
        df = pd.read_csv(args.file)
        
        if 'hindi' not in df.columns or 'chinese' not in df.columns:
            raise ValueError("CSV file must have 'hindi' and 'chinese' columns")
        
        sentence_pairs = list(zip(df['hindi'], df['chinese']))
        results = predict_batch(sentence_pairs)
        
        # Add predictions to dataframe
        df['predicted_score'] = [r['predicted_score'] for r in results]
        df['confidence_lower'] = [r['confidence_lower'] for r in results]
        df['confidence_upper'] = [r['confidence_upper'] for r in results]
        
        # Save results
        if args.output:
            df.to_csv(args.output, index=False)
            print(f"Predictions saved to {args.output}")
        else:
            print(df[['hindi', 'chinese', 'predicted_score', 'confidence_lower', 'confidence_upper']].to_string())
    
    elif args.hindi and args.chinese:
        # Single prediction
        score, confidence_interval = predict_single(args.hindi, args.chinese)
        
        print("\n" + "=" * 70)
        print("TRANSLATION QUALITY ESTIMATION")
        print("=" * 70)
        print(f"\nHindi: {args.hindi}")
        print(f"Chinese: {args.chinese}")
        print(f"\nPredicted CCMatrix Score: {score:.4f}")
        print(f"95% Confidence Interval: [{confidence_interval[0]:.4f}, {confidence_interval[1]:.4f}]")
        print("\nInterpretation:")
        print("  Higher scores indicate better alignment/similarity between sentences.")
        print("  Scores typically range from ~0.0 to ~1.0 (depending on training data).")
        print("=" * 70)
    
    else:
        # Interactive mode
        print("=" * 70)
        print("Translation Quality Estimation - Interactive Mode")
        print("=" * 70)
        print("\nEnter Hindi-Chinese sentence pairs to predict quality.")
        print("Type 'quit' to exit.\n")
        
        model, scaler, target_scaler, feature_columns, model_type = load_model_and_artifacts()
        
        while True:
            hindi = input("Hindi sentence: ").strip()
            if hindi.lower() == 'quit':
                break
            
            chinese = input("Chinese translation: ").strip()
            if chinese.lower() == 'quit':
                break
            
            score, confidence_interval = predict_quality(
                hindi, chinese,
                model, scaler, target_scaler, feature_columns, model_type
            )
            
            print(f"\nPredicted CCMatrix Score: {score:.4f}")
            print(f"95% Confidence Interval: [{confidence_interval[0]:.4f}, {confidence_interval[1]:.4f}]")
            print()
