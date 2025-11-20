"""
Example: Using Quality Estimation Models to Evaluate Translations

This script demonstrates how to use the trained quality estimation models
to evaluate translations from any source: NMT models, LLMs, human translators, etc.
"""

from predict_quality import predict_single, predict_batch

# ============================================================================
# Example 1: Evaluate a single translation
# ============================================================================

print("=" * 70)
print("Example 1: Single Translation Evaluation")
print("=" * 70)

hindi_source = "मैं आज स्कूल जा रहा हूँ।"
chinese_translation = "我今天要去学校。"

quality, confidence = predict_single(hindi_source, chinese_translation)

print(f"\nHindi: {hindi_source}")
print(f"Chinese: {chinese_translation}")
print(f"\nPredicted Quality: {quality.upper()}")
print(f"Confidence: {confidence}")

# ============================================================================
# Example 2: Evaluate translations from different systems
# ============================================================================

print("\n" + "=" * 70)
print("Example 2: Comparing Translations from Different Systems")
print("=" * 70)

hindi_source = "यह एक बहुत अच्छी किताब है।"

# Translation from System A (e.g., Google Translate, NMT model)
translation_a = "这是一本非常好的书。"

# Translation from System B (e.g., ChatGPT, LLM)
translation_b = "这是一本很棒的书。"

# Translation from System C (e.g., different model)
translation_c = "书好。"

print(f"\nSource (Hindi): {hindi_source}")
print("\nEvaluating translations from different systems:")

for i, (system, translation) in enumerate([("System A", translation_a), 
                                            ("System B", translation_b),
                                            ("System C", translation_c)], 1):
    quality, confidence = predict_single(hindi_source, translation)
    print(f"\n{system}:")
    print(f"  Translation: {translation}")
    print(f"  Quality: {quality.upper()}")
    print(f"  Confidence: {confidence[quality]:.4f}")

# ============================================================================
# Example 3: Batch evaluation
# ============================================================================

print("\n" + "=" * 70)
print("Example 3: Batch Evaluation")
print("=" * 70)

# Multiple sentence pairs to evaluate
sentence_pairs = [
    ("नमस्ते, आप कैसे हैं?", "你好，你好吗？"),
    ("मुझे यह पसंद है।", "我喜欢这个。"),
    ("क्या आप मेरी मदद कर सकते हैं?", "你能帮我吗？"),
]

results = predict_batch(sentence_pairs)

print("\nBatch Evaluation Results:")
for i, result in enumerate(results, 1):
    print(f"\n{i}. Hindi: {result['hindi']}")
    print(f"   Chinese: {result['chinese']}")
    print(f"   Quality: {result['quality_class'].upper()}")
    print(f"   Confidence: {result['confidence'][result['quality_class']]:.4f}")

# ============================================================================
# Example 4: Using with translation models/LLMs
# ============================================================================

print("\n" + "=" * 70)
print("Example 4: Integration with Translation Systems")
print("=" * 70)

def evaluate_translation_system(hindi_sentences, translation_function):
    """
    Evaluate a translation system by predicting quality for all translations.
    
    Args:
        hindi_sentences: List of Hindi source sentences
        translation_function: Function that takes Hindi text and returns Chinese translation
    
    Returns:
        List of quality predictions
    """
    sentence_pairs = []
    for hindi in hindi_sentences:
        chinese = translation_function(hindi)
        sentence_pairs.append((hindi, chinese))
    
    return predict_batch(sentence_pairs)

# Example: Simulated translation function (replace with actual model)
def mock_translation_model(hindi_text):
    """Mock translation - replace with actual model"""
    # In practice, this would call your NMT model or LLM
    # e.g., return translation_model.translate(hindi_text)
    translations = {
        "नमस्ते": "你好",
        "धन्यवाद": "谢谢",
    }
    return translations.get(hindi_text, "翻译")

# Example usage
hindi_sentences = ["नमस्ते", "धन्यवाद"]
results = evaluate_translation_system(hindi_sentences, mock_translation_model)

print("\nTranslation System Evaluation:")
for result in results:
    print(f"  {result['hindi']} → {result['chinese']}: {result['quality_class'].upper()}")

print("\n" + "=" * 70)
print("Note: Replace mock_translation_model with your actual translation system")
print("=" * 70)

