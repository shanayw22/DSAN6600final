## Task 1: Hindi ↔ Mandarin Emotional Machine Translation

### Datasets

- **Parallel Text (Hindi–Chinese)**  
  - **CCMatrix hi–zh** (~500k filtered good sentence pairs)

- **Hindi Emotion**  
  - **IITP Hindi Emotion Dataset** (~10k labeled sentences)

- **Chinese Emotion**  
  - **SMP2020** (~90k labeled sentences)

### Goal

- **Train an emotional MT system** for **Hindi ↔ Mandarin** using:
  - CCMatrix hi–zh as the **base parallel corpus**
  - IITP Hindi Emotion and SMP2020 as **monolingual emotion-labeled corpora**
  - A **classifier to label CCMatrix sentences with emotions** and then **filter** to a high-quality emotional parallel subset.

### High-Level Pipeline

1. **Data Collection & Cleaning**
   - Download CCMatrix hi–zh, IITP Hindi Emotion, and SMP2020.
   - Normalize text (Unicode normalization, strip weird whitespace, standardize punctuation).
   - Remove obviously noisy pairs from CCMatrix (very long sentences, empty sides, extreme length ratios).

2. **Emotion Labeling Models (Monolingual)**
   - **Hindi emotion classifier**:
     - Start from a multilingual transformer (e.g., mBERT, XLM-R) or a Hindi-specific model.
     - Fine-tune on IITP Hindi Emotion (~10k) for emotion classification (e.g., anger, joy, sadness, fear, neutral, etc.).
   - **Chinese emotion classifier**:
     - Start from a Chinese-pretrained model (e.g., BERT-base-Chinese, RoBERTa-wwm-ext, or a Chinese RoBERTa variant).
     - Fine-tune on SMP2020 (~90k) with the same or mapped label set.

3. **Label CCMatrix with Emotions**
   - **Option A (simpler, monolingual)**:
     - Run the **Hindi classifier** on the Hindi side of CCMatrix.
     - Run the **Chinese classifier** on the Chinese side.
     - Keep only pairs where the **predicted emotion matches** on both sides.
   - **Option B (more advanced, consistency-aware)**:
     - Use the predicted emotion probabilities and keep pairs where:
       - The **top emotion** matches **and**
       - The **confidence** is above a threshold on both sides (e.g., >0.7).
     - Discard pairs with low-confidence or conflicting emotions.
   - **Result**: A subset of CCMatrix (~150k) with **emotionally consistent parallel sentences**.

4. **Construct the Emotional MT Corpus**
   - Combine:
     - The filtered emotional CCMatrix subset (~150k).
     - Optionally, **back-translate** monolingual emotional sentences:
       - Translate Hindi IITP sentences into Chinese using a baseline MT model.
       - Translate Chinese SMP2020 sentences into Hindi.
       - Keep these as **noisier auxiliary parallel data** for low-resource emotions.

5. **Train Baseline MT Model**
   - Start with a standard MT architecture:
     - **Option A**: Transformer (e.g., fairseq, OpenNMT, Marian, or custom PyTorch/TF).
     - **Option B**: Fine-tune a pretrained multilingual NMT model (e.g., mBART, Marian, NLLB) on hi–zh.
   - Train on:
     - Full CCMatrix hi–zh for **general translation quality**.
     - Then fine-tune on the **emotional subset** for **emotion preservation**.

6. **Emotion-Aware MT Fine-Tuning (Improvements)**
   - **Emotion tags as special tokens**:
     - Prepend an emotion token to the source (e.g., `<joy>`, `<sadness>`).
     - Train the model to condition on the target emotion.
   - **Multi-task learning**:
     - Jointly train MT and emotion prediction:
       - Add an auxiliary classifier head on encoder or decoder states.
       - Optimize MT loss + emotion classification loss.
   - **Curriculum learning**:
     - Start training with **general** pairs.
     - Gradually increase the proportion of **emotionally strong** pairs.

7. **Evaluation**
   - **Automatic metrics**:
     - BLEU / chrF or COMET for translation quality.
     - Accuracy / F1 for emotion consistency:
       - Run emotion classifiers on both source and generated translations and compare labels.
   - **Human evaluation**:
     - Evaluate **fluency**, **adequacy**, and **emotion preservation** for a sample.

### Pipeline Improvements & Fixes (Suggestions)

- **Better Filtering of CCMatrix**
  - Use scoring methods (e.g., LASER, LaBSE, or sentence-transformer cosine similarity) to:
    - Remove misaligned or non-parallel pairs.
    - Rank pairs by semantic similarity before emotion filtering.

- **Unified Emotion Label Space**
  - Map the emotions from IITP and SMP2020 into a **common label set** (e.g., {joy, sadness, anger, fear, disgust, surprise, neutral}).
  - Manually inspect ambiguous labels and merge under broader categories if needed.

- **Handling Class Imbalance**
  - Emotion datasets often have **imbalanced classes**:
    - Use class weighting, focal loss, or oversampling for rare emotions.
    - When selecting emotional CCMatrix pairs, **downsample majority emotions** to avoid skew.

- **Noise-Robust Training**
  - Emotion labels from classifiers will be noisy:
    - Use **confidence thresholds** and **label smoothing**.
    - Consider **soft-label training** (use probabilities instead of hard argmax labels when doing multi-task).

- **Cross-Lingual Consistency Checks**
  - After training MT:
    - Translate Hindi → Chinese and Chinese → Hindi.
    - Run both emotion classifiers on both source and translation.
    - Flag and analyze pairs where emotion changes (useful for debugging and further data cleaning).

### How to Start (Concrete Steps)

1. **Environment & Repo Setup**
   - Create a new Python environment (conda/venv) and install:
     - `transformers`, `datasets`, `sentencepiece`, `torch` or `tensorflow`, `sentence-transformers`, `pandas`, `scikit-learn`.
   - Set up the repo structure, for example:
     - `data/` (raw + processed)
     - `scripts/` (download, preprocess, training)
     - `models/` (saved checkpoints)
     - `notebooks/` (exploration and experiments)

2. **Step 1: Get One Classifier Working**
   - Start with **Hindi emotion classifier**:
     - Load IITP dataset.
     - Clean and tokenize.
     - Fine-tune a transformer and get a **reasonable F1** on validation.
   - Save the model and write a small **inference script** that takes text and returns emotion + confidence.

3. **Step 2: Label a Small CCMatrix Sample**
   - Take a **small subset** (e.g., 20–50k pairs) of CCMatrix hi–zh.
   - Run your classifier on the Hindi side.
   - Inspect distributions, example predictions, and **qualitative correctness**.
   - Iterate on classifier or filtering thresholds before scaling up.

4. **Step 3: Build the Emotional Parallel Subset**
   - Once the labeling looks good:
     - Run labeling on the full CCMatrix hi–zh.
     - (Optional but recommended) Add the Chinese classifier and enforce **emotion agreement**.
   - Filter to about **150k high-quality, emotion-consistent pairs**.

5. **Step 4: Train a Baseline MT Model**
   - Train or fine-tune a hi–zh MT model:
     - First on **full CCMatrix**.
     - Then **fine-tune on the emotional subset**.
   - Evaluate on a small test set you manually create with emotional sentences.

6. **Step 5: Add Emotion Conditioning**
   - Modify the MT inputs to include emotion tags.
   - Re-train or fine-tune and compare:
     - Baseline MT vs. Emotion-tagged MT on both BLEU and emotion preservation.

This gives you a clear roadmap: **start with one emotion classifier, validate labeling on a CCMatrix sample, build an emotional subset, then train and refine an emotion-aware MT model for Hindi ↔ Mandarin.**


