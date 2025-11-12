
# Legal Document Summarization & Translation (Hindi → Mandarin Chinese)

📘 Overview

This project automates the summarization and translation of legal documents from Hindi to Mandarin Chinese.
It is designed for cross-border legal understanding, multilingual compliance, and policy research applications — where legal texts need to be accurately condensed and translated while preserving key legal semantics, tone, and references.

The system performs three main functions:
	1.	Text Extraction — Converts scanned PDFs or DOCX legal documents into clean Hindi text.
	2.	Summarization — Produces concise, legally faithful summaries in Hindi.
	3.	Translation — Translates the summary (or full text) from Hindi to Mandarin Chinese with consistent terminology and formal legal register.

⸻

🧩 System Architecture
```
+------------------+
|  Input Documents |
|  (PDF / DOCX)    |
+--------+---------+
         |
         v
+------------------+
| Text Extraction  |
| (OCR / Parser)   |
+--------+---------+
         |
         v
+--------------------------+
| Hindi Legal Summarizer   |
| (Transformer / LLM model)|
+--------+-----------------+
         |
         v
+--------------------------+
| Hindi → Mandarin         |
| Translation Model        |
| (NLLB / M2M100 / GPT-5)  |
+--------+-----------------+
         |
         v
+--------------------------+
| Output: Bilingual Summary|
+--------------------------+
```

⸻

🧠 Core Components

1. Text Extraction
	•	Scanned PDFs: Use Tesseract OCR￼ with the hin language pack.
	•	Digital PDFs / DOCX: Use pdfminer.six or python-docx for text extraction.
	•	Post-processing: Remove headers, footers, stamps, and line breaks that confuse NLP models.

2. Summarization (Hindi)
	•	Models:
	•	Start with facebook/mbart-large-50-many-to-many-mmt or google/pegasus-xsum fine-tuned on Hindi legal corpora.
	•	If data permits, fine-tune a Hindi summarizer using the HindSum￼ dataset.
	•	Evaluation: Use ROUGE-L and BLEU metrics to check summary quality against reference texts.

3. Translation (Hindi → Mandarin)
	•	Models:
	•	facebook/nllb-200-distilled-600M or facebook/m2m100_418M for open-source translation.
	•	For high-accuracy enterprise settings, you can use GPT-5 with "translate legal Hindi to Mandarin Chinese" prompts.
	•	Terminology alignment: Use a bilingual glossary of legal terms (e.g., “आदेश” → “裁决”, “अदालत” → “法院”).

4. Post-Processing
	•	Preserve named entities (e.g., court names, parties) in both scripts.
	•	Validate alignment between summary and full translation using sentence embeddings (Cosine similarity ≥ 0.85).

⸻

⚙️ Setup Instructions

Prerequisites
	•	Python ≥ 3.10
	•	CUDA-enabled GPU (recommended for transformer models)
	•	Conda or venv for dependency management

Installation

git clone https://github.com/<your-username>/legal-hindi-mandarin.git
cd legal-hindi-mandarin
conda create -n legal-pipeline python=3.10
conda activate legal-pipeline
pip install -r requirements.txt

Requirements (requirements.txt)

torch
transformers
sentencepiece
pdfminer.six
python-docx
pytesseract
opencv-python
langdetect
nltk
sacremoses
rouge-score

Configuration

Create a .env file:

MODEL_SUMMARIZER=facebook/mbart-large-50-many-to-many-mmt
MODEL_TRANSLATOR=facebook/nllb-200-distilled-600M
OCR_LANG=hin


⸻

🚀 Running the Pipeline

1. Extract text

python extract_text.py --input data/legal_doc.pdf --output data/legal_doc.txt

2. Summarize

python summarize.py --input data/legal_doc.txt --output data/legal_summary_hi.txt

3. Translate

python translate.py --input data/legal_summary_hi.txt --output data/legal_summary_zh.txt

4. Combined run

python run_pipeline.py --input data/legal_doc.pdf --output data/output_summary_zh.txt


⸻

📊 Evaluation Metrics

Task	Metric	Description
Summarization	ROUGE-L	Measures overlap with human summary
Translation	BLEU, chrF	Measures fidelity to reference translation
Semantic Consistency	Cosine Similarity	Checks if meaning preserved between Hindi & Mandarin embeddings


⸻

🌐 Deployment Options
	•	Streamlit Web App for document upload + bilingual summary display
	•	FastAPI REST API for programmatic use
	•	Docker Containerization for cloud deployment (AWS, Azure, GCP)

Example Streamlit UI command:

streamlit run app.py


⸻

⚖️ Legal & Ethical Considerations
	1.	Data Privacy: Only process documents with proper authorization.
	2.	Translation Accuracy: Always include human-in-the-loop validation for legal texts.
	3.	Bias Handling: Test models on diverse document types — contracts, court rulings, statutes — to ensure neutrality.
	4.	Model Transparency: Log model versions, prompts, and confidence scores for each output (for auditability).
	5.	Attribution: If using public datasets (e.g., HindSum, OPUS), include citation and comply with their licenses.

⸻

🔍 Future Enhancements
	•	Add Named Entity Recognition (NER) for legal entities.
	•	Implement cross-lingual summarization directly (Hindi → Mandarin summary in one step).
	•	Integrate retrieval-based factual correction (RAG) for citations.
	•	Deploy a multilingual glossary management tool for consistency.

⸻

👥 Contributors
	•	Shanay Wadhwani
  •	Ruijie Xu
  

⸻

📜 License

MIT License — see LICENSE￼ file for details.

⸻
