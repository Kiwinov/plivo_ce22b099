# PII Entity Recognition for Noisy STT Transcripts

**Name: Samarth K J**
**Roll Number: CE22B099**

## Project Overview
This project implements a high-precision, low-latency Named Entity Recognition (NER) system designed to detect Personally Identifiable Information (PII) in noisy Speech-to-Text (STT) transcripts.

The objective was to identify sensitive entities (Credit Cards, Phones, Emails, Names, Dates) with a **PII Precision target of ≥ 0.80** while maintaining a strict inference latency budget of **p95 ≤ 20ms** on a standard CPU.

## Deliverables and Links
*   **Trained Model Weights:** [Google Drive Link](https://drive.google.com/drive/folders/1MjIAEWO-y_hQfykwNDu5uoPSV-9Wtcun?usp=sharing)
*   **Generated Datasets:** [Google Drive Link](https://drive.google.com/drive/folders/1A2-h_VbV4HGppKXOBV2Vq68Yhsm7xv2R?usp=sharing)
*   **Output Files (Dev):** [Google Drive Link](https://drive.google.com/drive/folders/1VGD_FN5TUriX0OJ15d5Vl4SGM8OzlzWO?usp=sharing)

---

## 1. Data Preparation Strategy

To ensure the model generalizes well to the specific quirks of STT data (lack of casing, spoken punctuation, dysfluencies), a multi-stage data generation and consolidation strategy was employed.

### Stage 1: Structural Noise Generation (Python)
A programmatic generator was built to create high volumes of structural patterns.
*   **Decoys:** Non-PII numbers (e.g., "Room 402", "Flight 99") were explicitly generated without labels to teach the model to distinguish sensitive numbers from generic ones.
*   **STT Artifacts:** Simulated spoken punctuation ("dot", "at", "underscore") and variable digit representation (mixing digits and words, e.g., "four 2").

### Stage 2: Semantic Variety (LLM)
To capture natural conversational flow, data was generated using the Gemini API.
*   **Scenarios:** Focused on Banking, Travel, and Tech Support contexts.
*   **Dysfluencies:** Included natural speech artifacts like "um", "uh", "actually, wait", and self-corrections which are difficult to script rule-based systems for.

### Stage 3: Stratified Consolidation
To prevent class imbalance (e.g., the model overfitting to "Person Names" because they are common), the generated data was not merged randomly. A **stratified sampling** script scanned the generated pool and selected a balanced dataset containing approximately ~200 examples per entity class.

---

## 2. Model Optimization & Hyperparameters

While newer architectures like DeBERTa were evaluated, **DeBERTa yielded an inference latency of ~40ms**, violating the project constraints. Consequently, the solution optimized the `distilbert-base-uncased` architecture to maximize performance within the latency budget.

Significant improvements over the baseline (PII Precision ~0.686) were achieved through specific hyperparameter tuning:

### A. Training Epochs (3 $\to$ 20)
The noisy nature of STT data requires the model to learn subtle contextual cues rather than relying on capitalization or punctuation. Increasing the training duration to 20 epochs allowed the model to converge on these more complex patterns.

### B. Batch Size (8 $\to$ 32)
Increasing the batch size stabilized the gradient descent process, leading to better generalization on the validation set.

### C. Sequence Length (256 $\to$ 128)
This was the most critical optimization for latency.
*   **Analysis:** An analysis of the generated dataset revealed that the longest transcript contained only **85 tokens**. (See `results/token_length_analysis.png` for the distribution).
*   **Optimization:** Reducing `max_length` from 256 to 128 had no negative impact on accuracy (as no data was truncated) but significantly reduced the self-attention computational overhead.
*   **Result:** This reduction allowed the model to comfortably hit the sub-20ms target.

---

## 3. Results

Detailed screenshots of the evaluation metrics, latency logs, and token length analysis can be found in the `results/` folder of this repository.

### PII Metrics
The final model achieved a massive improvement over the baseline, exceeding the precision target.

| Metric | Score | Notes |
| :--- | :--- | :--- |
| **PII Precision** | **0.941** | **Exceeds target of 0.80** |
| PII Recall | 0.933 | |
| PII F1 | 0.937 | |
| **Credit Card Precision** | 0.954 | |
| **Phone Precision** | 0.985 | |

*Evidence: See `results/pii_metrics.png`*

### Latency Performance
By optimizing the input sequence length to 128 and using DistilBERT, the model satisfies the strict latency requirements.

*   **p50 Latency:** 13.66 ms
*   **p95 Latency:** **16.02 ms** (Target: ≤ 20ms)

*Evidence: See `results/latency_log.png`*

---

## 4. Reproducibility

### Installation
```bash
pip install -r requirements.txt
```

### Training
```bash
python src/train.py \
  --model_name distilbert-base-uncased \
  --train data_combined/train.jsonl \
  --dev data_combined/dev.jsonl \
  --out_dir out \
  --epochs 20 \
  --batch_size 32 \
  --max_length 128
```

### Evaluation
```bash
python src/predict.py --model_dir out --input data/dev.jsonl --output out/dev_pred.json
python src/eval_span_f1.py --gold data/dev.jsonl --pred out/dev_pred.json
```

### Latency Check
```bash
python src/measure_latency.py --model_dir out --input data/dev.jsonl --runs 50
```
```
