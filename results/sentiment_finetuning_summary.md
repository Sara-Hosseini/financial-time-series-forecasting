# Sentiment Fine-Tuning Summary
## FinBERT + LoRA for Stock Sentiment Analysis

---

### 1. Base Model

**Model**: [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)
- Pre-trained BERT variant specialised on financial text (Reuters, financial news)
- 3-class output: `negative`, `neutral`, `positive`
- ~110M parameters

---

### 2. Fine-Tuning Data

**Dataset**: Financial PhraseBank
- **Source**: Malo et al. (2014), "Good Debt or Bad Debt: Detecting Semantic Orientations in Economic Texts"
- **Size**: 4,840 manually labelled English sentences from financial news headlines
- **Label distribution**: ~16% negative, ~59% neutral, ~25% positive (class imbalance)
- **Domain**: General financial-domain text (earnings, market commentary, company news)

**Split**:
| Split      | Size         |
|------------|--------------|
| Train      | ~3,870 (80%) |
| Validation | ~484  (10%)  |
| Test       | ~484  (10%)  |

---

### 3. Fine-Tuning Method: LoRA (Low-Rank Adaptation)

**Why LoRA instead of full fine-tuning?**

Full fine-tuning updates all 110M parameters, which:
- Requires significant GPU memory (16GB+ for stable training)
- Is prone to catastrophic forgetting on small datasets (~4,840 sentences)
- Produces a large model file (~440MB)

LoRA instead injects small trainable low-rank matrices into the attention layers:
- Only ~0.5–2% of parameters are updated during training
- Training is feasible on a single consumer GPU or even CPU with patience
- The base FinBERT weights are frozen and cannot be forgotten
- The LoRA adapter is tiny (~5MB) and can be saved separately

**LoRA configuration**:
| Parameter       | Value |
|-----------------|-------|
| rank (r)        | 8     |
| alpha           | 16    |
| target modules  | query, value attention projections |
| dropout         | 0.1   |
| task type       | sequence classification |

---

### 4. Training Configuration

| Parameter       | Value           |
|-----------------|-----------------|
| Epochs          | 5               |
| Batch size      | 16              |
| Learning rate   | 2e-5            |
| Optimiser       | AdamW           |
| Max token length| 256             |
| Early stopping  | validation loss |
| Saved adapter   | `outputs/models/finbert_lora_best/` |

---

### 5. Evaluation Metrics

**Why Macro F1 matters:**
The dataset has class imbalance (neutral dominates at ~59%). Accuracy is misleading because a model that always predicts "neutral" would score ~59%. Macro F1 averages F1 scores equally across all three classes, penalising the model for poor minority-class performance.

**Reported metrics** (on test split of Financial PhraseBank):

| Metric            | Value (example range)     |
|-------------------|---------------------------|
| Accuracy          | ~85–90%                   |
| Macro F1          | ~80–87%                   |
| Positive F1       | ~82–88%                   |
| Neutral F1        | ~88–92%                   |
| Negative F1       | ~72–80%                   |

> Note: Exact values depend on random seed and training run. See `outputs/metrics/sentiment_baseline_comparison.json` for the project's recorded results.

**Comparison with zero-shot baselines** (see `src/sentiment/zero_few_shot.py`):
- Zero-shot FinBERT (no fine-tuning): ~75–78% Macro F1
- LoRA fine-tuned FinBERT: ~83–87% Macro F1
- Improvement: +5–10 percentage points on Macro F1

---

### 6. Application to Tickers

The fine-tuned model was **not** fine-tuned on Apple, Microsoft, or NVIDIA-specific news. It was fine-tuned on **general financial-domain text** from Financial PhraseBank and then applied zero-shot to AAPL, MSFT, and NVDA news articles.

This is standard practice in financial NLP:
- Domain-adapted (FinBERT) → Fine-tuned on general financial sentiment → Applied to specific stocks
- The LoRA adapter captures general financial sentiment patterns, not company-specific signals

**Output per article**: `[P(negative), P(neutral), P(positive)]`
**Sentiment score**: `sentiment_score = P(positive) - P(negative)` ∈ [-1, +1]

---

### 7. Daily Aggregation

For each ticker and trading date, the following 4 **core sentiment features** are extracted and used in the thesis models:

| Feature               | Description                                      |
|-----------------------|--------------------------------------------------|
| `avg_sentiment_score` | Mean (P_pos - P_neg) across all articles that day |
| `positive_ratio`      | Fraction of articles classified as positive      |
| `negative_ratio`      | Fraction of articles classified as negative      |
| `news_count`          | Number of articles collected for that day        |

Days with no news are filled with 0 (neutral baseline). The binary `has_news` flag indicates whether data was available.

---

### 8. Limitations

1. **General-domain training**: The model is not fine-tuned on ticker-specific language. NVDA-specific jargon (e.g., "CUDA cores", "data-center GPU ramp") may not be optimally scored.

2. **Short sentences only**: Financial PhraseBank contains short news headline fragments. Long analyst reports may be tokenised sub-optimally at max_length=256.

3. **Class imbalance**: Neutral sentences dominate. The negative class is underrepresented in training data, potentially leading to over-neutral predictions.

4. **Data coverage gaps**: News APIs have coverage gaps (especially pre-2018 for GDELT). Days with no news are filled with zeros rather than imputed, which may under-represent low-sentiment periods.

---

### 9. Knowledge Distillation (Future Work)

See `src/sentiment/knowledge_distillation_plan.md` for the planned approach to compress the fine-tuned FinBERT+LoRA model into a smaller student model (DistilBERT or MiniLM).

Knowledge distillation is a **scalability and efficiency** improvement, not a guaranteed price-prediction improvement. The thesis presents it as future work.
