# Knowledge Distillation Plan
## Compressing FinBERT+LoRA into a Lightweight Student Model

> **Status**: Future work — not implemented in the current thesis pipeline.
> This document describes the planned approach for thesis Section 6 (Future Work).

---

### 1. Motivation

The current FinBERT+LoRA model is accurate (~83–87% Macro F1) but slow at inference:
- Model size: ~110M parameters (BERT-base) + ~2M LoRA adapter weights
- Inference per batch of 16 sentences: ~0.5–2 seconds on CPU
- Daily sentiment scoring for 3 tickers × hundreds of articles: ~5–20 minutes

For production use or real-time applications, a smaller model is needed.

**Knowledge distillation** trains a compact "student" model to mimic the soft probability outputs of the large "teacher" model, preserving most accuracy at a fraction of the cost.

---

### 2. Teacher Model

**Teacher**: Fine-tuned ProsusAI/FinBERT + LoRA adapter
- Saved at: `outputs/models/finbert_lora_best/`
- Output: 3-class softmax probabilities `[P_neg, P_neu, P_pos]`
- Macro F1: ~83–87% on Financial PhraseBank test set

The teacher's soft probability outputs (temperature-scaled) are used as targets for the student, not the hard labels. This transfers uncertainty information from the teacher.

---

### 3. Student Model Candidates

| Model         | Parameters | Expected Macro F1 | Speed-up |
|---------------|------------|-------------------|----------|
| DistilBERT    | ~66M       | ~78–83%           | ~2×      |
| MiniLM-L6-v2  | ~22M       | ~75–80%           | ~4–5×    |
| TinyBERT      | ~14M       | ~72–77%           | ~6–7×    |

**Recommended**: `microsoft/MiniLM-L6-v2` for maximum compression with acceptable accuracy loss.

---

### 4. Distillation Procedure

**Step 1 — Teacher inference**:
```python
teacher_probs = teacher_model(article_texts)  # shape: (n, 3)
# Temperature scaling to soften outputs
T = 4.0
soft_labels = softmax(logits / T, axis=-1)
```

**Step 2 — Student training with KL-divergence loss**:
```python
# Combined loss: hard label CE + soft label KD
alpha = 0.5  # balance between hard and soft targets
T = 4.0

loss = (alpha * hard_label_cross_entropy
      + (1 - alpha) * T**2 * kl_divergence(soft_labels, student_probs))
```

**Step 3 — Fine-tune student on Financial PhraseBank**:
- Same 80/10/10 split as teacher training
- Epochs: 10–15 (student converges faster)
- Learning rate: 3e-5
- Batch size: 32

---

### 5. Evaluation Protocol

Compare teacher vs student on:
- Financial PhraseBank test set: Accuracy, Macro F1, per-class F1, confusion matrix
- Inference speed: sentences per second (CPU and GPU)
- Model size: disk footprint (MB)
- Price prediction MAPE: run full walk-forward pipeline with student sentiment features

Expected outcome table (illustrative):

| Model          | Macro F1 | Inf. speed (sent/s CPU) | Size (MB) |
|----------------|----------|-------------------------|-----------|
| FinBERT+LoRA   | ~85%     | ~50                     | ~440      |
| MiniLM student | ~78–82%  | ~200–250                | ~90       |

---

### 6. Implementation Files (Planned)

```
src/sentiment/
├── train_student.py          # KD training loop
├── evaluate_student.py       # Accuracy / F1 / speed comparison
└── score_with_student.py     # Sentiment scoring using student model
```

---

### 7. Important Caveat for the Thesis

Knowledge distillation is an **efficiency improvement**, not a guaranteed forecast improvement.

- The student may produce slightly less accurate sentiment scores
- These small sentiment accuracy differences may or may not affect Q50 price predictions
- The thesis should NOT claim that KD improves price prediction accuracy
- The thesis CAN claim: KD enables real-time or low-latency deployment at acceptable accuracy cost

The appropriate framing is:
> "In future work, knowledge distillation could be applied to reduce inference latency
> from ~2s to ~0.2s per batch while retaining ~95% of the Macro F1 score. Whether this
> accuracy reduction propagates to price prediction degradation is an open question."

---

### 8. References

- Hinton et al. (2015), "Distilling the Knowledge in a Neural Network"
- Sanh et al. (2019), "DistilBERT, a distilled version of BERT"
- Wang et al. (2020), "MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers"
- Malo et al. (2014), "Good Debt or Bad Debt: Detecting Semantic Orientations in Economic Texts"
