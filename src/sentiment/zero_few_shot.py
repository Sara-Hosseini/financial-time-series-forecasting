"""
zero_few_shot.py
────────────────────────────────────────────────────────────
Baseline comparison for financial sentiment classification:

  1. Zero-shot  : FinBERT base model (no fine-tuning, no examples)
  2. Few-shot   : FinBERT base model + 6 labeled examples in prompt
  3. Fine-tuned : FinBERT + LoRA (our trained model)

Metrics: Accuracy, F1-macro, Precision, Recall
Output : outputs/metrics/sentiment_baseline_comparison.json
         outputs/metrics/sentiment_baseline_comparison.txt

Usage:
    python -m src.sentiment.zero_few_shot
────────────────────────────────────────────────────────────
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)
from peft import PeftModel
from tqdm import tqdm


# ============================================================
# Config
# ============================================================
SEED             = 42
BASE_MODEL       = "ProsusAI/finbert"
LORA_MODEL_DIR   = "outputs/models/finbert_lora_best"
DATA_CSV         = "financial_phrasebank.csv"
OUT_JSON         = "outputs/metrics/sentiment_baseline_comparison.json"
OUT_TXT          = "outputs/metrics/sentiment_baseline_comparison.txt"

BATCH_SIZE       = 32
MAX_LENGTH       = 128
N_TEST_SAMPLES   = 500    # subset for fast evaluation
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

label2id = {"negative": 0, "neutral": 1, "positive": 2}
id2label  = {0: "negative", 1: "neutral", 2: "positive"}

# Few-shot examples (2 per class)
FEW_SHOT_EXAMPLES = [
    ("Company reports record quarterly profits exceeding all expectations.", "positive"),
    ("Stock surges 15% after better than expected earnings report.", "positive"),
    ("Company misses earnings estimates amid declining consumer demand.", "negative"),
    ("Stock plunges after weak quarterly results disappoint investors.", "negative"),
    ("The company will release its quarterly results next week.", "neutral"),
    ("Management announced a strategic review of its business units.", "neutral"),
]


# ============================================================
# Helpers
# ============================================================
def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def normalize_label(x) -> str:
    x = str(x).strip().lower()
    mapping = {
        "positive": "positive", "pos": "positive", "2": "positive",
        "negative": "negative", "neg": "negative", "0": "negative",
        "neutral":  "neutral",  "neu": "neutral",  "1": "neutral",
    }
    return mapping.get(x, "neutral")


def load_test_data(csv_path: str, n_samples: int = N_TEST_SAMPLES) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # detect columns
    cols_lower = {c.lower(): c for c in df.columns}
    text_col  = next((cols_lower[c] for c in ["text","sentence","phrase"] if c in cols_lower), None)
    label_col = next((cols_lower[c] for c in ["label","sentiment","class"] if c in cols_lower), None)

    if not text_col or not label_col:
        raise ValueError(f"Cannot find text/label columns. Got: {list(df.columns)}")

    df = df[[text_col, label_col]].copy()
    df.columns = ["text", "label"]
    df["text"]  = df["text"].fillna("").astype(str).str.strip()
    df["label"] = df["label"].apply(normalize_label)
    df = df[df["text"].str.len() > 5].drop_duplicates(subset=["text"]).reset_index(drop=True)
    df["label_id"] = df["label"].map(label2id)

    # stratified sample for speed
    _, test_df = train_test_split(
        df, test_size=min(n_samples / len(df), 0.9),
        random_state=SEED, stratify=df["label_id"]
    )
    return test_df.reset_index(drop=True)


def compute_metrics(y_true, y_pred, name: str) -> dict:
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(
        y_true, y_pred,
        target_names=["negative", "neutral", "positive"],
        zero_division=0,
    )
    print(f"\n── {name} ──")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  F1-macro  : {f1:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"\n{report}")
    return {
        "accuracy":         round(float(acc),  4),
        "f1_macro":         round(float(f1),   4),
        "precision_macro":  round(float(prec), 4),
        "recall_macro":     round(float(rec),  4),
        "classification_report": report,
    }


# ============================================================
# Method 1 — Zero-shot (FinBERT base, no examples)
# ============================================================
def run_zero_shot(test_df: pd.DataFrame, tokenizer, model) -> dict:
    """
    Uses FinBERT base model directly without any fine-tuning or examples.
    The model already has some financial sentiment knowledge from pre-training.
    """
    print("\n[1/3] Zero-shot evaluation (FinBERT base)...")
    model.eval()
    model.to(DEVICE)

    preds = []
    texts = test_df["text"].tolist()

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="  zero-shot"):
        batch = tokenizer(
            texts[i:i+BATCH_SIZE],
            padding=True, truncation=True,
            max_length=MAX_LENGTH, return_tensors="pt"
        )
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.no_grad():
            logits = model(**batch).logits
        pred_ids = logits.argmax(dim=-1).cpu().numpy()
        # map model's id2label to our label2id
        model_id2label = model.config.id2label
        for pid in pred_ids:
            label_str = str(model_id2label[int(pid)]).lower()
            preds.append(label2id.get(label_str, 1))

    y_true = test_df["label_id"].tolist()
    return compute_metrics(y_true, preds, "Zero-shot (FinBERT base)")


# ============================================================
# Method 2 — Few-shot (FinBERT base + examples as context)
# ============================================================
def run_few_shot(test_df: pd.DataFrame, tokenizer, model) -> dict:
    """
    Few-shot: prepends labeled examples to each input before classification.
    Format: "[POS] example1 [NEG] example2 ... [?] test_sentence"
    The model scores the full context.
    """
    print("\n[2/3] Few-shot evaluation (FinBERT base + 6 examples)...")
    model.eval()
    model.to(DEVICE)

    # Build few-shot prefix
    prefix_parts = []
    for text, label in FEW_SHOT_EXAMPLES:
        tag = {"positive": "[POS]", "negative": "[NEG]", "neutral": "[NEU]"}[label]
        prefix_parts.append(f"{tag} {text}")
    prefix = " ".join(prefix_parts) + " [?] "

    preds = []
    texts = test_df["text"].tolist()

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="  few-shot"):
        batch_texts = [prefix + t for t in texts[i:i+BATCH_SIZE]]
        batch = tokenizer(
            batch_texts,
            padding=True, truncation=True,
            max_length=MAX_LENGTH, return_tensors="pt"
        )
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.no_grad():
            logits = model(**batch).logits
        pred_ids = logits.argmax(dim=-1).cpu().numpy()
        model_id2label = model.config.id2label
        for pid in pred_ids:
            label_str = str(model_id2label[int(pid)]).lower()
            preds.append(label2id.get(label_str, 1))

    y_true = test_df["label_id"].tolist()
    return compute_metrics(y_true, preds, "Few-shot (FinBERT + 6 examples)")


# ============================================================
# Method 3 — Fine-tuned FinBERT + LoRA
# ============================================================
def run_finetuned(test_df: pd.DataFrame, tokenizer) -> dict:
    """
    Our fine-tuned FinBERT+LoRA model trained on financial sentiment data.
    """
    print("\n[3/3] Fine-tuned LoRA evaluation...")

    base = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=3,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    model = PeftModel.from_pretrained(base, LORA_MODEL_DIR)
    model = model.merge_and_unload()
    model.to(DEVICE)
    model.eval()

    preds = []
    texts = test_df["text"].tolist()

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="  fine-tuned"):
        batch = tokenizer(
            texts[i:i+BATCH_SIZE],
            padding=True, truncation=True,
            max_length=MAX_LENGTH, return_tensors="pt"
        )
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.no_grad():
            logits = model(**batch).logits
        pred_ids = logits.argmax(dim=-1).cpu().numpy()
        for pid in pred_ids:
            preds.append(int(pid))

    y_true = test_df["label_id"].tolist()
    return compute_metrics(y_true, preds, "Fine-tuned FinBERT + LoRA")


# ============================================================
# Summary table
# ============================================================
def print_summary(results: dict) -> str:
    lines = []
    lines.append("\n" + "="*65)
    lines.append("  SENTIMENT BASELINE COMPARISON SUMMARY")
    lines.append("="*65)
    lines.append(f"  {'Method':<35} {'Acc':>7} {'F1':>7} {'Prec':>7} {'Rec':>7}")
    lines.append("  " + "-"*61)

    for name, m in results.items():
        lines.append(
            f"  {name:<35} "
            f"{m['accuracy']:>7.4f} "
            f"{m['f1_macro']:>7.4f} "
            f"{m['precision_macro']:>7.4f} "
            f"{m['recall_macro']:>7.4f}"
        )
    lines.append("="*65)

    # improvement
    if "Zero-shot" in results and "Fine-tuned LoRA" in results:
        base_f1  = results["Zero-shot"]["f1_macro"]
        tuned_f1 = results["Fine-tuned LoRA"]["f1_macro"]
        improvement = (tuned_f1 - base_f1) / base_f1 * 100
        lines.append(f"\n  Fine-tuning improvement over zero-shot: +{improvement:.1f}% F1")

    summary = "\n".join(lines)
    print(summary)
    return summary


# ============================================================
# Main
# ============================================================
def ensure_dataset(csv_path: str) -> str:
    """Auto-download dataset if file is missing — same logic as train_finbert_phrasebank.py"""
    if os.path.exists(csv_path):
        return csv_path
    print(f"Dataset not found at '{csv_path}'. Auto-downloading from HuggingFace...")
    os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else ".", exist_ok=True)
    try:
        from datasets import load_dataset as hf_load
        try:
            ds  = hf_load("zeroshot/twitter-financial-news-sentiment", split="train")
            df  = ds.to_pandas()
            df["label"] = df["label"].map({0: "negative", 1: "positive", 2: "neutral"})
        except Exception:
            ds  = hf_load("financial_phrasebank", "sentences_allagree", trust_remote_code=True)
            df  = ds["train"].to_pandas()
            df["label"] = df["label"].map({0: "negative", 1: "neutral", 2: "positive"})
            df  = df.rename(columns={"sentence": "text"})
        df = df[["text", "label"]].dropna()
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"  Downloaded {len(df)} rows -> {csv_path}")
        return csv_path
    except ImportError:
        raise RuntimeError("Run: pip install datasets")
    except Exception as e:
        raise RuntimeError(f"Auto-download failed: {e}\nPlace CSV manually at: {csv_path}")


def main():
    print(f"Device: {DEVICE}")
    ensure_dataset(DATA_CSV)
    print(f"Loading test data: {DATA_CSV} ({N_TEST_SAMPLES} samples)")
    test_df = load_test_data(DATA_CSV, N_TEST_SAMPLES)
    print(f"Test samples: {len(test_df)}")
    print(test_df["label"].value_counts())

    print(f"\nLoading FinBERT base: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=3,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    results = {}
    results["Zero-shot"]      = run_zero_shot(test_df, tokenizer, base_model)
    results["Few-shot"]       = run_few_shot(test_df, tokenizer, base_model)
    results["Fine-tuned LoRA"] = run_finetuned(test_df, tokenizer)

    summary_text = print_summary(results)

    # Save JSON
    ensure_dir(OUT_JSON)
    out = {
        "n_test_samples": len(test_df),
        "label_distribution": test_df["label"].value_counts().to_dict(),
        "results": {k: {ek: ev for ek, ev in v.items() if ek != "classification_report"}
                    for k, v in results.items()},
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved JSON → {OUT_JSON}")

    # Save TXT
    ensure_dir(OUT_TXT)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n\n")
        for name, m in results.items():
            f.write(f"\n── {name} ──\n")
            f.write(m["classification_report"])
    print(f"Saved TXT  → {OUT_TXT}")

    print("\n✅ Done.")


if __name__ == "__main__":
    main()