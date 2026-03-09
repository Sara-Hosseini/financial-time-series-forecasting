"""
train_finbert_phrasebank.py
────────────────────────────────────────────────────────────
Fine-tunes FinBERT on financial news sentiment using LoRA.

Teacher model : ProsusAI/finbert
Dataset       : Twitter Financial News Sentiment (9543 samples)
Method        : LoRA (Low-Rank Adaptation)
Metrics       : Accuracy, F1-macro, Precision, Recall

Usage (local, CPU/GPU):
    python src/train_finbert_phrasebank.py

Usage (Kaggle — copy this file into a notebook cell with %%writefile):
    DATA_CSV = "/kaggle/input/sentiment/financial_phrasebank.csv"
    OUT_DIR  = "/kaggle/working/finbert_lora_best"
────────────────────────────────────────────────────────────
"""

import os
import json
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from tqdm import tqdm


# ============================================================
# Config
# ============================================================
SEED           = 42
MODEL_NAME     = "ProsusAI/finbert"   # teacher model

DATA_CSV       = "financial_phrasebank.csv"

OUT_DIR        = "outputs/models/finbert_lora_best"
METRICS_PATH   = "outputs/metrics/finbert_lora_metrics.json"

MAX_LENGTH     = 128
BATCH_SIZE     = 16
EPOCHS         = 5
LEARNING_RATE  = 2e-4

# LoRA hyper-parameters (طبق خواسته استاد)
LORA_R         = 16      # rank
LORA_ALPHA     = 32      # scaling factor
LORA_DROPOUT   = 0.1
LORA_TARGETS   = ["query", "value"]   # FinBERT attention layers

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

label2id = {"negative": 0, "neutral": 1, "positive": 2}
id2label = {0: "negative", 1: "neutral", 2: "positive"}


# ============================================================
# Reproducibility
# ============================================================
def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Helpers
# ============================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def ensure_parent_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def normalize_label(x) -> str:
    x = str(x).strip().lower()
    mapping = {
        "positive": "positive", "pos": "positive", "2": "positive",
        "negative": "negative", "neg": "negative", "0": "negative",
        "neutral":  "neutral",  "neu": "neutral",  "1": "neutral",
        # Twitter Financial News uses: bearish=0, bullish=2, neutral=1
        # already mapped above via "0","1","2"
    }
    if x not in mapping:
        raise ValueError(f"Unknown label value: '{x}'")
    return mapping[x]


def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Loads financial sentiment CSV.
    Supports: Financial PhraseBank, Twitter Financial News, custom.
    Detects text/label columns automatically.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)

    cols_lower = {c.lower(): c for c in df.columns}

    # find text column
    for candidate in ["text", "sentence", "phrase", "content", "headline"]:
        if candidate in cols_lower:
            text_col = cols_lower[candidate]
            break
    else:
        raise ValueError(f"Cannot find text column. Columns: {list(df.columns)}")

    # find label column
    for candidate in ["label", "sentiment", "class", "target"]:
        if candidate in cols_lower:
            label_col = cols_lower[candidate]
            break
    else:
        raise ValueError(f"Cannot find label column. Columns: {list(df.columns)}")

    df = df[[text_col, label_col]].copy()
    df.columns = ["text", "label"]

    df["text"]  = df["text"].fillna("").astype(str).str.strip()
    df["label"] = df["label"].apply(normalize_label)

    df = df[df["text"].str.len() > 5].drop_duplicates(subset=["text"]).reset_index(drop=True)
    return df


# ============================================================
# Dataset
# ============================================================
class SentimentDataset(Dataset):
    def __init__(self, texts, label_ids, tokenizer, max_length=MAX_LENGTH):
        self.texts     = list(texts)
        self.labels    = list(label_ids)
        self.tokenizer = tokenizer
        self.max_len   = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ============================================================
# Metrics
# ============================================================
def compute_metrics(y_true, y_pred) -> dict:
    acc = accuracy_score(y_true, y_pred)
    prec_mac, rec_mac, f1_mac, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    prec_wgt, rec_wgt, f1_wgt, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy":           round(float(acc),      4),
        "precision_macro":    round(float(prec_mac),  4),
        "recall_macro":       round(float(rec_mac),   4),
        "f1_macro":           round(float(f1_mac),    4),
        "precision_weighted": round(float(prec_wgt),  4),
        "recall_weighted":    round(float(rec_wgt),   4),
        "f1_weighted":        round(float(f1_wgt),    4),
    }


# ============================================================
# Train / Eval
# ============================================================
def train_one_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="  train", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    total_loss, all_preds, all_true = 0.0, [], []
    for batch in tqdm(loader, desc="  eval", leave=False):
        batch   = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        total_loss += outputs.loss.item()
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_true.extend(batch["labels"].cpu().numpy().tolist())
    metrics = compute_metrics(all_true, all_preds)
    metrics["loss"] = round(total_loss / max(len(loader), 1), 4)
    return metrics


# ============================================================
# Main
# ============================================================
def main():
    set_seed()
    print(f"Device : {DEVICE}")
    print(f"LoRA   : r={LORA_R}, alpha={LORA_ALPHA}, targets={LORA_TARGETS}")

    # ── Load data ──────────────────────────────────────────
    print(f"\nLoading dataset: {DATA_CSV}")
    df = load_dataset(DATA_CSV)
    df["label_id"] = df["label"].map(label2id)
    print(f"Total samples : {len(df)}")
    print("Label distribution:")
    print(df["label"].value_counts())

    # ── Split 70 / 15 / 15 ────────────────────────────────
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=SEED, stratify=df["label_id"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=SEED, stratify=temp_df["label_id"]
    )
    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    # ── Tokenizer ──────────────────────────────────────────
    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = SentimentDataset(train_df["text"], train_df["label_id"], tokenizer)
    val_ds   = SentimentDataset(val_df["text"],   val_df["label_id"],   tokenizer)
    test_ds  = SentimentDataset(test_df["text"],  test_df["label_id"],  tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Base model ─────────────────────────────────────────
    print(f"Loading FinBERT: {MODEL_NAME}")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    # ── Apply LoRA ─────────────────────────────────────────
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=LORA_TARGETS,
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    model.to(DEVICE)

    # ── Optimizer ──────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # ── Training loop ──────────────────────────────────────
    best_val_f1  = -1.0
    best_state   = None
    history      = []

    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'='*55}")
        print(f"Epoch {epoch}/{EPOCHS}")
        print(f"{'='*55}")

        train_loss  = train_one_epoch(model, train_loader, optimizer, DEVICE)
        val_metrics = evaluate(model, val_loader, DEVICE)

        history.append({
            "epoch":      epoch,
            "train_loss": round(train_loss, 4),
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

        print(f"Train loss : {train_loss:.4f}")
        print(f"Val loss   : {val_metrics['loss']:.4f}")
        print(f"Val Acc    : {val_metrics['accuracy']:.4f}")
        print(f"Val F1-mac : {val_metrics['f1_macro']:.4f}")

        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print("  ✅ Best model updated.")

    # ── Test evaluation ────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)

    print("\nEvaluating on test set...")
    test_metrics = evaluate(model, test_loader, DEVICE)

    print("\nTest Metrics:")
    for k, v in test_metrics.items():
        print(f"  {k:<25}: {v:.4f}")

    # per-class report
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            preds = torch.argmax(model(**batch).logits, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_true.extend(batch["labels"].cpu().numpy().tolist())

    print("\nPer-class Report:")
    print(classification_report(
        all_true, all_preds,
        target_names=["negative", "neutral", "positive"]
    ))

    # ── Save model ─────────────────────────────────────────
    ensure_dir(OUT_DIR)
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"\nSaved LoRA model → {OUT_DIR}")

    # ── Save metrics ───────────────────────────────────────
    metrics_out = {
        "model_name":        MODEL_NAME,
        "method":            "LoRA",
        "lora_config": {
            "r":              LORA_R,
            "lora_alpha":     LORA_ALPHA,
            "lora_dropout":   LORA_DROPOUT,
            "target_modules": LORA_TARGETS,
        },
        "device":            DEVICE,
        "n_total":           int(len(df)),
        "n_train":           int(len(train_df)),
        "n_val":             int(len(val_df)),
        "n_test":            int(len(test_df)),
        "label_distribution": df["label"].value_counts().to_dict(),
        "history":           history,
        "best_val_f1_macro": float(best_val_f1),
        "test_metrics":      test_metrics,
        "config": {
            "max_length":    MAX_LENGTH,
            "batch_size":    BATCH_SIZE,
            "epochs":        EPOCHS,
            "learning_rate": LEARNING_RATE,
            "seed":          SEED,
        },
    }

    ensure_parent_dir(METRICS_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)

    print(f"Saved metrics → {METRICS_PATH}")
    print("\n✅ Done.")


if __name__ == "__main__":
    main()