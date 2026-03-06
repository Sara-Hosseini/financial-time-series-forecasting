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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


# ============================================================
# Config
# ============================================================
SEED = 42
MODEL_NAME = "ProsusAI/finbert"

DATA_CSV = "financial_phrasebank.csv"

OUT_DIR = "outputs/models/finbert_phrasebank_best"
METRICS_PATH = "outputs/metrics/finbert_phrasebank_metrics.json"

MAX_LENGTH = 128
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Reproducibility
# ============================================================
def set_seed(seed: int = 42):
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


def detect_columns(df: pd.DataFrame):
    """
    Tries to detect text and label columns robustly.
    """
    cols_lower = {c.lower(): c for c in df.columns}

    text_candidates = ["text", "sentence", "phrase", "content"]
    label_candidates = ["label", "sentiment", "class", "target"]

    text_col = None
    label_col = None

    for c in text_candidates:
        if c in cols_lower:
            text_col = cols_lower[c]
            break

    for c in label_candidates:
        if c in cols_lower:
            label_col = cols_lower[c]
            break

    if text_col is None or label_col is None:
        raise ValueError(
            f"Could not detect text/label columns automatically. "
            f"Available columns: {list(df.columns)}"
        )

    return text_col, label_col


def normalize_label(x: str) -> str:
    x = str(x).strip().lower()

    if x in ["positive", "pos", "2"]:
        return "positive"
    if x in ["negative", "neg", "0"]:
        return "negative"
    if x in ["neutral", "neu", "1"]:
        return "neutral"

    raise ValueError(f"Unknown label value: {x}")


def load_phrasebank(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)

    text_col, label_col = detect_columns(df)

    df = df[[text_col, label_col]].copy()
    df.columns = ["text", "label"]

    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df["label"] = df["label"].apply(normalize_label)

    df = df[df["text"].str.len() > 5].copy()
    df = df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)

    return df


# ============================================================
# Dataset
# ============================================================
class PhraseBankDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


# ============================================================
# Metrics
# ============================================================
def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    return {
        "accuracy": float(acc),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
    }


# ============================================================
# Train / Eval
# ============================================================
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Training", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_true = []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)

        loss = outputs.loss
        logits = outputs.logits

        preds = torch.argmax(logits, dim=1)

        total_loss += loss.item()
        all_preds.extend(preds.cpu().numpy().tolist())
        all_true.extend(batch["labels"].cpu().numpy().tolist())

    metrics = compute_metrics(all_true, all_preds)
    metrics["loss"] = float(total_loss / max(len(loader), 1))
    return metrics


# ============================================================
# Main
# ============================================================
def main():
    set_seed(SEED)
    print(f"Using device: {DEVICE}")

    print(f"Loading dataset from: {DATA_CSV}")
    df = load_phrasebank(DATA_CSV)
    print(f"Loaded {len(df)} rows.")

    label2id = {"negative": 0, "neutral": 1, "positive": 2}
    id2label = {0: "negative", 1: "neutral", 2: "positive"}

    df["label_id"] = df["label"].map(label2id)

    print("\nLabel distribution:")
    print(df["label"].value_counts())

    # Split: 70 train / 15 val / 15 test
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=SEED,
        stratify=df["label_id"]
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_df["label_id"]
    )

    print(f"\nTrain: {len(train_df)}")
    print(f"Val  : {len(val_df)}")
    print(f"Test : {len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = PhraseBankDataset(train_df["text"], train_df["label_id"], tokenizer, MAX_LENGTH)
    val_ds = PhraseBankDataset(val_df["text"], val_df["label_id"], tokenizer, MAX_LENGTH)
    test_ds = PhraseBankDataset(test_df["text"], test_df["label_id"], tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=id2label,
        label2id=label2id
    )
    model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    best_val_f1 = -1.0
    best_state = None
    history = []

    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{EPOCHS}")
        print(f"{'='*60}")

        train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
        val_metrics = evaluate(model, val_loader, DEVICE)

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            **{f"val_{k}": v for k, v in val_metrics.items()}
        }
        history.append(row)

        print(f"Train loss : {train_loss:.4f}")
        print(f"Val loss   : {val_metrics['loss']:.4f}")
        print(f"Val Acc    : {val_metrics['accuracy']:.4f}")
        print(f"Val F1-mac : {val_metrics['f1_macro']:.4f}")
        print(f"Val F1-wgt : {val_metrics['f1_weighted']:.4f}")

        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print("Best model updated.")

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    print("\nEvaluating on test set...")
    test_metrics = evaluate(model, test_loader, DEVICE)

    print("\nTest metrics:")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")

    # Save model + tokenizer
    ensure_dir(OUT_DIR)
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"\nSaved best model -> {OUT_DIR}")

    # Save metrics
    metrics_out = {
        "model_name": MODEL_NAME,
        "device": DEVICE,
        "n_total": int(len(df)),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "label_distribution": df["label"].value_counts().to_dict(),
        "history": history,
        "best_val_f1_macro": float(best_val_f1),
        "test_metrics": test_metrics,
        "config": {
            "max_length": MAX_LENGTH,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "seed": SEED,
        }
    }

    ensure_parent_dir(METRICS_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)

    print(f"Saved metrics -> {METRICS_PATH}")
    print("\nDone.")


if __name__ == "__main__":
    main()