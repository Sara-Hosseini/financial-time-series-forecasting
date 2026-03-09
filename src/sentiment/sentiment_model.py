"""
sentiment_model.py
────────────────────────────────────────────────────────────
Loads fine-tuned FinBERT+LoRA model and scores daily AAPL news.

Input  : data/processed/sentiment/news_raw.csv  (from collector.py)
Output : outputs/AAPL_news_scored.csv           (article-level)
         outputs/AAPL_news_daily.csv            (daily aggregated)

Usage:
    python -m src.sentiment.sentiment_model
────────────────────────────────────────────────────────────
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel


# ============================================================
# Config
# ============================================================
BASE_MODEL_NAME  = "ProsusAI/finbert"
LORA_MODEL_DIR   = "outputs/models/finbert_lora_best"

INPUT_CSV        = "data/processed/sentiment/news_raw.csv"
ARTICLE_OUT_CSV  = "outputs/AAPL_news_scored.csv"
DAILY_OUT_CSV    = "outputs/AAPL_news_daily.csv"

BATCH_SIZE  = 16
MAX_LENGTH  = 256
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

label2id = {"negative": 0, "neutral": 1, "positive": 2}
id2label  = {0: "negative", 1: "neutral", 2: "positive"}


# ============================================================
# Helpers
# ============================================================
def ensure_output_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def load_news(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if "date" not in df.columns:
        raise ValueError("Input CSV must contain a 'date' column.")
    if "title" not in df.columns:
        raise ValueError("Input CSV must contain a 'title' column.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()

    if "summary" not in df.columns:
        df["summary"] = ""

    df["title"]   = df["title"].fillna("").astype(str).str.strip()
    df["summary"] = df["summary"].fillna("").astype(str).str.strip()

    df["text"] = np.where(
        df["summary"].str.len() > 0,
        df["title"] + ". " + df["summary"],
        df["title"]
    )

    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df = df[df["text"].str.len() > 10].copy()

    dedup_cols = [c for c in ["date", "title"] if c in df.columns]
    df = df.drop_duplicates(subset=dedup_cols)
    df = df.sort_values("date").reset_index(drop=True)

    return df


def load_finbert_lora():
    """Loads FinBERT base model + LoRA adapter."""
    print(f"Loading base model: {BASE_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=3,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    print(f"Loading LoRA adapter: {LORA_MODEL_DIR}")
    model = PeftModel.from_pretrained(base_model, LORA_MODEL_DIR)
    model = model.merge_and_unload()   # merge LoRA weights for faster inference
    model.to(DEVICE)
    model.eval()

    return tokenizer, model


def predict_batch(texts, tokenizer, model):
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()

    pred_ids = probs.argmax(axis=1)

    results = []
    for i in range(len(texts)):
        pos = float(probs[i][label2id["positive"]])
        neg = float(probs[i][label2id["negative"]])
        neu = float(probs[i][label2id["neutral"]])
        results.append({
            "pred_label":     id2label[int(pred_ids[i])],
            "positive_prob":  pos,
            "negative_prob":  neg,
            "neutral_prob":   neu,
            "sentiment_score": pos - neg,   # [-1, +1]
        })
    return results


def score_news(df: pd.DataFrame, tokenizer, model) -> pd.DataFrame:
    all_results = []
    for start in tqdm(range(0, len(df), BATCH_SIZE), desc="Scoring news"):
        batch_texts  = df["text"].iloc[start:start + BATCH_SIZE].tolist()
        all_results.extend(predict_batch(batch_texts, tokenizer, model))

    sent_df = pd.DataFrame(all_results)
    return pd.concat([df.reset_index(drop=True), sent_df], axis=1)


def build_daily_sentiment(df_scored: pd.DataFrame) -> pd.DataFrame:
    """Aggregates article-level sentiment into daily features."""
    df = df_scored.loc[:, ~df_scored.columns.duplicated()].copy()
    df["trading_date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["trading_date"])

    rows = []
    for trading_date, g in df.groupby("trading_date"):
        n   = len(g)
        pos = int((g["pred_label"] == "positive").sum())
        neg = int((g["pred_label"] == "negative").sum())
        neu = int((g["pred_label"] == "neutral").sum())

        rows.append({
            "trading_date":       trading_date,
            "news_count":         n,
            "avg_sentiment_score": float(g["sentiment_score"].mean()),
            "avg_positive_prob":  float(g["positive_prob"].mean()),
            "avg_negative_prob":  float(g["negative_prob"].mean()),
            "avg_neutral_prob":   float(g["neutral_prob"].mean()),
            "max_positive_prob":  float(g["positive_prob"].max()),
            "max_negative_prob":  float(g["negative_prob"].max()),
            "pos_news_count":     pos,
            "neg_news_count":     neg,
            "neu_news_count":     neu,
            "sentiment_balance":  pos - neg,
            "positive_ratio":     pos / n if n > 0 else 0.0,
            "negative_ratio":     neg / n if n > 0 else 0.0,
            "neutral_ratio":      neu / n if n > 0 else 0.0,
        })

    return pd.DataFrame(rows).sort_values("trading_date").reset_index(drop=True)


# ============================================================
# Main
# ============================================================
def main():
    print(f"Device: {DEVICE}")

    print(f"\nLoading news from: {INPUT_CSV}")
    df = load_news(INPUT_CSV)
    print(f"Loaded {len(df)} news rows.")

    tokenizer, model = load_finbert_lora()

    print("\nScoring news...")
    df_scored = score_news(df, tokenizer, model)

    ensure_output_dir(ARTICLE_OUT_CSV)
    ensure_output_dir(DAILY_OUT_CSV)

    df_scored.to_csv(ARTICLE_OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved article-level → {ARTICLE_OUT_CSV}")

    daily_df = build_daily_sentiment(df_scored)
    daily_df.to_csv(DAILY_OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved daily sentiment → {DAILY_OUT_CSV}")

    print("\nLabel distribution:")
    print(df_scored["pred_label"].value_counts())

    print("\nDaily sentiment preview:")
    print(daily_df.head())

    print("\n✅ Done.")


if __name__ == "__main__":
    main()