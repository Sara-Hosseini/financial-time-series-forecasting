import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ============================================================
# Config
# ============================================================
MODEL_NAME = "outputs/models/finbert_phrasebank_best"

INPUT_CSV = "outputs/AAPL_news_raw.csv"
ARTICLE_OUT_CSV = "outputs/AAPL_news_scored.csv"
DAILY_OUT_CSV = "outputs/AAPL_news_daily.csv"

BATCH_SIZE = 16
MAX_LENGTH = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Helpers
# ============================================================
def ensure_output_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def load_news(csv_path: str) -> pd.DataFrame:
    """
    Loads raw news CSV and prepares a single text column.
    Expected columns:
        - date
        - title
        - summary   (optional but preferred)
        - ticker    (optional)
    """
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

    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["summary"] = df["summary"].fillna("").astype(str).str.strip()

    # combine title + summary
    df["text"] = np.where(
        df["summary"].str.len() > 0,
        df["title"] + ". " + df["summary"],
        df["title"]
    )

    # remove empty / too-short texts
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df = df[df["text"].str.len() > 10].copy()

    # remove duplicates if possible
    dedup_cols = [c for c in ["date", "title", "summary"] if c in df.columns]
    if dedup_cols:
        df = df.drop_duplicates(subset=dedup_cols)

    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_finbert(model_name: str = MODEL_NAME):
    """
    Loads FinBERT tokenizer and sequence classification model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


def get_label_indices(id2label: dict) -> dict:
    """
    Maps label names to class indices safely.
    """
    label2id = {str(v).lower(): int(k) for k, v in id2label.items()}
    required = {"positive", "negative", "neutral"}
    if not required.issubset(set(label2id.keys())):
        raise ValueError(f"Unexpected model labels: {label2id}")
    return label2id


def predict_batch(texts, tokenizer, model, label2id):
    """
    Predicts sentiment probabilities for a batch of texts.
    """
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    with torch.no_grad():
        outputs = model(**enc)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

    pred_ids = probs.argmax(axis=1)

    id2label = model.config.id2label
    pred_labels = [str(id2label[int(i)]).lower() for i in pred_ids]

    results = []
    for i in range(len(texts)):
        positive_prob = float(probs[i][label2id["positive"]])
        negative_prob = float(probs[i][label2id["negative"]])
        neutral_prob = float(probs[i][label2id["neutral"]])

        results.append({
            "pred_label": pred_labels[i],
            "positive_prob": positive_prob,
            "negative_prob": negative_prob,
            "neutral_prob": neutral_prob,
            "sentiment_score": positive_prob - negative_prob,
        })

    return results


def score_news(df: pd.DataFrame, tokenizer, model) -> pd.DataFrame:
    """
    Runs FinBERT inference over all news rows.
    """
    id2label = model.config.id2label
    label2id = get_label_indices(id2label)

    all_results = []

    for start in tqdm(range(0, len(df), BATCH_SIZE), desc="Scoring Apple news"):
        batch_texts = df["text"].iloc[start:start + BATCH_SIZE].tolist()
        batch_results = predict_batch(batch_texts, tokenizer, model, label2id)
        all_results.extend(batch_results)

    sent_df = pd.DataFrame(all_results)
    out = pd.concat([df.reset_index(drop=True), sent_df], axis=1)
    return out


def build_daily_sentiment(df_scored: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates article-level sentiment into daily sentiment features.
    Safe version against duplicate-column pandas issues.
    """

    # remove duplicate column names if any
    df = df_scored.loc[:, ~df_scored.columns.duplicated()].copy()

    # keep only needed columns
    needed_cols = [
        "date",
        "text",
        "pred_label",
        "sentiment_score",
        "positive_prob",
        "negative_prob",
        "neutral_prob",
    ]
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in scored news data: {missing}")

    df = df[needed_cols].copy()

    # normalize date
    df["trading_date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["trading_date"]).copy()

    rows = []

    for trading_date, g in df.groupby("trading_date"):
        news_count = int(len(g))
        pos_count = int((g["pred_label"] == "positive").sum())
        neg_count = int((g["pred_label"] == "negative").sum())
        neu_count = int((g["pred_label"] == "neutral").sum())

        row = {
            "trading_date": trading_date,
            "news_count": news_count,
            "avg_sentiment_score": float(g["sentiment_score"].mean()),
            "avg_positive_prob": float(g["positive_prob"].mean()),
            "avg_negative_prob": float(g["negative_prob"].mean()),
            "avg_neutral_prob": float(g["neutral_prob"].mean()),
            "max_positive_prob": float(g["positive_prob"].max()),
            "max_negative_prob": float(g["negative_prob"].max()),
            "pos_news_count": pos_count,
            "neg_news_count": neg_count,
            "neu_news_count": neu_count,
        }

        row["sentiment_balance"] = pos_count - neg_count
        row["positive_ratio"] = pos_count / news_count if news_count > 0 else 0.0
        row["negative_ratio"] = neg_count / news_count if news_count > 0 else 0.0
        row["neutral_ratio"] = neu_count / news_count if news_count > 0 else 0.0

        rows.append(row)

    daily = pd.DataFrame(rows).sort_values("trading_date").reset_index(drop=True)
    return daily

def main():
    print(f"Loading news from: {INPUT_CSV}")
    df = load_news(INPUT_CSV)
    print(f"Loaded {len(df)} valid Apple news rows.")

    print(f"Loading FinBERT model: {MODEL_NAME}")
    tokenizer, model = load_finbert(MODEL_NAME)

    print("Scoring news...")
    df_scored = score_news(df, tokenizer, model)

    ensure_output_dir(ARTICLE_OUT_CSV)
    ensure_output_dir(DAILY_OUT_CSV)

    df_scored.to_csv(ARTICLE_OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved article-level sentiment -> {ARTICLE_OUT_CSV}")

    daily_df = build_daily_sentiment(df_scored)
    daily_df.to_csv(DAILY_OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved daily sentiment -> {DAILY_OUT_CSV}")

    print("\nLabel distribution:")
    print(df_scored["pred_label"].value_counts())

    print("\nDaily sentiment preview:")
    print(daily_df.head())

    print("\nDone.")


if __name__ == "__main__":
    main()