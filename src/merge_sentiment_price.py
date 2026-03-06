import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd


# ============================================================
# Config
# ============================================================
PRICE_CSV = "data/raw/AAPL_daily_10y_yahoo.csv"
SENTIMENT_CSV = "outputs/AAPL_news_daily.csv"
OUT_CSV = "outputs/AAPL_price_sentiment_merged.csv"


# ============================================================
# Helpers
# ============================================================
def ensure_output_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def load_price_data(csv_path: str) -> pd.DataFrame:
    """
    Loads AAPL daily price data and standardizes the date column.
    Supports the same Yahoo-style format you used before.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Price file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # handle odd Yahoo export format
    first_cell = str(df.iloc[0, 0]).strip().lower() if len(df) >= 1 else ""
    if first_cell == "ticker":
        df = df.iloc[2:].copy()

    if "Price" in df.columns and "Date" not in df.columns:
        df = df.rename(columns={"Price": "Date"})
    if "Date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Date"})

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()

    # numeric conversion
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    keep = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].dropna(subset=["Close"]).copy()

    df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
    return df


def load_sentiment_data(csv_path: str) -> pd.DataFrame:
    """
    Loads daily sentiment features and standardizes the date column.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Sentiment file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "trading_date" not in df.columns:
        raise ValueError("Sentiment CSV must contain 'trading_date'.")

    df["trading_date"] = pd.to_datetime(df["trading_date"], errors="coerce")
    df = df.dropna(subset=["trading_date"]).copy()

    df = df.sort_values("trading_date").drop_duplicates(subset=["trading_date"], keep="last").reset_index(drop=True)
    return df


def merge_price_and_sentiment(price_df: pd.DataFrame, sent_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges price and daily sentiment features on date.
    Missing sentiment days are filled safely.
    """
    merged = price_df.merge(
        sent_df,
        left_on="Date",
        right_on="trading_date",
        how="left"
    )

    sentiment_cols = [
        c for c in sent_df.columns
        if c != "trading_date"
    ]

    # no-news days -> fill with neutral / zero-like defaults
    fill_zero_cols = [
        "news_count",
        "avg_sentiment_score",
        "avg_positive_prob",
        "avg_negative_prob",
        "avg_neutral_prob",
        "max_positive_prob",
        "max_negative_prob",
        "pos_news_count",
        "neg_news_count",
        "neu_news_count",
        "sentiment_balance",
        "positive_ratio",
        "negative_ratio",
        "neutral_ratio",
    ]

    for col in fill_zero_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0.0)

    # optional indicator: was there at least one news item that day?
    merged["has_news"] = 0
    if "news_count" in merged.columns:
        merged["has_news"] = (merged["news_count"] > 0).astype(int)

    # remove duplicated date column from sentiment side
    if "trading_date" in merged.columns:
        merged = merged.drop(columns=["trading_date"])

    merged = merged.sort_values("Date").reset_index(drop=True)
    return merged


def main():
    print(f"Loading price data from: {PRICE_CSV}")
    price_df = load_price_data(PRICE_CSV)
    print(f"Loaded {len(price_df)} price rows.")

    print(f"Loading sentiment data from: {SENTIMENT_CSV}")
    sent_df = load_sentiment_data(SENTIMENT_CSV)
    print(f"Loaded {len(sent_df)} daily sentiment rows.")

    merged = merge_price_and_sentiment(price_df, sent_df)

    ensure_output_dir(OUT_CSV)
    merged.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"Saved merged dataset -> {OUT_CSV}")
    print("\nMerged preview:")
    print(merged.tail(10))

    if "has_news" in merged.columns:
        print("\nNews coverage:")
        print(merged["has_news"].value_counts())

    print("\nDone.")


if __name__ == "__main__":
    main()