# src/features/merge_event_features.py
# Merge Event Features into the Model Dataset
#
# Combines:
#   - Existing price + sentiment merged CSV (outputs/{ticker}_price_sentiment_merged.csv)
#   - Daily event features (results/revised_experiments/event_features/event_features_daily.csv)
#
# Produces a single multi-ticker dataset:
#   results/revised_experiments/event_features/price_sentiment_event_dataset.csv
#
# Feature totals:
#   9 price technical features  (computed by add_features)
#   4 core sentiment features   (avg_sentiment_score, positive_ratio,
#                                 negative_ratio, news_count)
#   11 event features           (9 binary flags + event_count + event_intensity)
#   ── total 24 model features ──
#
# Trading days with no event news receive zeros for all event columns.
# The original outputs/ files are NEVER overwritten.
#
# Usage:
#   python -m src.features.merge_event_features

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.features.event_features import (
    run_event_feature_extraction,
    EVENT_COLS,
    OUTPUT_CSV as EVENT_CSV,
    TICKERS,
)

MERGED_PRICE_TMPL  = "outputs/{ticker}_price_sentiment_merged.csv"
OUTPUT_DIR         = "results/revised_experiments/event_features"
OUTPUT_DATASET_CSV = os.path.join(OUTPUT_DIR, "price_sentiment_event_dataset.csv")

# 11 event feature columns (9 binary + 2 aggregates)
EVENT_FEATURE_COLS = EVENT_COLS + ["event_count", "event_intensity"]

# The 4 core sentiment features used by the thesis models
CORE_SENTIMENT_COLS = [
    "avg_sentiment_score", "positive_ratio", "negative_ratio", "news_count",
]

# All 15 sentiment cols present in the merged CSV (needed for completeness check)
ALL_SENTIMENT_COLS = [
    "news_count", "avg_sentiment_score", "avg_positive_prob", "avg_negative_prob",
    "avg_neutral_prob", "max_positive_prob", "max_negative_prob", "pos_news_count",
    "neg_news_count", "neu_news_count", "sentiment_balance", "positive_ratio",
    "negative_ratio", "neutral_ratio", "has_news",
]


def _load_price_sentiment(ticker: str) -> pd.DataFrame:
    """Load the existing price + sentiment merged CSV for one ticker."""
    path = MERGED_PRICE_TMPL.format(ticker=ticker)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Merged price+sentiment CSV not found: {path}\n"
            f"Run the full pipeline first (run_pipeline.py or lstm_quantile.py)."
        )
    df = pd.read_csv(path)

    # Normalise the date column to lowercase 'date'
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["date"])

    # Numeric coercion for price and sentiment columns
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ALL_SENTIMENT_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        else:
            df[c] = 0.0

    df["ticker"] = ticker
    return df.sort_values("date").reset_index(drop=True)


def merge_event_features(rebuild_events: bool = False) -> pd.DataFrame:
    """
    Build the event-enhanced model dataset for all three tickers.

    Parameters
    ----------
    rebuild_events : bool
        If True, re-run keyword extraction even if event_features_daily.csv exists.
        Default False (use cached file if available).

    Returns
    -------
    pd.DataFrame
        Multi-ticker dataset saved to OUTPUT_DATASET_CSV.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Step 1: ensure event features exist ──────────────────────
    if rebuild_events or not os.path.exists(EVENT_CSV):
        print("  Building event features from news_raw.csv ...")
        run_event_feature_extraction()
    else:
        print(f"  Using cached event features: {EVENT_CSV}")

    event_df = pd.read_csv(EVENT_CSV)
    event_df["date"] = pd.to_datetime(event_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    event_df = event_df.dropna(subset=["date"])
    print(f"  Event features loaded: {len(event_df)} rows")

    # ── Step 2: load and merge per ticker ────────────────────────
    frames = []
    for ticker in TICKERS:
        print(f"\n  Processing {ticker} ...")

        df = _load_price_sentiment(ticker)
        print(f"    Price+sentiment: {len(df)} trading days")

        # Filter event features to this ticker
        ev = event_df[event_df["ticker"] == ticker][
            ["date"] + EVENT_FEATURE_COLS
        ].copy()
        print(f"    Event news days: {len(ev)}")

        # Left join: every trading day is kept; no-event days → 0
        merged = df.merge(ev, on="date", how="left")
        for col in EVENT_FEATURE_COLS:
            merged[col] = merged[col].fillna(0.0)

        event_days = (merged["event_count"] > 0).sum()
        print(f"    After merge: {len(merged)} rows | "
              f"{event_days} ({100*event_days/len(merged):.1f}%) event days")

        frames.append(merged)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)

    # ── Step 3: save ─────────────────────────────────────────────
    combined.to_csv(OUTPUT_DATASET_CSV, index=False)

    print(f"\n  {'='*50}")
    print(f"  Saved {len(combined)} rows x {len(combined.columns)} cols")
    print(f"  -> {OUTPUT_DATASET_CSV}")
    print(f"\n  Feature summary:")
    print(f"    Price technical cols  : (computed by add_features at load time)")
    print(f"    Core sentiment cols   : {CORE_SENTIMENT_COLS}")
    print(f"    Event feature cols    : {EVENT_FEATURE_COLS}")
    print(f"  {'='*50}")

    return combined


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--rebuild-events", action="store_true",
                   help="Re-run keyword extraction even if cached file exists")
    args = p.parse_args()
    merge_event_features(rebuild_events=args.rebuild_events)
