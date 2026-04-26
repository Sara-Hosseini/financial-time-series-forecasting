# src/sentiment/collector.py
# Collects news for tech stocks from yfinance
# Auto-labels each news item using next-day price movement
#
# Usage:
#   python -m src.sentiment.collector

import os
import time
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
TICKERS = ["AAPL", "MSFT", "NVDA"]  # stable / moderate / volatile

# Label thresholds based on next-day price change
POSITIVE_THRESHOLD = 0.5   # > +0.5% → positive
NEGATIVE_THRESHOLD = -0.5  # < -0.5% → negative
# between -0.5% and +0.5% → neutral

OUT_DIR  = "data/processed/sentiment"
RAW_FILE = "data/processed/sentiment/news_raw.csv"


# ─────────────────────────────────────────────
# Step 1 — Fetch news from yfinance
# ─────────────────────────────────────────────
def fetch_news(ticker: str) -> pd.DataFrame:
    """Fetches recent news for a ticker from yfinance."""
    print(f"  Fetching news for {ticker}...")
    try:
        t    = yf.Ticker(ticker)
        news = t.news
        if not news:
            print(f"  No news found for {ticker}")
            return pd.DataFrame()

        rows = []
        for item in news:
            # Extract content from nested structure
            content = item.get("content", {})
            title   = content.get("title", "")
            summary = content.get("summary", "")

            # Get publish date
            pub_date = content.get("pubDate", "")
            if not pub_date:
                continue

            try:
                date = pd.to_datetime(pub_date).date()
            except Exception:
                continue

            rows.append({
                "ticker":  ticker,
                "date":    date,
                "title":   title,
                "summary": summary,
                "source":  content.get("provider", {}).get("displayName", ""),
            })

        df = pd.DataFrame(rows)
        print(f"  Found {len(df)} news items for {ticker}")
        return df

    except Exception as e:
        print(f"  Error fetching {ticker}: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────
# Step 2 — Auto-label using price movement
# ─────────────────────────────────────────────
def label_with_price(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Labels each news item based on next-day price change.
    positive : next-day return > +0.5%
    negative : next-day return < -0.5%
    neutral  : between -0.5% and +0.5%
    """
    if df.empty:
        return df

    print(f"  Labeling {ticker} news with price movement...")

    # Download price data
    try:
        price_df = yf.download(ticker, period="6mo",
                               interval="1d", auto_adjust=True, progress=False)
        if isinstance(price_df.columns, pd.MultiIndex):
            price_df.columns = [col[0] for col in price_df.columns]
        price_df.index = pd.to_datetime(price_df.index).date
    except Exception as e:
        print(f"  Could not download prices for {ticker}: {e}")
        df["label"]       = "neutral"
        df["price_change"] = None
        return df

    # Calculate next-day returns
    price_df["next_return"] = price_df["Close"].pct_change().shift(-1) * 100

    # Match news date to price
    labels       = []
    price_changes = []

    for _, row in df.iterrows():
        news_date = row["date"]
        if news_date in price_df.index:
            ret = price_df.loc[news_date, "next_return"]
            try:
                ret = float(ret)
                if ret > POSITIVE_THRESHOLD:
                    label = "positive"
                elif ret < NEGATIVE_THRESHOLD:
                    label = "negative"
                else:
                    label = "neutral"
                labels.append(label)
                price_changes.append(round(ret, 4))
            except Exception:
                labels.append("neutral")
                price_changes.append(None)
        else:
            labels.append("neutral")
            price_changes.append(None)

    df = df.copy()
    df["label"]        = labels
    df["price_change"] = price_changes
    return df


# ─────────────────────────────────────────────
# Step 3 — Combine title + summary as text
# ─────────────────────────────────────────────
def prepare_text(df: pd.DataFrame) -> pd.DataFrame:
    """Combines title and summary into a single text field."""
    df = df.copy()
    df["text"] = df.apply(
        lambda r: r["title"] if not r["summary"]
                  else f"{r['title']}. {r['summary']}",
        axis=1
    )
    # Remove empty texts
    df = df[df["text"].str.strip().str.len() > 10].copy()
    return df


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def collect_and_label():
    os.makedirs(OUT_DIR, exist_ok=True)

    all_dfs = []

    # Load existing data if available
    if os.path.exists(RAW_FILE):
        existing = pd.read_csv(RAW_FILE)
        existing["date"] = pd.to_datetime(existing["date"]).dt.date
        print(f"Loaded {len(existing)} existing news items")
        all_dfs.append(existing)

    # Fetch new news for each ticker
    for i, ticker in enumerate(TICKERS):
        print(f"\n[{i+1}/{len(TICKERS)}] Processing {ticker}...")

        df = fetch_news(ticker)
        if df.empty:
            continue

        df = label_with_price(df, ticker)
        df = prepare_text(df)
        all_dfs.append(df)

        time.sleep(1)  # be polite to yfinance

    if not all_dfs:
        print("No news collected.")
        return

    # Merge and deduplicate
    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["ticker", "date", "title"])
    combined = combined.sort_values(["ticker", "date"])

    # Save
    combined.to_csv(RAW_FILE, index=False, encoding="utf-8-sig")
    print(f"\nSaved {len(combined)} news items -> {RAW_FILE}")

    # Stats
    print(f"\nLabel distribution:")
    print(combined["label"].value_counts())
    print(f"\nBy ticker:")
    print(combined.groupby("ticker")["label"].value_counts().unstack(fill_value=0))

    return combined


if __name__ == "__main__":
    collect_and_label()