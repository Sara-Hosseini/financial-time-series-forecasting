# src/analysis_pipeline.py

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Uses your existing sentiment_collection.py
from src.sentiment_collection import fetch_news_sentiment, news_json_to_df

def load_price_data(path: str) -> pd.DataFrame:
    """
    Load Yahoo Finance CSV (daily) and compute numerical features:
    - log_return
    - rolling volatility (20-day)
    """
    df = pd.read_csv(path)

    # Force first column to be Date (robust even if column name changes)
    df.rename(columns={df.columns[0]: "Date"}, inplace=True)

    # Parse dates and set index
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()

    # Convert core columns to numeric
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Close"])

    # Numerical features
    df["log_return"] = np.log(df["Close"]).diff()
    df["vol_20"] = df["log_return"].rolling(20).std()

    return df


def build_daily_sentiment(api_key: str, ticker: str, limit: int = 50) -> pd.DataFrame:
    """
    Fetch Alpha Vantage news sentiment and aggregate daily:
    - sentiment_mean
    - news_count
    """
    data = fetch_news_sentiment(api_key=api_key, tickers=ticker, limit=limit)
    df = news_json_to_df(data, target_ticker=ticker)

    if df.empty:
        return pd.DataFrame(columns=["Date", "sentiment_mean", "news_count", "relevance_mean"])

    df["Date"] = df["time_published"].dt.floor("D")

    daily = (
        df.groupby("Date")
        .agg(
            sentiment_mean=("sentiment_score", "mean"),
            news_count=("sentiment_score", "size"),
            relevance_mean=("relevance_score", "mean"),
        )
        .reset_index()
    )

    return daily


def run_pipeline(
    ticker: str,
    price_csv_path: str,
    *,
    sentiment_limit: int = 50,
    out_dir: str = "data/processed"
) -> pd.DataFrame:
    """
    End-to-end pipeline:
    1) Load price data + compute numerical features
    2) Fetch news sentiment + daily aggregate
    3) Merge into one unified dataset
    4) Save CSV output
    """
    load_dotenv()
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise RuntimeError("ALPHAVANTAGE_API_KEY is not found in .env")

    price = load_price_data(price_csv_path)
    sentiment_daily = build_daily_sentiment(api_key, ticker, limit=sentiment_limit)

    if not sentiment_daily.empty:
        sentiment_daily = sentiment_daily.set_index("Date")

    merged = price.join(sentiment_daily, how="left")

    # Fill missing sentiment days as neutral (0) and no-news (0)
    for col in ["sentiment_mean", "news_count", "relevance_mean"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)

    # Save output
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    merged_csv = out_path / f"{ticker}_price_sentiment_features.csv"
    merged.to_csv(merged_csv)

    return merged


def save_price_vs_sentiment_plot(df: pd.DataFrame, ticker: str, out_dir: str = "data/processed") -> str:
    """
    Save a dual-axis plot: Close vs sentiment_mean
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(12, 4))

    ax1.plot(df.index, df["Close"])
    ax1.set_ylabel("Close Price")

    ax2 = ax1.twinx()
    if "sentiment_mean" in df.columns:
        ax2.plot(df.index, df["sentiment_mean"], alpha=0.4)
    ax2.set_ylabel("Daily Sentiment (mean)")

    plt.title(f"{ticker} Close Price vs Daily News Sentiment")
    plt.tight_layout()

    png_path = out_path / f"{ticker}_close_vs_sentiment.png"
    plt.savefig(png_path, dpi=200)
    plt.close(fig)

    return str(png_path)


if __name__ == "__main__":
    # --- Single stock run (AAPL) ---
    ticker = "AAPL"
    price_csv = "data/raw/AAPL_daily_10y_yahoo.csv"

    df_merged = run_pipeline(ticker, price_csv, sentiment_limit=50)
    plot_path = save_price_vs_sentiment_plot(df_merged, ticker)

    print("Pipeline complete.")
    print("Saved merged CSV to:", f"data/processed/{ticker}_price_sentiment_features.csv")
    print("Saved plot to:", plot_path)
    print(df_merged.tail(5))

   
