# src/analysis_pipeline.py

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
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

    df = df.dropna(subset=["Open", "Close"])

    # Numerical features
    df["log_return"] = np.log(df["Close"]).diff()
    df["vol_20"] = df["log_return"].rolling(20).std()

    return df


def build_daily_sentiment_10days(
    api_key: str,
    ticker: str,
    *,
    start_yyyymmdd: str = "20251201",
    end_yyyymmdd: str = "20251210",
    limit_per_call: int = 200,
    sleep_seconds: int = 13,
) -> pd.DataFrame:
    """
    Fetch Alpha Vantage NEWS_SENTIMENT in smaller windows to overcome free API limitations.
    Default: 10-day window split into 3 chunks:
      2025-12-01..03, 04..06, 07..10

    Returns DataFrame with columns:
      Date, sentiment_mean, news_count, relevance_mean
    """

    # Hard-coded 10-day split for reliability (you can generalize later)
    windows = [
        ("20251201T0000", "20251203T2359"),
        ("20251204T0000", "20251206T2359"),
        ("20251207T0000", "20251210T2359"),
    ]

    all_raw = []
    all_daily = []

    for i, (tf, tt) in enumerate(windows):
        # Respect free-tier rate limits
        if i > 0:
            time.sleep(sleep_seconds)

        data = fetch_news_sentiment(
            api_key=api_key,
            tickers=ticker,
            limit=limit_per_call,
            time_from=tf,
            time_to=tt,
            sort="EARLIEST",
        )

        df_raw, daily = news_json_to_df(data, target_ticker=ticker)

        # Helpful debug (keep it; it shows coverage)
        print(f"[{ticker}] window {tf} -> {tt} | raw_rows={len(df_raw)}")
        if not df_raw.empty:
            print("   raw min/max:", df_raw["time_published"].min(), "->", df_raw["time_published"].max())

        if not df_raw.empty:
            all_raw.append(df_raw)
        if not daily.empty:
            all_daily.append(daily)

    if not all_raw:
        return pd.DataFrame(columns=["Date", "sentiment_mean", "news_count", "relevance_mean"])

    raw = pd.concat(all_raw, ignore_index=True)
    daily = pd.concat(all_daily, ignore_index=True)

    # daily from sentiment_collection.py: date, sentiment_mean, sentiment_median, n_news
    daily_agg = (
        daily.groupby("date", as_index=False)
        .agg(
            sentiment_mean=("sentiment_mean", "mean"),
            news_count=("n_news", "sum"),
        )
        .rename(columns={"date": "Date"})
    )
    daily_agg["Date"] = pd.to_datetime(daily_agg["Date"])

    # relevance_mean from raw
    raw["Date"] = raw["time_published"].dt.floor("D")
    rel = raw.groupby("Date", as_index=False).agg(relevance_mean=("relevance_score", "mean"))
    rel["Date"] = pd.to_datetime(rel["Date"])

    out = daily_agg.merge(rel, on="Date", how="left")
    out["relevance_mean"] = out["relevance_mean"].fillna(0.0)

    # Keep only requested 10 days (safety)
    out = out.sort_values("Date")
    out = out[(out["Date"] >= pd.to_datetime("2025-12-01")) & (out["Date"] <= pd.to_datetime("2025-12-10"))]

    return out[["Date", "sentiment_mean", "news_count", "relevance_mean"]]


def run_pipeline(
    ticker: str,
    price_csv_path: str,
    *,
    out_dir: str = "data/processed",
) -> pd.DataFrame:
    """
    End-to-end pipeline:
    1) Load price data + compute numerical features
    2) Fetch news sentiment (10-day chunked)
    3) Merge into one unified dataset
    4) Save merged CSV output
    """
    load_dotenv()
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise RuntimeError("ALPHAVANTAGE_API_KEY is not found in .env")

    price = load_price_data(price_csv_path)

    sentiment_daily = build_daily_sentiment_10days(
        api_key,
        ticker,
        limit_per_call=200,
        sleep_seconds=13,
    )

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


def build_sample_for_supervisor(
    merged: pd.DataFrame,
    *,
    ticker: str,
    start: str,
    end: str,
    out_dir: str = "data/processed",
) -> pd.DataFrame:
    """
    Supervisor-ready sample (short time window):
    Date, Open, Close, Return, Abs_Return, sentiment_mean, Sentiment_Lag1,
    news_count, relevance_mean, Impact_score
    """
    df = merged.copy().sort_index()
    df = df.loc[pd.to_datetime(start): pd.to_datetime(end)].copy()

    if "Open" not in df.columns or "Close" not in df.columns:
        raise KeyError("Expected 'Open' and 'Close' columns in merged price data.")

    df["Return"] = (df["Close"] - df["Open"]) / df["Open"]
    df["Abs_Return"] = df["Return"].abs()

    if "sentiment_mean" in df.columns:
        df["Sentiment_Lag1"] = df["sentiment_mean"].shift(1)
        df["Impact_score"] = df["sentiment_mean"] * df["Return"]
    else:
        df["Sentiment_Lag1"] = 0.0
        df["Impact_score"] = 0.0

    cols = [
        "Open",
        "Close",
        "Return",
        "Abs_Return",
        "sentiment_mean",
        "Sentiment_Lag1",
        "news_count",
        "relevance_mean",
        "Impact_score",
    ]
    cols = [c for c in cols if c in df.columns]

    out = df[cols].copy()
    out.index.name = "Date"

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    sample_csv = out_path / f"{ticker}_SAMPLE_price_news_impact_{start}_to_{end}.csv"
    out.to_csv(sample_csv, index=True)

    print("Saved supervisor sample to:", sample_csv)
    return out


if __name__ == "__main__":
    ticker = "AAPL"
    price_csv = "data/raw/AAPL_daily_10y_yahoo.csv"

    df_merged = run_pipeline(ticker, price_csv, out_dir="data/processed")

    print("Pipeline complete.")
    print("Saved merged CSV to:", f"data/processed/{ticker}_price_sentiment_features.csv")
    print(df_merged.tail(5))

    # ✅ 10-day sample
    sample = build_sample_for_supervisor(
        df_merged,
        ticker=ticker,
        start="2025-12-01",
        end="2025-12-10",
        out_dir="data/processed",
    )
    print(sample.head())
