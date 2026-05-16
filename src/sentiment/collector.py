# src/sentiment/collector.py
# Collects news for AAPL, MSFT, NVDA from multiple sources:
#   1. Alpha Vantage News API (5 years history, free)
#   2. GDELT Project (10 years history, free)
#   3. yfinance (recent news, fallback)
# Auto-labels each news item using next-day price movement
#
# Setup:
#   pip install newsapi-python requests yfinance pandas
#   Set ALPHA_VANTAGE_KEY in environment or below
#
# Usage:
#   python -m src.sentiment.collector

import os
import time
import requests
from datetime import datetime, timedelta, date
import pandas as pd
import yfinance as yf

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
TICKERS = ["AAPL", "MSFT", "NVDA"]

# Alpha Vantage free API key — get from https://www.alphavantage.co/support/#api-key
ALPHA_VANTAGE_KEY = os.environ.get("ALPHA_VANTAGE_KEY", "I6X7B1YLOHY2SPKW")

# Label thresholds
POSITIVE_THRESHOLD =  0.5   # > +0.5% next-day return → positive
NEGATIVE_THRESHOLD = -0.5   # < -0.5% next-day return → negative

# Date range for historical collection
START_DATE = "2015-01-01"
END_DATE   = datetime.today().strftime("%Y-%m-%d")

OUT_DIR  = "data/processed/sentiment"
RAW_FILE = "data/processed/sentiment/news_raw.csv"

# Ticker → search keywords for GDELT
TICKER_KEYWORDS = {
    "AAPL": "Apple AAPL stock",
    "MSFT": "Microsoft MSFT stock",
    "NVDA": "NVIDIA NVDA stock",
}

# Company names for Alpha Vantage
TICKER_COMPANY = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "NVIDIA",
}


# ─────────────────────────────────────────────────────────────
# Source 1 — Alpha Vantage News API
# Up to 5 years of history, free tier: 25 requests/day
# ─────────────────────────────────────────────────────────────
def fetch_alpha_vantage(ticker: str) -> pd.DataFrame:
    """
    Fetches news from Alpha Vantage News Sentiment API.
    Free key gives access to historical financial news with timestamps.
    """
    print(f"  [Alpha Vantage] Fetching news for {ticker}...")

    if ALPHA_VANTAGE_KEY == "demo":
        print("  [Alpha Vantage] No API key set — skipping. "
              "Get free key at alphavantage.co")
        return pd.DataFrame()

    url = "https://www.alphavantage.co/query"
    params = {
        "function":  "NEWS_SENTIMENT",
        "tickers":   ticker,
        "apikey":    ALPHA_VANTAGE_KEY,
        "limit":     1000,
        "sort":      "EARLIEST",
        "time_from": START_DATE.replace("-", "") + "T0000",
    }

    rows = []
    try:
        resp = requests.get(url, params=params, timeout=30)
        data = resp.json()

        feed = data.get("feed", [])
        print(f"  [Alpha Vantage] Found {len(feed)} articles for {ticker}")

        for item in feed:
            pub = item.get("time_published", "")
            if not pub:
                continue
            try:
                pub_date = datetime.strptime(pub[:8], "%Y%m%d").date()
            except ValueError:
                continue

            title   = item.get("title", "")
            summary = item.get("summary", "")

            # Alpha Vantage provides overall_sentiment_score
            av_score = item.get("overall_sentiment_score", None)

            rows.append({
                "ticker":   ticker,
                "date":     pub_date,
                "title":    title,
                "summary":  summary,
                "source":   "alpha_vantage",
                "av_score": av_score,
            })

        time.sleep(12)  # respect free tier rate limit (5 req/min)

    except Exception as e:
        print(f"  [Alpha Vantage] Error: {e}")

    df = pd.DataFrame(rows)
    print(f"  [Alpha Vantage] {ticker}: {len(df)} articles collected")
    return df


# ─────────────────────────────────────────────────────────────
# Source 2 — GDELT Project
# Free, no API key needed, covers 2015 to present
# ─────────────────────────────────────────────────────────────
def fetch_gdelt(ticker: str) -> pd.DataFrame:
    """
    Fetches news from GDELT Project API.
    Completely free, no registration required.
    Covers global news from 2015 onwards.
    """
    print(f"  [GDELT] Fetching news for {ticker}...")

    keyword = TICKER_KEYWORDS.get(ticker, ticker)
    url = "https://api.gdeltproject.org/api/v2/doc/doc"

    rows = []
    # GDELT returns max 250 per request — paginate by year
    years = range(2015, datetime.today().year + 1)

    for year in years:
        params = {
            "query":           f'"{keyword}"',
            "mode":            "artlist",
            "startdatetime":   f"{year}0101000000",
            "enddatetime":     f"{year}1231235959",
            "maxrecords":      250,
            "format":          "json",
            "sourcelang":      "English",
        }
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code != 200:
                continue

            data = resp.json()
            articles = data.get("articles", [])

            for art in articles:
                raw_date = art.get("seendate", "")
                if not raw_date:
                    continue
                try:
                    pub_date = datetime.strptime(
                        raw_date[:8], "%Y%m%d"
                    ).date()
                except ValueError:
                    continue

                rows.append({
                    "ticker":   ticker,
                    "date":     pub_date,
                    "title":    art.get("title", ""),
                    "summary":  "",
                    "source":   "gdelt",
                    "av_score": None,
                })

            time.sleep(1)   # be polite to GDELT

        except Exception as e:
            print(f"  [GDELT] {ticker} year={year}: {e}")
            time.sleep(2)
            continue

    df = pd.DataFrame(rows)
    print(f"  [GDELT] {ticker}: {len(df)} articles collected")
    return df


# ─────────────────────────────────────────────────────────────
# Source 3 — yfinance (recent news, fallback)
# ─────────────────────────────────────────────────────────────
def fetch_yfinance(ticker: str) -> pd.DataFrame:
    """
    Fetches recent news from yfinance as a fallback source.
    Only covers last few days but always available.
    """
    print(f"  [yfinance] Fetching news for {ticker}...")
    rows = []
    try:
        t    = yf.Ticker(ticker)
        news = t.news or []

        for item in news:
            content  = item.get("content", {})
            title    = content.get("title", "")
            summary  = content.get("summary", "")
            pub_date = content.get("pubDate", "")
            if not pub_date or not title:
                continue
            try:
                d = pd.to_datetime(pub_date).date()
            except Exception:
                continue
            rows.append({
                "ticker":   ticker,
                "date":     d,
                "title":    title,
                "summary":  summary,
                "source":   "yfinance",
                "av_score": None,
            })

    except Exception as e:
        print(f"  [yfinance] Error: {e}")

    df = pd.DataFrame(rows)
    print(f"  [yfinance] {ticker}: {len(df)} articles collected")
    return df


# ─────────────────────────────────────────────────────────────
# Merge and deduplicate all sources
# ─────────────────────────────────────────────────────────────
def merge_sources(*dfs) -> pd.DataFrame:
    """Concatenates multiple DataFrames and removes duplicates."""
    valid = [df for df in dfs if not df.empty]
    if not valid:
        return pd.DataFrame()

    combined = pd.concat(valid, ignore_index=True)

    # Deduplicate on ticker + date + title
    before = len(combined)
    combined = combined.drop_duplicates(
        subset=["ticker", "date", "title"]
    )
    after = len(combined)
    print(f"  Deduplication: {before} → {after} articles "
          f"({before - after} removed)")

    combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)
    return combined


# ─────────────────────────────────────────────────────────────
# Auto-label using next-day price movement
# ─────────────────────────────────────────────────────────────
def label_with_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    Labels each article based on next-day price change.
    positive : next-day return > +0.5%
    negative : next-day return < -0.5%
    neutral  : between -0.5% and +0.5%
    """
    if df.empty:
        return df

    print("\nLabeling articles with price movement...")

    # Download full price history for all tickers at once
    price_cache = {}
    for ticker in TICKERS:
        try:
            pdf = yf.download(
                ticker,
                start=START_DATE,
                end=END_DATE,
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
            if isinstance(pdf.columns, pd.MultiIndex):
                pdf.columns = [c[0] for c in pdf.columns]
            pdf.index = pd.to_datetime(pdf.index).date
            pdf["next_return"] = pdf["Close"].pct_change().shift(-1) * 100
            price_cache[ticker] = pdf
            print(f"  Price data loaded: {ticker} "
                  f"({len(pdf)} rows)")
        except Exception as e:
            print(f"  Price data failed for {ticker}: {e}")

    labels        = []
    price_changes = []

    for _, row in df.iterrows():
        ticker    = row["ticker"]
        news_date = row["date"]
        pdf       = price_cache.get(ticker)

        if pdf is not None and news_date in pdf.index:
            ret = pdf.loc[news_date, "next_return"]
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
                continue
            except Exception:
                pass

        labels.append("neutral")
        price_changes.append(None)

    df = df.copy()
    df["label"]        = labels
    df["price_change"] = price_changes
    return df


# ─────────────────────────────────────────────────────────────
# Prepare text field
# ─────────────────────────────────────────────────────────────
def prepare_text(df: pd.DataFrame) -> pd.DataFrame:
    """Combines title and summary into single text field."""
    df = df.copy()
    df["text"] = df.apply(
        lambda r: r["title"] if not r.get("summary", "")
                  else f"{r['title']}. {r['summary']}",
        axis=1,
    )
    df = df[df["text"].str.strip().str.len() > 10].copy()
    return df


# ─────────────────────────────────────────────────────────────
# Coverage report
# ─────────────────────────────────────────────────────────────
def print_coverage_report(df: pd.DataFrame):
    """Shows how many trading days have news coverage."""
    print("\n" + "="*55)
    print("  SENTIMENT COVERAGE REPORT")
    print("="*55)

    # Total trading days (approx)
    total_trading_days = len(pd.bdate_range(START_DATE, END_DATE))

    for ticker in TICKERS:
        tdf = df[df["ticker"] == ticker]
        days_with_news = tdf["date"].nunique()
        articles       = len(tdf)
        coverage_pct   = days_with_news / total_trading_days * 100

        print(f"  {ticker}:")
        print(f"    Articles       : {articles:,}")
        print(f"    Days with news : {days_with_news:,}")
        print(f"    Coverage       : {coverage_pct:.1f}% of trading days")

        if tdf.empty:
            continue
        print(f"    Label dist     : "
              f"pos={sum(tdf['label']=='positive')} "
              f"neu={sum(tdf['label']=='neutral')} "
              f"neg={sum(tdf['label']=='negative')}")

    print("="*55)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def collect_and_label():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_collected = []

    # Load existing data
    if os.path.exists(RAW_FILE):
        existing = pd.read_csv(RAW_FILE)
        existing["date"] = pd.to_datetime(existing["date"]).dt.date
        print(f"Loaded {len(existing)} existing news items\n")
        all_collected.append(existing)

    for i, ticker in enumerate(TICKERS):
        print(f"\n{'='*55}")
        print(f"  [{i+1}/{len(TICKERS)}] Collecting news for {ticker}")
        print(f"{'='*55}")

        # Source 1: Alpha Vantage (best historical coverage)
        av_df = fetch_alpha_vantage(ticker)

        # Source 2: GDELT (10 years, completely free)
        gdelt_df = fetch_gdelt(ticker)

        # Source 3: yfinance (recent fallback)
        yf_df = fetch_yfinance(ticker)

        # Merge all sources for this ticker
        ticker_df = merge_sources(av_df, gdelt_df, yf_df)

        if not ticker_df.empty:
            all_collected.append(ticker_df)

        print(f"  Total for {ticker}: {len(ticker_df)} articles")

    if not all_collected:
        print("No news collected from any source.")
        return pd.DataFrame()

    # Merge all tickers
    print("\nMerging all sources...")
    combined = merge_sources(*all_collected)

    # Label with price movement
    combined = label_with_price(combined)

    # Prepare text field
    combined = prepare_text(combined)

    # Save
    combined.to_csv(RAW_FILE, index=False, encoding="utf-8-sig")
    print(f"\nSaved {len(combined)} articles → {RAW_FILE}")

    # Coverage report
    print_coverage_report(combined)

    return combined


if __name__ == "__main__":
    collect_and_label()