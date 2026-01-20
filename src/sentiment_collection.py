import os
import time
import pandas as pd
import requests
from dotenv import load_dotenv


def fetch_news_sentiment(
    api_key: str,
    tickers: str = "AAPL",
    limit: int = 50,
    *,
    session: requests.Session | None = None,
    max_retries: int = 5,
    base_sleep: float = 1.2
) -> dict:
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": tickers,
        "limit": limit,
        "apikey": api_key
    }

    sess = session or requests.Session()

    for attempt in range(max_retries + 1):
        r = sess.get(url, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()

     
        if "Error Message" in data:
            raise ValueError(f"Alpha Vantage error: {data['Error Message']}")

        #  Rate limit
        if "Note" in data or "Information" in data:
            if attempt == max_retries:
                msg = data.get("Note") or data.get("Information")
                raise RuntimeError(f"Alpha Vantage rate limit: {msg}")

            sleep_s = base_sleep * (2 ** attempt)
            time.sleep(sleep_s)
            continue

     
        return data

    raise RuntimeError("Unexpected retry failure")


def news_json_to_df(data: dict, target_ticker: str = "AAPL") -> pd.DataFrame:
    rows = []
    feed = data.get("feed", [])

    for item in feed:
        t = item.get("time_published")
        title = item.get("title")
        source = item.get("source")

        ticker_sentiments = item.get("ticker_sentiment", [])
        rec_for_ticker = None

        for rec in ticker_sentiments:
            if rec.get("ticker") == target_ticker:
                rec_for_ticker = rec
                break

        if rec_for_ticker is None:
            continue

        sentiment_score = rec_for_ticker.get("ticker_sentiment_score")
        relevance_score = rec_for_ticker.get("relevance_score")

        rows.append({
            "time_published": t,
            "title": title,
            "source": source,
            "sentiment_score": float(sentiment_score) if sentiment_score is not None else None,
            "relevance_score": float(relevance_score) if relevance_score is not None else None
        })

    df = pd.DataFrame(rows)
    df["time_published"] = pd.to_datetime(df["time_published"], format="%Y%m%dT%H%M%S", errors="coerce")
    df = df.dropna(subset=["time_published", "sentiment_score", "relevance_score"])
    df = df.sort_values("time_published").reset_index(drop=True)
    return df


def main():
    load_dotenv()
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise RuntimeError("ALPHAVANTAGE_API_KEY is not found in .env")

    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    summary_rows = []

    with requests.Session() as sess:
        for i, t in enumerate(tickers):
            print("\n" + "=" * 60)
            print(f"Processing ticker: {t}")

       
            if i > 0:
                time.sleep(1.2)

            data = fetch_news_sentiment(api_key=api_key, tickers=t, limit=50, session=sess)
            df = news_json_to_df(data, target_ticker=t)

            n_news = len(df)
            mean_sent = df["sentiment_score"].mean() if n_news > 0 else 0.0
            mean_rel = df["relevance_score"].mean() if n_news > 0 else 0.0

            summary_rows.append({
                "ticker": t,
                "news_count": n_news,
                "sentiment_mean": mean_sent,
                "relevance_mean": mean_rel
            })

            if n_news > 0:
                print(df[["time_published", "title", "sentiment_score", "relevance_score"]].head(3))
            else:
                print(f"No {t}-related sentiment records found for this ticker.")

    summary_df = pd.DataFrame(summary_rows).sort_values("news_count", ascending=False)
    print("\n" + "=" * 60)
    print("Multi-stock sentiment summary:")
    print(summary_df)


if __name__ == "__main__":
    main()
