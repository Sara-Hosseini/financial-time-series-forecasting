import os
import time
import pandas as pd
import requests
from dotenv import load_dotenv

BASE_URL = "https://www.alphavantage.co/query"

def fetch_news_sentiment(api_key: str, tickers: str, limit: int = 50, session=None,
                         max_retries: int = 5, base_sleep: float = 2.0) -> dict:
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": tickers,
        "limit": limit,
        "apikey": api_key,
    }

    sess = session or requests.Session()

    for attempt in range(max_retries + 1):
        r = sess.get(BASE_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()

        if "Error Message" in data:
            raise ValueError(f"Alpha Vantage error: {data['Error Message']}")

        # rate limit / info
        if "Note" in data or "Information" in data:
            if attempt == max_retries:
                msg = data.get("Note") or data.get("Information")
                raise RuntimeError(f"Alpha Vantage rate limit/info: {msg}")

            sleep_s = base_sleep * (2 ** attempt)
            print(f"[rate-limit] sleeping {sleep_s:.1f}s then retry...")
            time.sleep(sleep_s)
            continue

        return data

    raise RuntimeError("Unexpected retry failure")


def news_json_to_df(data: dict, target_ticker: str):
    feed = data.get("feed") or []
    target = target_ticker.upper()
    rows = []

    for item in feed:
        t = item.get("time_published")
        title = item.get("title")
        source = item.get("source")

        rec_for_ticker = None
        for rec in (item.get("ticker_sentiment") or []):
            if (rec.get("ticker") or "").upper() == target:
                rec_for_ticker = rec
                break

        if rec_for_ticker:
            score = rec_for_ticker.get("ticker_sentiment_score")
            rel = rec_for_ticker.get("relevance_score")
            mode = "ticker"
        else:
            score = item.get("overall_sentiment_score")
            rel = None
            mode = "overall"

        try:
            score = float(score)
        except Exception:
            continue

        try:
            rel = float(rel) if rel is not None else None
        except Exception:
            rel = None

        rows.append({
            "time_published": t,
            "title": title,
            "source": source,
            "sentiment_score": score,
            "relevance_score": rel,
            "mode": mode,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        daily = pd.DataFrame(columns=["date", "sentiment_mean", "sentiment_median", "n_news"])
        return df, daily

    df["time_published"] = pd.to_datetime(df["time_published"], format="%Y%m%dT%H%M%S", errors="coerce")
    df = df.dropna(subset=["time_published", "sentiment_score"]).sort_values("time_published")

    df["date"] = df["time_published"].dt.date

    daily = (df.groupby("date")
               .agg(sentiment_mean=("sentiment_score", "mean"),
                    sentiment_median=("sentiment_score", "median"),
                    n_news=("sentiment_score", "size"))
               .reset_index())

    return df, daily


def main():
    load_dotenv()
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise RuntimeError("ALPHAVANTAGE_API_KEY is not found in .env")

    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    summary_rows = []

    out_dir = r"D:\lessons\season 4\code\stock-transformer-project\data\processed"
    os.makedirs(out_dir, exist_ok=True)

    with requests.Session() as sess:
        for i, t in enumerate(tickers):
            print("\n" + "=" * 60)
            print(f"Processing ticker: {t}")

            # paths
            news_raw_path = os.path.join(out_dir, f"{t}_news_raw.csv")
            news_daily_path = os.path.join(out_dir, f"{t}_news_daily.csv")

            # ✅ CACHE: اگر فایل‌ها موجود باشند، API نزن
            if os.path.exists(news_raw_path) and os.path.exists(news_daily_path):
                print(f"[CACHE] Using existing files for {t}")
                df = pd.read_csv(news_raw_path)
                daily = pd.read_csv(news_daily_path)

            else:
                # pace requests فقط وقتی واقعاً API می‌زنیم
                if i > 0:
                    print("Sleeping 13s to avoid rate-limit ...")
                    time.sleep(13)

                try:
                    data = fetch_news_sentiment(api_key=api_key, tickers=t, limit=50, session=sess)
                except RuntimeError as e:
                    # quota/day limit
                    print(f"[STOP] {t}: {e}")
                    break

                df, daily = news_json_to_df(data, target_ticker=t)

                # save cached files
                df.to_csv(news_raw_path, index=False, encoding="utf-8-sig")
                daily.to_csv(news_daily_path, index=False, encoding="utf-8-sig")

                print("Saved raw news:", news_raw_path)
                print("Saved daily news:", news_daily_path)

            # summary stats
            n_news = len(df)
            mean_sent = df["sentiment_score"].mean() if n_news > 0 else float("nan")
            mean_rel = df["relevance_score"].dropna().mean() if n_news > 0 else float("nan")

            summary_rows.append({
                "ticker": t,
                "news_count": n_news,
                "sentiment_mean": mean_sent,
                "relevance_mean": mean_rel
            })

            # preview
            cols = ["time_published", "title", "sentiment_score", "relevance_score", "mode"]
            print(df[cols].head(3))
            print("Daily head:")
            print(daily.head(3))

    summary_df = pd.DataFrame(summary_rows).sort_values("news_count", ascending=False)
    print("\nMulti-stock sentiment summary:")
    print(summary_df)
    summary_path = os.path.join(out_dir, "summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print("Saved summary:", summary_path)

