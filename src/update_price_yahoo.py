import yfinance as yf
from pathlib import Path

def update_price_data(ticker: str):
    df = yf.download(
        ticker,
        period="max",
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    df = df.reset_index()

    out = Path("data/raw")
    out.mkdir(parents=True, exist_ok=True)

    out_file = out / f"{ticker}_daily_yahoo_latest.csv"
    df.to_csv(out_file, index=False)

    print(f"Saved: {out_file}")
    print(df.tail())


if __name__ == "__main__":
    update_price_data("AAPL")
