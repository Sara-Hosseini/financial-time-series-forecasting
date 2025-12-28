import os
from pathlib import Path
import requests
import pandas as pd
from dotenv import load_dotenv
import yfinance as yf

def fetch_daily(symbol: str, api_key: str )  -> pd.DataFrame:
    url="https://www.alphavantage.co/query"
    params={"function":"TIME_SERIES_DAILY",
             "symbol": symbol, "outputsize": "compact", "apikey":api_key}
    response=requests.get(url,params=params, timeout=60)
    response.raise_for_status()
    data=response.json()

    if "Error Message" in data:
        raise ValueError(f"Alpha Vantage error:{data['Error Message']}")
    if "Note" in data:
        raise RuntimeError(f"Alpha Vantage rate limit:{data['Note']}")
    if "Information" in data:
        raise RuntimeError(f"Alpha Vantage information: {data['Information']}")
    if "Time Series (Daily)" not in data:
        raise KeyError(f"Unexpected response keys: {list(data.keys())}")
    

    df=pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
    df.index=pd.to_datetime(df.index)
    df=df.sort_index()
    df=df.rename(columns={
        "1. open": "Open",
        "2. high": "High",
        "3. low": "Low" ,   
        "4. close": "Close",
        "5. volume": "Volume",
    })
    for col in df.columns:
        df[col]=pd.to_numeric(df[col], errors="coerce")
    return df
def fetch_yahoo_daily(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    Download daily OHLCV from Yahoo Finance via yfinance.
    Returns a DataFrame with columns: Open, High, Low, Close, Volume
    """
    df = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False)

    if df is None or df.empty:
        raise RuntimeError(f"Yahoo returned empty data for {symbol} ({start} -> {end}).")

    # Ensure standard column names (yfinance already uses these)
    df = df.rename(columns={
        "Open": "Open",
        "High": "High",
        "Low": "Low",
        "Close": "Close",
        "Volume": "Volume",
    })

    # Keep only what we need
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

    # Index should be datetime and sorted
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    return df

def main():
    load_dotenv()
    api_key=os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise RuntimeError("API key not found. Check your .env file.") 
    symbol= "AAPL"
    df= fetch_daily(symbol, api_key)
    output_dir=Path("data")/"raw"
    output_dir.mkdir(parents= True, exist_ok= True)
    output_Path=output_dir/ f"{symbol}_daily_compact.csv"
    df.to_csv(output_Path)
     # ---- Yahoo 10-year dataset (for thesis) ----
    yahoo_symbol = "AAPL"
    start_date = "2015-01-01"
    end_date = "2026-01-01"  # end is exclusive

    df_yahoo = fetch_yahoo_daily(yahoo_symbol, start=start_date, end=end_date)

    yahoo_path = output_dir / f"{yahoo_symbol}_daily_10y_yahoo.csv"
    df_yahoo.to_csv(yahoo_path, index=True)

    print("Saved (Yahoo 10y):", yahoo_path)
    print("Rows (Yahoo 10y):", len(df_yahoo))
    print("Date range (Yahoo 10y):", df_yahoo.index.min().date(), "->", df_yahoo.index.max().date())
    print("Saved:", output_Path)
    print("Shape:", df.shape)
    print("Rows:", len(df))
    print("Date range:", df.index.min().date(), "->", df.index.max().date())
    print(df.head())
    print(df.tail())
if __name__=="__main__":
    main()
