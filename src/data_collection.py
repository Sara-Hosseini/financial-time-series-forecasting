import os
from pathlib import Path
import requests
import pandas as pd
from dotenv import load_dotenv
def fetch_daily_adjusted(symbol: str, api_key: str, outputsize:str = "compact" )  -> pd.DataFrame:
    url="https://www.alphavantage.co/query"
    params={"function":"TIME_SERIES_DAILY",
             "symbol": symbol, "outputsize": outputsize, "apikey":api_key}
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
def main():
    load_dotenv()
    api_key=os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise RuntimeError("API key not found. Check your .env file.") 
    symbol= "AAPL"
    df= fetch_daily_adjusted(symbol, api_key)
    output_dir=Path("data")/"raw"
    output_dir.mkdir(parents= True, exist_ok= True)
    output_Path=output_dir/ f"{symbol}_daily_compact.csv"
    df.to_csv(output_Path)
    print("Saved:", output_Path)
    print("Shape:", df.shape)
    print(df.head())
if __name__=="__main__":
    main()
