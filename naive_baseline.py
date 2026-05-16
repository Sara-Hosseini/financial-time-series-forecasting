import pandas as pd
import numpy as np

tickers = ['AAPL', 'MSFT', 'NVDA']
horizons = [1, 5, 10]

for ticker in tickers:
    df = pd.read_csv(f'data/raw/{ticker}_daily_10y_yahoo.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # Test years only (2021-2025)
    test_df = df[df['Date'].dt.year.isin([2021, 2022, 2023, 2024, 2025])].reset_index(drop=True)

    for h in horizons:
        actual = test_df['Close'].values[h:]
        predicted = test_df['Close'].values[:-h]  # naive: predict p(t+h) = p(t)
        mape = np.mean(np.abs(actual - predicted) / actual) * 100
        print(f"{ticker} H={h}d Naive MAPE: {mape:.2f}%")
