import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def load_prices(csv_path: str) -> pd.DataFrame:

    df=pd.read_csv(csv_path)
    df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    df["Date"]=pd.to_datetime(df["Date"],format="%Y-%m-%d" ,errors="coerce")
    df=df.dropna(subset=["Date"]).sort_values("Date")
    df=df.drop_duplicates(subset=["Date"], keep= "last")

    if "Close" not in df.columns:
        raise KeyError ("Column 'Close' not found in CSV.")
    df["Close"]=pd.to_numeric(df["Close"], errors= "coerce")
    df=df.dropna(subset=["Close"])
    df=df.set_index("Date")
    return df[["Close"]]


def create_sequences(series_2d: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    X=[]
    y=[]
    for i in range(window_size, len(series_2d)):
        X.append(series_2d[i-window_size:i])
        y.append(series_2d[i])
    return np.array(X), np.array(y)


def prepare_lstm_data(close_2d: np.ndarray, window_size: int, train_ratio: float):
    split_idx=int(len(close_2d)*train_ratio)
    scaler=MinMaxScaler()
    train_scaled=scaler.fit_transform(close_2d[:split_idx])
    test_scaled=scaler.transform(close_2d[split_idx:])
    all_scaled=np.vstack((train_scaled,test_scaled))
    X,y= create_sequences(all_scaled, window_size)
    seq_split=split_idx- window_size
    X_train , y_train= X[:seq_split], y[:seq_split]
    X_test, y_test= X[seq_split:], y[seq_split:]

    return X_train, y_train, X_test, y_test, scaler, split_idx, seq_split

if __name__=="__main__":
    path="data/raw/AAPL_daily_10y_yahoo.csv"
    df=load_prices(path)
    close_2d=df["Close"].values.reshape(-1,1)
    
    window_size=20
    train_ratio=0.8
    X_train, y_train, X_test, y_test, scaler, split_idx, seq_split=prepare_lstm_data(
        close_2d, window_size, train_ratio
    )

       
    print("X_tarin shape:", X_train.shape, "y_train:", y_train.shape)
    print("X_test:", X_test.shape, "y_test:", y_test.shape)
    print("Train last date:", df.index[split_idx -1])
    print("Test firt date:", df.index[split_idx])

