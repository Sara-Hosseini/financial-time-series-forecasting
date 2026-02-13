import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler 
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error

import matplotlib.pyplot as plt 

np.random.seed(42)
tf.random.set_seed(42)



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



def build_lstm_model(look_back: int, n_fearures: int):
    model=Sequential([Input(shape=(look_back, n_fearures)),
                      LSTM(50), Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    return model


def evaluate_model(model, X_test,y_test,scaler):
    """Evaluate trained model on test data.
    returns:
    test_mse_scaled, rmse_usd, mae_usd
    """
    y_pred_scale=model.predict(X_test,verbose=0)
    y_test_real=scaler.inverse_transform(y_test)
    y_pred_real=scaler.inverse_transform(y_pred_scale)
    test_mse_scaled=mean_squared_error(y_test,y_pred_scale)
    rmse_usd=np.sqrt(mean_squared_error(y_test_real,y_pred_real))
    mae_usd=mean_absolute_error(y_test_real,y_pred_real)
    return test_mse_scaled,rmse_usd,mae_usd


def plot_actual_vs_predicted(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler: MinMaxScaler,
    dates_test: pd.DatetimeIndex,
    save_path: str = "data/processed/aapl_actual_vs_predicted.png",
):
    """
    Plot actual vs predicted on test set (in real USD) and save as PNG.
    """

    
    y_pred_scaled = model.predict(X_test, verbose=0)

    
    y_test_real = scaler.inverse_transform(y_test).flatten()
    y_pred_real = scaler.inverse_transform(y_pred_scaled).flatten()

    
    plt.figure(figsize=(12, 5))
    plt.plot(dates_test, y_test_real, label="Actual")
    plt.plot(dates_test, y_pred_real, label="Predicted")
    plt.title("Actual vs Predicted (Test Set)")
    plt.xlabel("Date")
    plt.ylabel("Close Price (USD)")
    plt.legend()
    plt.tight_layout()

    # 4) ذخیره
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()

    return save_path


def plot_training_history(
    history,
    save_path: str = "data/processed/aapl_train_val_loss.png",
):
    """
    Plot training loss and validation loss across epochs and save as PNG.
    """

    
    loss = history.history["loss"]
    val_loss = history.history.get("val_loss", None)

    
    epochs = range(1, len(loss) + 1)

    
    plt.figure(figsize=(10, 4))
    plt.plot(epochs, loss, label="Train Loss")
    if val_loss is not None:
        plt.plot(epochs, val_loss, label="Val Loss")

    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.tight_layout()

    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()

    return save_path




if __name__=="__main__":
    path="data/raw/AAPL_daily_10y_yahoo.csv"
    df=load_prices(path)
    close_2d=df["Close"].values.reshape(-1,1)
    
    window_size=20
    train_ratio=0.8
    X_train, y_train, X_test, y_test, scaler, split_idx, seq_split=prepare_lstm_data(
        close_2d, window_size, train_ratio
    )
    model=build_lstm_model(look_back=window_size, n_fearures=1)
    model.summary()
    es=EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    history=model.fit(X_train, y_train,
                      validation_split=0.2, epochs=30,
                      batch_size=32,
                      callbacks=[es],verbose=1)
    test_mse=model.evaluate(X_test,y_test, verbose=0)
    test_mse_scaled, rmse_usd, mae_usd = evaluate_model(model, X_test, y_test, scaler)
    print("Test MSE (scaled):", test_mse_scaled)
    print("Test RMSE (USD):", rmse_usd)
    print("Test MAE (USD):", mae_usd)
    print("Test MSE:", test_mse)
    dates_test = df.index[split_idx:]
    plot_path = plot_actual_vs_predicted(model, X_test, y_test, scaler, dates_test)
    print("Saved plot to:", plot_path)

    loss_plot_path = plot_training_history(history)
    print("Saved loss plot to:", loss_plot_path)



