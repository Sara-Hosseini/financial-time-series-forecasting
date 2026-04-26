# Train final LSTM models for AAPL, MSFT, NVDA

import os
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import RobustScaler, StandardScaler

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Settings
TICKERS    = ["AAPL", "MSFT", "NVDA"]
HORIZONS   = [5, 10]
WINDOW     = 30
VAL_RATIO  = 0.10
UNITS1     = 64
UNITS2     = 32
DROPOUT    = 0.25
LR         = 5e-4
EPOCHS     = 150
BATCH_SIZE = 32

OUT_MODELS  = "outputs/models"
OUT_SCALERS = "outputs/scalers"
os.makedirs(OUT_MODELS,  exist_ok=True)
os.makedirs(OUT_SCALERS, exist_ok=True)

FEATURE_COLS = [
    "log_ret_1", "ret_5", "ret_10",
    "ma_ratio_5", "ma_ratio_10",
    "hl_range", "vol_10", "vol_20", "rsi_14",
]

def load_prices(csv_path):
    df = pd.read_csv(csv_path)
    first_cell = str(df.iloc[0, 0]).strip().lower() if len(df) >= 1 else ""
    if first_cell == "ticker":
        df = df.iloc[2:].copy()
    if "Price" in df.columns and "Date" not in df.columns:
        df = df.rename(columns={"Price": "Date"})
    if "Date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date").set_index("Date")
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["Close"])

def add_features(df):
    out = df.copy()
    out["log_ret_1"]   = np.log(out["Close"] / out["Close"].shift(1))
    out["ret_5"]       = np.log(out["Close"] / out["Close"].shift(5))
    out["ret_10"]      = np.log(out["Close"] / out["Close"].shift(10))
    out["ma_ratio_5"]  = out["Close"].rolling(5).mean()  / (out["Close"] + 1e-12) - 1
    out["ma_ratio_10"] = out["Close"].rolling(10).mean() / (out["Close"] + 1e-12) - 1
    out["hl_range"]    = (out["High"] - out["Low"]) / (out["Close"] + 1e-12)
    out["vol_10"]      = out["log_ret_1"].rolling(10).std()
    out["vol_20"]      = out["log_ret_1"].rolling(20).std()
    delta = out["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    out["rsi_14"] = 100 - (100 / (1 + gain / (loss + 1e-12)))
    return out.dropna()

def make_sequences(X, y, start, end, window):
    Xs, ys, idxs = [], [], []
    for t in range(start, end):
        if t < window:
            continue
        Xs.append(X[t - window: t, :])
        ys.append(y[t])
        idxs.append(t)
    return np.array(Xs), np.array(ys).reshape(-1, 1), np.array(idxs)

def build_model(window, n_features):
    model = Sequential([
        Input(shape=(window, n_features)),
        LSTM(UNITS1, return_sequences=True),
        Dropout(DROPOUT),
        LSTM(UNITS2),
        Dropout(DROPOUT),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR, clipnorm=1.0),
        loss=tf.keras.losses.Huber(delta=1.0),
    )
    return model

if __name__ == "__main__":
    for TICKER in TICKERS:
        CSV_PATH = f"data/raw/{TICKER}_daily_10y_yahoo.csv"
        if not os.path.exists(CSV_PATH):
            print(f"\nSkipping {TICKER} — file not found: {CSV_PATH}")
            continue

        print(f"\n{'='*55}")
        print(f"  Training ticker: {TICKER}")
        print(f"{'='*55}")
        df_raw  = load_prices(CSV_PATH)
        df_feat = add_features(df_raw)
        print(f"Dataset: {len(df_feat)} rows ({df_feat.index[0].date()} -> {df_feat.index[-1].date()})")

        for H in HORIZONS:
            print(f"\n{'='*50}")
            print(f"  Horizon = {H} days")
            print(f"{'='*50}")

            y_full  = np.log(df_feat["Close"].shift(-H) / (df_feat["Close"] + 1e-12)).dropna()
            df_work = df_feat.loc[y_full.index].copy()
            X_raw   = df_work[FEATURE_COLS].values
            y_raw   = y_full.values
            n       = len(y_raw)

            val_start = int(n * (1 - VAL_RATIO))
            print(f"  Train: {val_start} rows | Val: {n - val_start} rows")

            scaler_X = RobustScaler().fit(X_raw[:val_start])
            scaler_y = StandardScaler().fit(y_raw[:val_start].reshape(-1, 1))
            X_s = scaler_X.transform(X_raw)
            y_s = scaler_y.transform(y_raw.reshape(-1, 1)).flatten()

            X_tr, y_tr, _ = make_sequences(X_s, y_s, 0,         val_start, WINDOW)
            X_va, y_va, _ = make_sequences(X_s, y_s, val_start, n,         WINDOW)
            print(f"  Sequences — Train: {len(X_tr)} | Val: {len(X_va)}")

            tf.keras.backend.clear_session()
            model = build_model(WINDOW, X_tr.shape[-1])

            callbacks = [
                EarlyStopping(monitor="val_loss", patience=20,
                              min_delta=1e-5, restore_best_weights=True, verbose=1),
                ReduceLROnPlateau(monitor="val_loss", patience=6,
                                  factor=0.5, min_lr=1e-6, verbose=1),
            ]

            model.fit(X_tr, y_tr, validation_data=(X_va, y_va),
                      epochs=EPOCHS, batch_size=BATCH_SIZE,
                      callbacks=callbacks, verbose=1, shuffle=False)

            model_path    = os.path.join(OUT_MODELS,  f"{TICKER}_H{H}_final.keras")
            scaler_X_path = os.path.join(OUT_SCALERS, f"{TICKER}_H{H}_scaler_X.joblib")
            scaler_y_path = os.path.join(OUT_SCALERS, f"{TICKER}_H{H}_scaler_y.joblib")

            model.save(model_path)
            joblib.dump(scaler_X, scaler_X_path)
            joblib.dump(scaler_y, scaler_y_path)

            print(f"\n  Model   -> {model_path}")
            print(f"  Scaler X -> {scaler_X_path}")
            print(f"  Scaler y -> {scaler_y_path}")

    print("\nTraining complete for all tickers.")