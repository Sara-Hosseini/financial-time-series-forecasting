# ============================================================
# LSTM Price Forecasting (Leakage-SAFE, Final Clean Version)
# Target: y_t = log(C_{t+H} / C_t)
# Input: price-derived features ONLY (NO Close in X)
# Outputs: metrics JSON + plots PNG + predictions CSV
# ============================================================

import os
import json
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Any, List

# --- silence TF noise BEFORE importing tensorflow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"         # hide INFO/WARN
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"        # hide oneDNN message (and make runs more consistent)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
@dataclass
class RunConfig:
    ticker: str = "AAPL"
    csv_path: str = "data/raw/AAPL_daily_10y_yahoo.csv"
    date_format: str = "%Y-%m-%d"

    horizons: Tuple[int, ...] = (5, 10)

    train_ratio: float = 0.70
    val_ratio: float = 0.10

    # (اگر خواستی بعداً CV اضافه می‌کنیم. فعلاً ساده و پایدار)
    window_candidates: Tuple[int, ...] = (30, 45)
    window_default: int = 30

    units1: int = 96
    units2: int = 32
    dropout: float = 0.10
    lr: float = 5e-4

    epochs: int = 120
    batch_size: int = 64
    patience: int = 8

    out_dir_models: str = "outputs/models"
    out_dir_scalers: str = "outputs/scalers"
    out_dir_metrics: str = "outputs/metrics"
    out_dir_plots: str = "data/processed"
    out_dir_preds: str = "outputs/predictions"

    save_model: bool = True
    save_scalers: bool = True


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def ensure_dirs(*paths: str):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def save_json(obj: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    return path


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------
def rmse(a, b) -> float:
    return float(np.sqrt(mean_squared_error(a, b)))


def mae(a, b) -> float:
    return float(mean_absolute_error(a, b))


def mape(a, b) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.mean(np.abs((b - a) / (a + 1e-12))) * 100.0)


def directional_accuracy(c_t, c_true, c_pred) -> float:
    return float(np.mean(np.sign(c_true - c_t) == np.sign(c_pred - c_t)) * 100.0)


# ------------------------------------------------------------
# Robust Yahoo Loader (your exact CSV format)
# ------------------------------------------------------------
def load_prices(csv_path: str, date_format: str = "%Y-%m-%d") -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Drop Yahoo meta rows: row0=TICKER, row1=Date labels
    if len(df) >= 2 and str(df.iloc[0, 0]).strip().lower() == "ticker":
        df = df.iloc[2:].copy()

    # In your file, first column "Price" is actually Date
    if "Price" in df.columns and "Date" not in df.columns:
        df = df.rename(columns={"Price": "Date"})

    if "Date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Date"})

    df["Date"] = pd.to_datetime(df["Date"], format=date_format, errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    df = df.set_index("Date")

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Close"])
    keep_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    return df[keep_cols].dropna()


# ------------------------------------------------------------
# Features (NO Close in X)
# ------------------------------------------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["log_ret_1"] = np.log(out["Close"] / out["Close"].shift(1))
    out["ret_5"] = np.log(out["Close"] / out["Close"].shift(5))
    out["ret_10"] = np.log(out["Close"] / out["Close"].shift(10))

    out["ma_5"] = out["Close"].rolling(5).mean()
    out["ma_10"] = out["Close"].rolling(10).mean()

    out["hl_range"] = (out["High"] - out["Low"]) / (out["Close"] + 1e-12)
    out["vol_10"] = out["log_ret_1"].rolling(10).std()
    out["vol_20"] = out["log_ret_1"].rolling(20).std()

    return out.dropna()


# ------------------------------------------------------------
# Target + sequences (bridge-correct, leakage-safe)
# ------------------------------------------------------------
def make_target(df: pd.DataFrame, H: int) -> pd.Series:
    # y_t = log(C_{t+H} / C_t)
    return np.log(df["Close"].shift(-H) / (df["Close"] + 1e-12))


def make_sequences(X: np.ndarray, y_1d: np.ndarray, start: int, end: int, window: int):
    Xs, ys = [], []
    for t in range(start, end):
        if t - window < 0:
            continue
        Xs.append(X[t - window: t, :])
        ys.append(y_1d[t])
    return np.asarray(Xs), np.asarray(ys).reshape(-1, 1)


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------
def build_model(window: int, n_features: int, cfg: RunConfig) -> tf.keras.Model:
    model = Sequential([
        Input(shape=(window, n_features)),
        LSTM(cfg.units1, return_sequences=True),
        Dropout(cfg.dropout),
        LSTM(cfg.units2),
        Dropout(cfg.dropout),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.lr, clipnorm=1.0),
        loss=tf.keras.losses.Huber(delta=1.0),
    )
    return model


# ------------------------------------------------------------
# Plot (IMPORTANT: plot on TARGET dates t+H)
# ------------------------------------------------------------
def save_plot_target_dates(ticker: str, dates_t, close_true, close_pred, H, save_path):
    dates_target = dates_t + pd.tseries.offsets.BDay(H)

    plt.figure(figsize=(12, 5))
    plt.plot(dates_target, close_true, label=f"Actual Close (t+{H})")
    plt.plot(dates_target, close_pred, label=f"Pred Close (t+{H})")
    plt.title(f"{ticker} TEST Prediction on Target Dates (H={H})")
    plt.xlabel("Target Date (t+H)")
    plt.ylabel("Close Price")
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    cfg = RunConfig()
    ensure_dirs(cfg.out_dir_models, cfg.out_dir_scalers, cfg.out_dir_metrics, cfg.out_dir_plots, cfg.out_dir_preds)

    df = load_prices(cfg.csv_path, cfg.date_format)
    df_feat = add_features(df)

    FEATURE_COLS = [
        "log_ret_1", "ret_5", "ret_10",
        "ma_5", "ma_10",
        "hl_range", "vol_10", "vol_20",
    ]

    for H in cfg.horizons:
        print(f"\n==================== {cfg.ticker} | Horizon={H} ====================")

        y = make_target(df_feat, H).dropna()                 # drop last H rows
        X = df_feat.loc[y.index, FEATURE_COLS].values        # align features to y index
        dates_t = y.index                                    # these are t dates (input time)

        n = len(y)
        train_end = int(n * cfg.train_ratio)
        val_end = int(n * (cfg.train_ratio + cfg.val_ratio))

        # choose window (you can later loop over candidates)
        window = cfg.window_default

        # Fit scalers ONLY on TRAIN (no leakage)
        scaler_X = RobustScaler().fit(X[:train_end])
        scaler_y = StandardScaler().fit(y.values[:train_end].reshape(-1, 1))

        Xs = scaler_X.transform(X)
        ys_1d = scaler_y.transform(y.values.reshape(-1, 1)).flatten()

        # Sequences (bridge-correct)
        X_tr, y_tr = make_sequences(Xs, ys_1d, start=window, end=train_end, window=window)
        X_va, y_va = make_sequences(Xs, ys_1d, start=train_end, end=val_end, window=window)
        X_te, y_te = make_sequences(Xs, ys_1d, start=val_end, end=n, window=window)

        # Align close arrays to EXACT same t-range as X_te
        # X_te corresponds to targets t in [val_end, n)
        close_t = df_feat.loc[dates_t[val_end:n], "Close"].to_numpy()
        close_true = df_feat["Close"].shift(-H).loc[dates_t[val_end:n]].to_numpy()
        dates_test_t = dates_t[val_end:n]

        # sanity check: must match
        if len(X_te) != len(close_t) or len(close_t) != len(close_true):
            raise RuntimeError(f"Alignment mismatch: X_te={len(X_te)}, close_t={len(close_t)}, close_true={len(close_true)}")

        model = build_model(window, X_tr.shape[-1], cfg)

        cb = ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5, min_lr=1e-6, verbose=0)

        model.fit(
            X_tr, y_tr,
            validation_data=(X_va, y_va),
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            callbacks=[cb],
            verbose=0
        )

        # Predict y, convert to close_{t+H}
        y_pred = scaler_y.inverse_transform(model.predict(X_te, verbose=0)).flatten()
        close_pred = close_t * np.exp(y_pred)

        # Metrics
        out_rmse = rmse(close_true, close_pred)
        out_mae = mae(close_true, close_pred)
        out_mape = mape(close_true, close_pred)
        out_dir = directional_accuracy(close_t, close_true, close_pred)

        print(f"TEST RMSE/MAE: {out_rmse:.4f} / {out_mae:.4f}")
        print(f"TEST MAPE/DirAcc: {out_mape:.2f}% / {out_dir:.2f}%")

        # Save artifacts
        tag = f"{cfg.ticker.lower()}_H{H}_win{window}_nocloseX"
        plot_path = os.path.join(cfg.out_dir_plots, f"{tag}_actual_vs_pred.png")
        preds_path = os.path.join(cfg.out_dir_preds, f"{tag}_predictions.csv")
        metrics_path = os.path.join(cfg.out_dir_metrics, f"{tag}_metrics.json")

        save_plot_target_dates(cfg.ticker, dates_test_t, close_true, close_pred, H, plot_path)

        pd.DataFrame({
            "date_t": dates_test_t,
            "date_target_t_plus_H": dates_test_t + pd.tseries.offsets.BDay(H),
            "close_t": close_t,
            "close_true_t_plus_H": close_true,
            "close_pred_t_plus_H": close_pred,
            "abs_error": np.abs(close_true - close_pred),
            "pct_error": (close_pred - close_true) / (close_true + 1e-12) * 100.0,
        }).to_csv(preds_path, index=False)

        metrics = {
            "ticker": cfg.ticker,
            "horizon_days": H,
            "window": window,
            "feature_cols": FEATURE_COLS,
            "test": {
                "rmse": out_rmse,
                "mae": out_mae,
                "mape": out_mape,
                "dir_acc_pct": out_dir,
            },
            "run_config": asdict(cfg),
            "artifacts": {
                "plot": plot_path,
                "preds_csv": preds_path,
            }
        }
        save_json(metrics, metrics_path)

        if cfg.save_model:
            model_path = os.path.join(cfg.out_dir_models, f"{tag}.keras")
            model.save(model_path)
            metrics["artifacts"]["model"] = model_path
            save_json(metrics, metrics_path)

        if cfg.save_scalers:
            sx_path = os.path.join(cfg.out_dir_scalers, f"{tag}_scaler_X.joblib")
            sy_path = os.path.join(cfg.out_dir_scalers, f"{tag}_scaler_y.joblib")
            joblib.dump(scaler_X, sx_path)
            joblib.dump(scaler_y, sy_path)
            metrics["artifacts"]["scaler_X"] = sx_path
            metrics["artifacts"]["scaler_y"] = sy_path
            save_json(metrics, metrics_path)

        tf.keras.backend.clear_session()
