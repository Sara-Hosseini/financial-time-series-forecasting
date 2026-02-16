# src/lstm_price_model_final_optimized.py
# LSTM-only (thesis-grade) | Price-only multi-horizon forecasting (FAST + correct bridge)
# - Proper bridge sequences for Val/Test
# - No leakage scaling (fit on train fold only)
# - Walk-forward CV (fast) inside Train+Val
# - EarlyStop/Selection on VAL_RMSE Close_{t+H}
# - tf.data pipeline (cache/prefetch) + clear_session to avoid slowdown
# - Optional mixed precision for GPU
# - Smaller but strong search space (big speed gain)

import os
import json
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Any, List

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau

import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────
# Repro + (optional) mixed precision
# ─────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# اگر GPU داری، این را True کن (روی NVIDIA خیلی سرعت می‌دهد)
USE_MIXED_PRECISION = False
if USE_MIXED_PRECISION:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy("mixed_float16")


# ─────────────────────────────────────────────────────────────
# Config (FAST defaults)
# ─────────────────────────────────────────────────────────────
@dataclass
class RunConfig:
    ticker: str = "AAPL"
    csv_path: str = "data/raw/AAPL_daily_10y_yahoo.csv"
    date_format: str = "%Y-%m-%d"

    horizons: Tuple[int, ...] = (5, 10)

    # Holdout split: Train | Val | Test (time-based)
    train_ratio: float = 0.70
    val_ratio: float = 0.10  # test is remaining 0.20

    # Walk-forward CV inside Train+Val block (faster)
    cv_folds: int = 2
    cv_min_train_size: int = 900
    cv_val_size: int = 180
    cv_step: int = 180

    # Smaller but strong search space (BIG speed gain)
    window_candidates: Tuple[int, ...] = (30, 45)
    include_close_candidates: Tuple[bool, ...] = (True,)

    units1_candidates: Tuple[int, ...] = (96,)
    units2_candidates: Tuple[int, ...] = (32,)
    dropout_candidates: Tuple[float, ...] = (0.10,)
    lr_candidates: Tuple[float, ...] = (5e-4,)

    # training
    epochs_cv: int = 25       # just to rank configs
    epochs_final: int = 150   # final training a bit longer
    batch_size: int = 64      # اگر RAM/GPU کم داری بگذار 32
    patience: int = 8

    # outputs
    out_dir_metrics: str = "outputs/metrics"
    out_dir_models: str = "outputs/models"
    out_dir_scalers: str = "outputs/scalers"
    out_dir_plots: str = "data/processed"
    out_dir_preds: str = "outputs/predictions"


# ─────────────────────────────────────────────────────────────
# Utils / Metrics
# ─────────────────────────────────────────────────────────────
def ensure_dirs(*paths: str):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def save_json(obj: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    return path


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(a, b)))


def nrmse(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.mean(a)) + 1e-12
    return float(rmse(a, b) / denom)


def mape(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.mean(np.abs((b - a) / (a + 1e-12))) * 100.0)


def directional_accuracy(close_t: np.ndarray, close_true: np.ndarray, close_pred: np.ndarray) -> float:
    true_dir = np.sign(close_true - close_t)
    pred_dir = np.sign(close_pred - close_t)
    return float(np.mean(true_dir == pred_dir) * 100.0)


# ─────────────────────────────────────────────────────────────
# 1) Load OHLCV
# ─────────────────────────────────────────────────────────────
def load_prices_ohlcv(csv_path: str, date_format: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], format=date_format, errors="coerce")

    df = (
        df.dropna(subset=["Date"])
          .sort_values("Date")
          .drop_duplicates(subset=["Date"], keep="last")
          .set_index("Date")
    )

    if "Close" not in df.columns:
        raise KeyError("Column 'Close' not found in CSV.")

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Close"])
    keep_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    return df[keep_cols]


# ─────────────────────────────────────────────────────────────
# 2) Features (price-only)
# ─────────────────────────────────────────────────────────────
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["log_ret_1"] = np.log(out["Close"] / out["Close"].shift(1))
    out["ret_5"] = np.log(out["Close"] / out["Close"].shift(5))
    out["ret_10"] = np.log(out["Close"] / out["Close"].shift(10))

    out["ma_5"] = out["Close"].rolling(5).mean()
    out["ma_10"] = out["Close"].rolling(10).mean()
    out["p_ma5"] = out["Close"] / (out["ma_5"] + 1e-12)
    out["p_ma10"] = out["Close"] / (out["ma_10"] + 1e-12)

    ema12 = out["Close"].ewm(span=12, adjust=False).mean()
    ema26 = out["Close"].ewm(span=26, adjust=False).mean()
    out["ema12"] = ema12
    out["ema26"] = ema26
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()

    out["hl_range"] = (out["High"] - out["Low"]) / (out["Close"] + 1e-12)
    out["vol_10"] = out["log_ret_1"].rolling(10).std()
    out["vol_20"] = out["log_ret_1"].rolling(20).std()

    out["rsi_14"] = rsi(out["Close"], 14)
    out["atr_14"] = atr(out, 14) / (out["Close"] + 1e-12)

    out["log_vol"] = np.log1p(out["Volume"])
    out["vol_chg"] = out["log_vol"] - out["log_vol"].shift(1)
    out["vol_z20"] = (out["log_vol"] - out["log_vol"].rolling(20).mean()) / (
        out["log_vol"].rolling(20).std() + 1e-12
    )

    return out.dropna()


# ─────────────────────────────────────────────────────────────
# 3) Target + feature matrix + bridge sequences
# ─────────────────────────────────────────────────────────────
def make_horizon_target(df_feat: pd.DataFrame, horizon: int) -> np.ndarray:
    y = np.log(df_feat["Close"].shift(-horizon) / (df_feat["Close"] + 1e-12))
    return y.to_numpy().reshape(-1, 1)


def build_feature_matrix(df_feat: pd.DataFrame, include_close_in_X: bool):
    base_cols = [
        "hl_range", "ma_5", "ma_10", "p_ma5", "p_ma10",
        "log_ret_1", "ret_5", "ret_10",
        "vol_10", "vol_20",
        "ema12", "ema26", "macd", "macd_signal",
        "rsi_14", "atr_14",
        "log_vol", "vol_chg", "vol_z20",
    ]
    feat_cols = (["Close"] + base_cols) if include_close_in_X else base_cols
    return df_feat[feat_cols].to_numpy(), feat_cols


def make_seq_with_context(X_all: np.ndarray, y_all_1d: np.ndarray, start: int, end: int, window: int):
    Xs, ys = [], []
    for t in range(start, end):
        if t - window < 0:
            continue
        Xs.append(X_all[t-window:t, :])
        ys.append(y_all_1d[t])
    return np.asarray(Xs), np.asarray(ys).reshape(-1, 1)


def split_final_holdout(df_feat: pd.DataFrame, horizon: int, train_ratio: float, val_ratio: float):
    valid_len = len(df_feat) - horizon
    n = valid_len
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return train_end, val_end, n


# ─────────────────────────────────────────────────────────────
# 4) Model
# ─────────────────────────────────────────────────────────────
def build_lstm_model(look_back: int, n_features: int, units1: int, units2: int, dropout: float, lr: float) -> tf.keras.Model:
    loss = tf.keras.losses.Huber(delta=1.0)
    opt = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)

    model = Sequential([
        Input(shape=(look_back, n_features)),
        LSTM(units1, return_sequences=True),
        Dropout(dropout),
        LSTM(units2, return_sequences=False),
        Dropout(dropout),
        Dense(32, activation="relu"),
        Dense(1, dtype="float32" if USE_MIXED_PRECISION else None),  # برای mixed precision
    ])
    model.compile(optimizer=opt, loss=loss)
    return model


def predict_close(model: tf.keras.Model, X_seq: np.ndarray, scaler_y: StandardScaler, close_t: np.ndarray) -> np.ndarray:
    y_pred_scaled = model.predict(X_seq, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
    return close_t * np.exp(y_pred)


<<<<<<< HEAD
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

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()

    return save_path
=======
def make_tf_dataset(X: np.ndarray, y: np.ndarray, batch_size: int, training: bool):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(min(len(X), 4096), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
    return ds
>>>>>>> 3b9ce98 (Add optimized LSTM pipeline + metrics and plots (AAPL H=5,10))


# ─────────────────────────────────────────────────────────────
# 5) Train with EarlyStop on VAL_RMSE_Close
# ─────────────────────────────────────────────────────────────
def train_select_by_val_rmse_close(
    model: tf.keras.Model,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    scaler_y: StandardScaler,
    close_val_t: np.ndarray,
    close_val_true: np.ndarray,
    epochs: int,
    batch_size: int,
    patience: int,
    weights_path: str,
) -> Dict[str, Any]:

    best_rmse = np.inf
    best_epoch = -1
    wait = 0
    hist = {"loss": [], "val_loss": [], "val_rmse_close": []}

    lr_cb = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=max(2, patience // 3),
        min_lr=1e-6,
        verbose=0
    )

    ds_tr = make_tf_dataset(X_train, y_train, batch_size, training=True)
    ds_va = make_tf_dataset(X_val, y_val, batch_size, training=False)

    for ep in range(epochs):
        h = model.fit(
            ds_tr,
            validation_data=ds_va,
            epochs=1,
            verbose=0,
            callbacks=[lr_cb]
        )

        tr_loss = float(h.history["loss"][-1])
        va_loss = float(h.history["val_loss"][-1])

        close_val_pred = predict_close(model, X_val, scaler_y, close_val_t)
        va_rmse_close = rmse(close_val_true, close_val_pred)

        hist["loss"].append(tr_loss)
        hist["val_loss"].append(va_loss)
        hist["val_rmse_close"].append(float(va_rmse_close))

        if va_rmse_close < best_rmse:
            best_rmse = va_rmse_close
            best_epoch = ep
            wait = 0
            os.makedirs(os.path.dirname(weights_path), exist_ok=True)
            model.save_weights(weights_path)
        else:
            wait += 1

        if wait >= patience:
            break

    model.load_weights(weights_path)

    return {
        "best_val_rmse_close": float(best_rmse),
        "best_epoch": int(best_epoch + 1),
        "epochs_ran": int(len(hist["loss"])),
        "history": hist
    }


# ─────────────────────────────────────────────────────────────
# 6) Baselines
# ─────────────────────────────────────────────────────────────
def baseline_naive(close_t: np.ndarray, close_true: np.ndarray):
    pred = close_t.copy()
    return rmse(close_true, pred), float(mean_absolute_error(close_true, pred))


def baseline_ma(df_feat: pd.DataFrame, dates_t: pd.DatetimeIndex, close_true: np.ndarray, ma_col: str):
    pred = df_feat.loc[dates_t, ma_col].to_numpy()
    return rmse(close_true, pred), float(mean_absolute_error(close_true, pred))


def baseline_drift(df_feat: pd.DataFrame, dates_t: pd.DatetimeIndex, close_t: np.ndarray, close_true: np.ndarray, horizon: int, window: int):
    mean_lr = df_feat["log_ret_1"].rolling(window).mean().loc[dates_t].to_numpy()
    pred = close_t * np.exp(horizon * mean_lr)
    return rmse(close_true, pred), float(mean_absolute_error(close_true, pred))


# ─────────────────────────────────────────────────────────────
# 7) Plots + Predictions
# ─────────────────────────────────────────────────────────────
def plot_actual_vs_predicted(dates_t, close_true, close_pred, horizon: int, save_path: str):
    dates_target = dates_t + pd.tseries.offsets.BDay(horizon)
    plt.figure(figsize=(12, 5))
    plt.plot(dates_target, close_true, label=f"Actual Close (t+{horizon})")
    plt.plot(dates_target, close_pred, label=f"Pred Close (t+{horizon})")
    plt.title(f"Actual vs Predicted CLOSE (H={horizon})")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path


def plot_history(history: Dict[str, List[float]], horizon: int, save_path: str):
    plt.figure(figsize=(10, 4))
    plt.plot(range(1, len(history["loss"]) + 1), history["loss"], label="Train Loss")
    plt.plot(range(1, len(history["val_loss"]) + 1), history["val_loss"], label="Val Loss")
    plt.plot(range(1, len(history["val_rmse_close"]) + 1), history["val_rmse_close"], label="Val RMSE Close")
    plt.title(f"Training Curves (H={horizon})")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path


def save_predictions_csv(dates_t, horizon, close_t, close_true, close_pred, save_path):
    dates_target = dates_t + pd.tseries.offsets.BDay(horizon)
    dfp = pd.DataFrame({
        "date_t": dates_t,
        "date_target_t_plus_H": dates_target,
        "close_t": close_t,
        "close_true_t_plus_H": close_true,
        "close_pred_t_plus_H": close_pred,
        "abs_error": np.abs(close_true - close_pred),
        "pct_error": (close_pred - close_true) / (close_true + 1e-12) * 100.0,
    })
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    dfp.to_csv(save_path, index=False)
    return save_path


# ─────────────────────────────────────────────────────────────
# 8) Walk-forward CV (bridge-correct) + clear_session
# ─────────────────────────────────────────────────────────────
def walk_forward_cv_score(
    df_feat: pd.DataFrame,
    horizon: int,
    window: int,
    include_close: bool,
    units1: int, units2: int,
    dropout: float, lr: float,
    cfg: RunConfig,
    train_val_end: int,
) -> Dict[str, Any]:

    X_raw, feat_cols = build_feature_matrix(df_feat, include_close)
    y_raw = make_horizon_target(df_feat, horizon)

    X_raw = X_raw[:train_val_end]
    y_raw = y_raw[:train_val_end]
    y_raw_1d = y_raw.flatten()

    close_all = df_feat["Close"].to_numpy()[:train_val_end]
    close_future = df_feat["Close"].shift(-horizon).to_numpy()[:train_val_end]

    fold_rmses: List[float] = []

    start_train = cfg.cv_min_train_size
    for k in range(cfg.cv_folds):
        tr_end = start_train + k * cfg.cv_step
        va_end = tr_end + cfg.cv_val_size

        if va_end >= len(X_raw):
            break
        if tr_end <= window + 50:
            continue

        sx = RobustScaler()
        sy = StandardScaler()
        sx.fit(X_raw[:tr_end])
        sy.fit(y_raw[:tr_end])

        X_scaled = sx.transform(X_raw)
        y_scaled_1d = sy.transform(y_raw).flatten()

        X_tr_seq, y_tr_seq = make_seq_with_context(X_scaled, y_scaled_1d, start=window, end=tr_end, window=window)
        X_va_seq, y_va_seq = make_seq_with_context(X_scaled, y_scaled_1d, start=tr_end, end=va_end, window=window)

        close_val_t = close_all[tr_end:va_end]
        close_val_true = close_future[tr_end:va_end]

        model = build_lstm_model(window, X_tr_seq.shape[-1], units1, units2, dropout, lr)

        tmp_weights = os.path.join(cfg.out_dir_models, "_tmp", f"tmp_{cfg.ticker.lower()}_H{horizon}_fold{k}.weights.h5")
        res = train_select_by_val_rmse_close(
            model,
            X_tr_seq, y_tr_seq,
            X_va_seq, y_va_seq,
            scaler_y=sy,
            close_val_t=close_val_t,
            close_val_true=close_val_true,
            epochs=cfg.epochs_cv,
            batch_size=cfg.batch_size,
            patience=max(5, cfg.patience // 2),
            weights_path=tmp_weights,
        )
        fold_rmses.append(res["best_val_rmse_close"])

        # جلوگیری از کند شدن تجمعی
        tf.keras.backend.clear_session()

    if len(fold_rmses) == 0:
        return {"ok": False, "reason": "No valid folds", "feat_cols": feat_cols}

    return {
        "ok": True,
        "feat_cols": feat_cols,
        "cv_fold_rmses": fold_rmses,
        "cv_mean_rmse": float(np.mean(fold_rmses)),
        "cv_std_rmse": float(np.std(fold_rmses)),
    }


# ─────────────────────────────────────────────────────────────
# 9) Final fit + Test (bridge-correct)
# ─────────────────────────────────────────────────────────────
def fit_final_and_test(
    df_feat: pd.DataFrame,
    horizon: int,
    window: int,
    include_close: bool,
    units1: int, units2: int,
    dropout: float, lr: float,
    cfg: RunConfig,
    train_end: int,
    val_end: int,
    n_valid: int,
) -> Dict[str, Any]:

    X_raw, feat_cols = build_feature_matrix(df_feat, include_close)
    y_raw = make_horizon_target(df_feat, horizon)

    X_raw = X_raw[:n_valid]
    y_raw = y_raw[:n_valid]
    y_raw_1d = y_raw.flatten()
    dates_all = df_feat.index[:n_valid]

    close_all = df_feat["Close"].to_numpy()[:n_valid]
    close_future = df_feat["Close"].shift(-horizon).to_numpy()[:n_valid]

    sx = RobustScaler()
    sy = StandardScaler()
    sx.fit(X_raw[:train_end])
    sy.fit(y_raw[:train_end])

    X_scaled = sx.transform(X_raw)
    y_scaled_1d = sy.transform(y_raw).flatten()

    X_tr_seq, y_tr_seq = make_seq_with_context(X_scaled, y_scaled_1d, start=window, end=train_end, window=window)

    X_va_seq, y_va_seq = make_seq_with_context(X_scaled, y_scaled_1d, start=train_end, end=val_end, window=window)
    close_val_t = close_all[train_end:val_end]
    close_val_true = close_future[train_end:val_end]

    X_te_seq, y_te_seq = make_seq_with_context(X_scaled, y_scaled_1d, start=val_end, end=n_valid, window=window)
    close_test_t = close_all[val_end:n_valid]
    close_test_true = close_future[val_end:n_valid]
    dates_test_t = dates_all[val_end:n_valid]

    model = build_lstm_model(window, X_tr_seq.shape[-1], units1, units2, dropout, lr)

    tag = f"{cfg.ticker.lower()}_H{horizon}_win{window}_close{int(include_close)}_u{units1}-{units2}_do{dropout}_lr{lr}"
    weights_path = os.path.join(cfg.out_dir_models, f"{tag}.best.weights.h5")

    train_res = train_select_by_val_rmse_close(
        model,
        X_tr_seq, y_tr_seq,
        X_va_seq, y_va_seq,
        scaler_y=sy,
        close_val_t=close_val_t,
        close_val_true=close_val_true,
        epochs=cfg.epochs_final,
        batch_size=cfg.batch_size,
        patience=cfg.patience,
        weights_path=weights_path
    )

    close_test_pred = predict_close(model, X_te_seq, sy, close_test_t)

    test_rmse = rmse(close_test_true, close_test_pred)
    test_mae = float(mean_absolute_error(close_test_true, close_test_pred))
    test_nrmse = nrmse(close_test_true, close_test_pred)
    test_mape = mape(close_test_true, close_test_pred)
    test_diracc = directional_accuracy(close_test_t, close_test_true, close_test_pred)

    rmse_naive, mae_naive = baseline_naive(close_test_t, close_test_true)
    rmse_ma5, mae_ma5 = baseline_ma(df_feat, dates_test_t, close_test_true, "ma_5")
    rmse_ma10, mae_ma10 = baseline_ma(df_feat, dates_test_t, close_test_true, "ma_10")
    rmse_drift, mae_drift = baseline_drift(df_feat, dates_test_t, close_test_t, close_test_true, horizon, window)

    ensure_dirs(cfg.out_dir_models, cfg.out_dir_scalers, cfg.out_dir_plots, cfg.out_dir_preds, cfg.out_dir_metrics)

    model_path = os.path.join(cfg.out_dir_models, f"{cfg.ticker.lower()}_H{horizon}_FINAL_best.keras")
    model.save(model_path)

    scaler_x_path = os.path.join(cfg.out_dir_scalers, f"{cfg.ticker.lower()}_H{horizon}_FINAL_scaler_X.joblib")
    scaler_y_path = os.path.join(cfg.out_dir_scalers, f"{cfg.ticker.lower()}_H{horizon}_FINAL_scaler_y.joblib")
    joblib.dump(sx, scaler_x_path)
    joblib.dump(sy, scaler_y_path)

    plot_path = os.path.join(cfg.out_dir_plots, f"{cfg.ticker.lower()}_H{horizon}_FINAL_actual_vs_predicted.png")
    hist_path = os.path.join(cfg.out_dir_plots, f"{cfg.ticker.lower()}_H{horizon}_FINAL_history.png")
    preds_path = os.path.join(cfg.out_dir_preds, f"{cfg.ticker.lower()}_H{horizon}_FINAL_predictions.csv")

    plot_actual_vs_predicted(dates_test_t, close_test_true, close_test_pred, horizon, plot_path)
    plot_history(train_res["history"], horizon, hist_path)
    save_predictions_csv(dates_test_t, horizon, close_test_t, close_test_true, close_test_pred, preds_path)

    return {
        "tag": tag,
        "features": feat_cols,
        "train": train_res,
        "test": {
            "rmse": float(test_rmse),
            "mae": float(test_mae),
            "nrmse": float(test_nrmse),
            "mape": float(test_mape),
            "directional_acc_pct": float(test_diracc),
        },
        "baselines_test": {
            "naive": {"rmse": float(rmse_naive), "mae": float(mae_naive)},
            "drift": {"rmse": float(rmse_drift), "mae": float(mae_drift), "window": int(window)},
            "ma_5": {"rmse": float(rmse_ma5), "mae": float(mae_ma5)},
            "ma_10": {"rmse": float(rmse_ma10), "mae": float(mae_ma10)},
        },
        "artifacts": {
            "model": model_path,
            "weights_best": weights_path,
            "scaler_X": scaler_x_path,
            "scaler_y": scaler_y_path,
            "plot_actual_vs_predicted": plot_path,
            "plot_history": hist_path,
            "predictions_csv": preds_path,
        }
    }


# ─────────────────────────────────────────────────────────────
# 10) Main
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cfg = RunConfig()
    ensure_dirs(cfg.out_dir_metrics, cfg.out_dir_models, cfg.out_dir_scalers, cfg.out_dir_plots, cfg.out_dir_preds)

    df = load_prices_ohlcv(cfg.csv_path, cfg.date_format)
    df_feat = add_price_features(df)

    for H in cfg.horizons:
        print(f"\n {cfg.ticker} | Horizon={H} ")

        train_end, val_end, n_valid = split_final_holdout(df_feat, H, cfg.train_ratio, cfg.val_ratio)
        train_val_end = val_end

        best: Dict[str, Any] = {"cv_mean_rmse": float("inf")}

        for win in cfg.window_candidates:
            for inc_close in cfg.include_close_candidates:
                for u1 in cfg.units1_candidates:
                    for u2 in cfg.units2_candidates:
                        for do in cfg.dropout_candidates:
                            for lr in cfg.lr_candidates:

                                cv = walk_forward_cv_score(
                                    df_feat=df_feat,
                                    horizon=H,
                                    window=win,
                                    include_close=inc_close,
                                    units1=u1, units2=u2,
                                    dropout=do, lr=lr,
                                    cfg=cfg,
                                    train_val_end=train_val_end
                                )

                                if not cv.get("ok", False):
                                    continue

                                mean_rmse = cv["cv_mean_rmse"]
                                print(f"H={H} win={win} close={inc_close} u=({u1},{u2}) do={do} lr={lr} -> CV_RMSE={mean_rmse:.4f} ±{cv['cv_std_rmse']:.4f}")

                                if mean_rmse < best["cv_mean_rmse"]:
                                    best = {
                                        "cv_mean_rmse": mean_rmse,
                                        "cv_std_rmse": cv["cv_std_rmse"],
                                        "cv_fold_rmses": cv["cv_fold_rmses"],
                                        "win": win,
                                        "include_close": inc_close,
                                        "units1": u1,
                                        "units2": u2,
                                        "dropout": do,
                                        "lr": lr,
                                        "features": cv["feat_cols"],
                                    }

        if best["cv_mean_rmse"] == float("inf"):
            print(f"[ERROR] No valid CV configs found for H={H}.")
            continue

        final_res = fit_final_and_test(
            df_feat=df_feat,
            horizon=H,
            window=best["win"],
            include_close=best["include_close"],
            units1=best["units1"],
            units2=best["units2"],
            dropout=best["dropout"],
            lr=best["lr"],
            cfg=cfg,
            train_end=train_end,
            val_end=val_end,
            n_valid=n_valid
        )

        metrics = {
            "ticker": cfg.ticker,
            "horizon_days": H,
            "seed": SEED,
            "selection": {
                "method": "walk_forward_cv_mean_rmse_close",
                "cv_folds": cfg.cv_folds,
                "cv_mean_rmse": float(best["cv_mean_rmse"]),
                "cv_std_rmse": float(best["cv_std_rmse"]),
                "cv_fold_rmses": list(best["cv_fold_rmses"]),
            },
            "best_config": {
                "window_size": int(best["win"]),
                "include_close_in_X": bool(best["include_close"]),
                "units1": int(best["units1"]),
                "units2": int(best["units2"]),
                "dropout": float(best["dropout"]),
                "lr": float(best["lr"]),
                "features": list(best["features"]),
            },
            "final_training": final_res["train"],
            "test": final_res["test"],
            "baselines_test": final_res["baselines_test"],
            "artifacts": final_res["artifacts"],
            "run_config": asdict(cfg),
        }

        metrics_path = os.path.join(cfg.out_dir_metrics, f"{cfg.ticker.lower()}_H{H}_FINAL_metrics.json")
        save_json(metrics, metrics_path)

        print("\n✅ FINAL BEST (CV-selected → Val-selected → Tested on holdout)")
        print(f"H={H} | win={best['win']} | close={best['include_close']} | u=({best['units1']},{best['units2']}) | do={best['dropout']} | lr={best['lr']}")
        print(f"CV mean RMSE: {best['cv_mean_rmse']:.4f} ± {best['cv_std_rmse']:.4f}")
        print(f"TEST RMSE/MAE: {final_res['test']['rmse']:.4f} / {final_res['test']['mae']:.4f}")
        print(f"TEST NRMSE/MAPE/DirAcc: {final_res['test']['nrmse']:.4f} / {final_res['test']['mape']:.2f}% / {final_res['test']['directional_acc_pct']:.2f}%")
        print("Saved metrics:", metrics_path)
        print("Artifacts:", final_res["artifacts"])
