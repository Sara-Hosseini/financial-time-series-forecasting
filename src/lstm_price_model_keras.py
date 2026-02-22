
import os
import json
import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass, asdict
from typing import Tuple, List

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
@dataclass
class RunConfig:
    ticker: str = "AAPL"
    csv_path: str = "data/raw/AAPL_daily_10y_yahoo.csv"
    date_format: str = "%Y-%m-%d"

    horizons: Tuple[int, ...] = (5, 10)

    test_years: Tuple[int, ...] = (2021, 2022, 2023, 2024, 2025)
    val_ratio_of_train: float = 0.10  

    window: int = 30                  
    min_train_rows: int = 500         

    units1: int = 64
    units2: int = 32
    dropout: float = 0.25
    lr: float = 5e-4

    epochs: int = 150
    batch_size: int = 32

    es_patience: int = 20
    es_min_delta: float = 1e-5

    lr_patience: int = 6
    lr_factor: float = 0.5
    lr_min: float = 1e-6

    out_dir_models: str = "outputs/models"
    out_dir_scalers: str = "outputs/scalers"
    out_dir_metrics: str = "outputs/metrics"
    out_dir_plots: str = "outputs/plots"
    out_dir_preds: str = "outputs/predictions"

    save_model: bool = False   # set True to save each fold's model
    save_scalers: bool = False

# Helpers
def ensure_dirs(*paths: str):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def save_json(obj: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)

# Metrics
def rmse(a, b) -> float:
    return float(np.sqrt(mean_squared_error(a, b)))

def mae_m(a, b) -> float:
    return float(mean_absolute_error(a, b))

def mape(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    mask = np.abs(a) > 1e-8
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((b[mask] - a[mask]) / a[mask])) * 100.0)

def dir_acc(c_t, c_true, c_pred) -> float:
    c_t, c_true, c_pred = map(np.asarray, (c_t, c_true, c_pred))
    actual = np.sign(c_true - c_t)
    pred   = np.sign(c_pred - c_t)
    mask   = actual != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(actual[mask] == pred[mask]) * 100.0)


# Data Loading

def load_prices(csv_path: str, date_format: str = "%Y-%m-%d") -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    first_cell = str(df.iloc[0, 0]).strip().lower() if len(df) >= 1 else ""
    if first_cell == "ticker":
        df = df.iloc[2:].copy()

    if "Price" in df.columns and "Date" not in df.columns:
        df = df.rename(columns={"Price": "Date"})
    if "Date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Date"})

    df["Date"] = pd.to_datetime(df["Date"], format=date_format, errors="coerce")
    df = (
        df.dropna(subset=["Date"])
          .sort_values("Date")
          .drop_duplicates(subset=["Date"], keep="last")
          .set_index("Date")
    )

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Close"])
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    return df[keep].dropna()

# Feature Engineering
FEATURE_COLS = [
    "log_ret_1", "ret_5", "ret_10",
    "ma_ratio_5", "ma_ratio_10",
    "hl_range",
    "vol_10", "vol_20",
    "rsi_14",
]

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["log_ret_1"] = np.log(out["Close"] / out["Close"].shift(1))
    out["ret_5"]     = np.log(out["Close"] / out["Close"].shift(5))
    out["ret_10"]    = np.log(out["Close"] / out["Close"].shift(10))

    out["ma_ratio_5"]  = out["Close"].rolling(5).mean()  / (out["Close"] + 1e-12) - 1
    out["ma_ratio_10"] = out["Close"].rolling(10).mean() / (out["Close"] + 1e-12) - 1

    out["hl_range"] = (out["High"] - out["Low"]) / (out["Close"] + 1e-12)

    out["vol_10"] = out["log_ret_1"].rolling(10).std()
    out["vol_20"] = out["log_ret_1"].rolling(20).std()

    delta = out["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-12)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    return out.dropna()


# Target: log-return over H days
def make_target(df: pd.DataFrame, H: int) -> pd.Series:
    return np.log(df["Close"].shift(-H) / (df["Close"] + 1e-12)).rename("target")

# Sequence Builder — explicit index tracking
def make_sequences(
    X_scaled: np.ndarray,
    y_1d: np.ndarray,
    idx_start: int,
    idx_end: int,
    window: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Xs, ys, idxs = [], [], []
    for t in range(idx_start, idx_end):
        if t < window:
            continue
        Xs.append(X_scaled[t - window : t, :])
        ys.append(y_1d[t])
        idxs.append(t)
    return np.array(Xs), np.array(ys).reshape(-1, 1), np.array(idxs)


# Model
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


# Plot: all folds combined
def save_combined_plot(ticker, all_dates, all_true, all_pred, H, save_path):
    all_dates = pd.DatetimeIndex(all_dates)
    pct_err   = (all_pred - all_true) / (all_true + 1e-12) * 100

    fig, axes = plt.subplots(2, 1, figsize=(16, 8),
                             gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    ax.plot(all_dates, all_true, label=f"Actual Close (t+{H})", linewidth=1.5)
    ax.plot(all_dates, all_pred, label=f"Pred  Close (t+{H})", linewidth=1.2, alpha=0.85)
    ax.set_title(f"{ticker} — Walk-Forward Test Predictions  (H={H} days, all folds)")
    ax.set_ylabel("Close Price")
    ax.legend()
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    pos = np.where(pct_err >= 0, pct_err, 0)
    neg = np.where(pct_err <  0, pct_err, 0)
    ax2.bar(all_dates, pos, color="green", alpha=0.6, width=1, label="Over-pred")
    ax2.bar(all_dates, neg, color="red",   alpha=0.6, width=1, label="Under-pred")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("% Error")
    ax2.set_xlabel("Date")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=180)
    plt.close()


def save_metrics_plot(fold_metrics: List[dict], H: int, save_path: str):
    """Bar chart of per-fold metrics for thesis."""
    years  = [str(m["test_year"]) for m in fold_metrics]
    rmses  = [m["rmse"]  for m in fold_metrics]
    mapes  = [m["mape"]  for m in fold_metrics]
    daccs  = [m["dir_acc"] for m in fold_metrics]

    x = np.arange(len(years))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, vals, label, color in zip(
        axes,
        [rmses, mapes, daccs],
        ["RMSE ($)", "MAPE (%)", "Dir Accuracy (%)"],
        ["steelblue", "darkorange", "seagreen"],
    ):
        bars = ax.bar(x, vals, color=color, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(years)
        ax.set_title(label)
        ax.set_xlabel("Test Year")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01 * max(vals),
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    plt.suptitle(f"Per-Fold Metrics  (H={H} days)", fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=160)
    plt.close()

# Main
if __name__ == "__main__":
    cfg = RunConfig()
    ensure_dirs(
        cfg.out_dir_models, cfg.out_dir_scalers,
        cfg.out_dir_metrics, cfg.out_dir_plots, cfg.out_dir_preds,
    )

    print(f"Loading data: {cfg.csv_path}")
    df_raw  = load_prices(cfg.csv_path, cfg.date_format)
    df_feat = add_features(df_raw)
    print(f"Dataset: {len(df_feat)} rows  "
          f"({df_feat.index[0].date()} → {df_feat.index[-1].date()})")

    for H in cfg.horizons:
        print(f"\n{'='*65}")
        print(f"  {cfg.ticker}  |  Horizon = {H} days  |  Walk-Forward")
        print(f"{'='*65}")

        # ── Prepare full aligned arrays
        y_full  = make_target(df_feat, H).dropna()
        df_work = df_feat.loc[y_full.index].copy()

        X_raw = df_work[FEATURE_COLS].values
        y_raw = y_full.values
        dates = df_work.index
        n     = len(y_raw)

        fold_metrics   = []
        all_dates_list = []
        all_true_list  = []
        all_pred_list  = []
        all_preds_df   = []

        # ── Walk-Forward Folds
        for test_year in cfg.test_years:
            # Test window: full calendar year
            test_start = pd.Timestamp(f"{test_year}-01-01")
            test_end   = pd.Timestamp(f"{test_year}-12-31")

            # Train: everything BEFORE test year
            train_mask = dates < test_start
            test_mask  = (dates >= test_start) & (dates <= test_end)

            train_idx = np.where(train_mask)[0]
            test_idx  = np.where(test_mask)[0]

            if len(train_idx) < cfg.min_train_rows:
                print(f"\n  [Fold {test_year}] Only {len(train_idx)} train rows — skipping.")
                continue
            if len(test_idx) == 0:
                print(f"\n  [Fold {test_year}]  No test data — skipping.")
                continue

            train_end_idx = int(train_idx[-1]) + 1  # exclusive end
            val_start_idx = int(train_end_idx - len(train_idx) * cfg.val_ratio_of_train)

            print(f"\n  ── Fold {test_year} ──────────────────────────────────────")
            print(f"  Train: rows 0–{val_start_idx-1} ({val_start_idx} rows)")
            print(f"  Val  : rows {val_start_idx}–{train_end_idx-1} ({train_end_idx-val_start_idx} rows)")
            print(f"  Test : {test_start.date()} → {test_end.date()} ({len(test_idx)} rows)")

            # ── Scaler fit on train only (leakage-safe)
            scaler_X = RobustScaler().fit(X_raw[:val_start_idx])
            scaler_y = StandardScaler().fit(y_raw[:val_start_idx].reshape(-1, 1))

            X_s = scaler_X.transform(X_raw)
            y_s = scaler_y.transform(y_raw.reshape(-1, 1)).flatten()

            # ── Sequences
            X_tr, y_tr, _       = make_sequences(X_s, y_s, 0,             val_start_idx,  cfg.window)
            X_va, y_va, _       = make_sequences(X_s, y_s, val_start_idx, train_end_idx,  cfg.window)
            X_te, y_te, idx_te  = make_sequences(X_s, y_s,
                                                  int(test_idx[0]),
                                                  int(test_idx[-1]) + 1,
                                                  cfg.window)

            if len(X_tr) == 0 or len(X_va) == 0 or len(X_te) == 0:
                print(f"  [Fold {test_year}]   Empty split after sequencing — skipping.")
                continue

            # ── Price alignment (from df_work, leakage-safe)
            close_t    = df_work["Close"].values[idx_te]
            close_true = df_work["Close"].shift(-H).values[idx_te]
            dates_te   = dates[idx_te]

            # Trim NaNs (last H rows of test window have no future close)
            nan_mask   = np.isnan(close_true)
            if nan_mask.any():
                X_te       = X_te[~nan_mask]
                close_t    = close_t[~nan_mask]
                close_true = close_true[~nan_mask]
                dates_te   = dates_te[~nan_mask]
                idx_te     = idx_te[~nan_mask]

            if len(X_te) == 0:
                print(f"  [Fold {test_year}]  All test rows were NaN — skipping.")
                continue

            # ── Build & train (fresh model each fold)
            tf.keras.backend.clear_session()
            model = build_model(cfg.window, X_tr.shape[-1], cfg)

            callbacks = [
                EarlyStopping(
                    monitor="val_loss",
                    patience=cfg.es_patience,
                    min_delta=cfg.es_min_delta,
                    restore_best_weights=True,
                    verbose=0,
                ),
                ReduceLROnPlateau(
                    monitor="val_loss",
                    patience=cfg.lr_patience,
                    factor=cfg.lr_factor,
                    min_lr=cfg.lr_min,
                    verbose=0,
                ),
            ]

            history = model.fit(
                X_tr, y_tr,
                validation_data=(X_va, y_va),
                epochs=cfg.epochs,
                batch_size=cfg.batch_size,
                callbacks=callbacks,
                verbose=0,
                shuffle=False,
            )

            best_epoch = int(np.argmin(history.history["val_loss"])) + 1
            best_val   = float(np.min(history.history["val_loss"]))
            print(f"  Best epoch: {best_epoch}  |  best val_loss: {best_val:.5f}")

            # ── Predict → price
            y_pred_s   = model.predict(X_te, verbose=0)
            y_pred     = scaler_y.inverse_transform(y_pred_s).flatten()
            close_pred = close_t * np.exp(y_pred)

            # ── Metrics
            r  = rmse(close_true, close_pred)
            m  = mae_m(close_true, close_pred)
            mp = mape(close_true, close_pred)
            da = dir_acc(close_t, close_true, close_pred)

            print(f"  RMSE: {r:.4f}  MAE: {m:.4f}  MAPE: {mp:.2f}%  DirAcc: {da:.2f}%")

            fold_metrics.append({
                "test_year": test_year,
                "best_epoch": best_epoch,
                "best_val_loss": best_val,
                "rmse": r, "mae": m, "mape": mp, "dir_acc": da,
                "n_test": int(len(X_te)),
            })

            all_dates_list.extend(dates_te + pd.tseries.offsets.BDay(H))
            all_true_list.extend(close_true)
            all_pred_list.extend(close_pred)

            all_preds_df.append(pd.DataFrame({
                "date_t":              dates_te,
                "date_t_plus_H":       dates_te + pd.tseries.offsets.BDay(H),
                "test_year":           test_year,
                "close_t":             close_t,
                "close_true_t_plus_H": close_true,
                "close_pred_t_plus_H": close_pred,
                "log_ret_true":        y_raw[idx_te],
                "log_ret_pred":        y_pred,
                "abs_error":           np.abs(close_true - close_pred),
                "pct_error":           (close_pred - close_true) / (close_true + 1e-12) * 100,
            }))

            if cfg.save_model:
                mp_ = os.path.join(cfg.out_dir_models,
                                   f"{cfg.ticker.lower()}_H{H}_{test_year}.keras")
                model.save(mp_)

            if cfg.save_scalers:
                joblib.dump(scaler_X, os.path.join(cfg.out_dir_scalers,
                            f"{cfg.ticker.lower()}_H{H}_{test_year}_scaler_X.joblib"))
                joblib.dump(scaler_y, os.path.join(cfg.out_dir_scalers,
                            f"{cfg.ticker.lower()}_H{H}_{test_year}_scaler_y.joblib"))

        # ── End of folds
        if not fold_metrics:
            print(f"\n  ⚠️  No folds completed for H={H}.")
            continue

        # ── Aggregate metrics
        avg_rmse = float(np.mean([m["rmse"]    for m in fold_metrics]))
        avg_mae  = float(np.mean([m["mae"]     for m in fold_metrics]))
        avg_mape = float(np.mean([m["mape"]    for m in fold_metrics]))
        avg_da   = float(np.mean([m["dir_acc"] for m in fold_metrics]))

        print(f"\n  ══ Walk-Forward Summary  (H={H}) ___________________")
        print(f"  Avg RMSE        : {avg_rmse:.4f}")
        print(f"  Avg MAE         : {avg_mae:.4f}")
        print(f"  Avg MAPE        : {avg_mape:.2f}%")
        print(f"  Avg Dir Accuracy: {avg_da:.2f}%")
        print(f"  Folds completed : {len(fold_metrics)}")
        print(f"  _________________________________")

        # ── Save outputs
        tag = f"{cfg.ticker.lower()}_H{H}_wf"

        plot_path   = os.path.join(cfg.out_dir_plots,   f"{tag}_pred.png")
        mplot_path  = os.path.join(cfg.out_dir_plots,   f"{tag}_fold_metrics.png")
        pred_path   = os.path.join(cfg.out_dir_preds,   f"{tag}_predictions.csv")
        metr_path   = os.path.join(cfg.out_dir_metrics, f"{tag}_metrics.json")

        all_dates_arr = pd.DatetimeIndex(all_dates_list)
        all_true_arr  = np.array(all_true_list)
        all_pred_arr  = np.array(all_pred_list)

        save_combined_plot(cfg.ticker, all_dates_arr, all_true_arr, all_pred_arr,
                           H, plot_path)
        save_metrics_plot(fold_metrics, H, mplot_path)
        print(f"  Plots → {plot_path}")
        print(f"          {mplot_path}")

        pd.concat(all_preds_df, ignore_index=True).to_csv(pred_path, index=False)
        print(f"  Preds → {pred_path}")

        summary = {
            "ticker":       cfg.ticker,
            "horizon_days": H,
            "window":       cfg.window,
            "feature_cols": FEATURE_COLS,
            "method":       "walk_forward_expanding_window",
            "folds":        fold_metrics,
            "aggregate": {
                "avg_rmse":    avg_rmse,
                "avg_mae":     avg_mae,
                "avg_mape":    avg_mape,
                "avg_dir_acc": avg_da,
                "n_folds":     len(fold_metrics),
            },
            "run_config": asdict(cfg),
            "artifacts": {
                "plot":       plot_path,
                "fold_plot":  mplot_path,
                "preds_csv":  pred_path,
            },
        }
        save_json(summary, metr_path)
        print(f"  Metrics → {metr_path}")

    print("\n✅  Walk-Forward complete.")