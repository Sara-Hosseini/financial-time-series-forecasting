# src/models/lstm_quantile.py
# Quantile Regression LSTM — Smart Price Range Predictor
#
# Usage:
#   python -m src.models.lstm_quantile          -> update data + predict (daily use)
#   python -m src.models.lstm_quantile --train  -> force retrain + predict
#   python -m src.models.lstm_quantile --eval   -> full Walk-Forward evaluation

import warnings
warnings.filterwarnings("ignore")
import os
import sys
import json
from datetime import date, datetime, timedelta
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from dataclasses import dataclass
from typing import Tuple
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import yfinance as yf

from src.utils import (
    load_prices, add_features, make_target,
    make_sequences, mape, dir_acc, coverage, interval_width,
    ensure_dirs, save_json,
)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
PRICE_FEATURE_COLS = [
    "log_ret_1", "ret_5", "ret_10",
    "ma_ratio_5", "ma_ratio_10",
    "hl_range",
    "vol_10", "vol_20",
    "rsi_14",
]

SENTIMENT_FEATURE_COLS = [
    "news_count",
    "avg_sentiment_score",
    "avg_positive_prob",
    "avg_negative_prob",
    "avg_neutral_prob",
    "max_positive_prob",
    "max_negative_prob",
    "pos_news_count",
    "neg_news_count",
    "neu_news_count",
    "sentiment_balance",
    "positive_ratio",
    "negative_ratio",
    "neutral_ratio",
    "has_news",
]

FEATURE_COLS = PRICE_FEATURE_COLS  # ABLATION: price-only, sentiment removed

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
@dataclass
class QuantileConfig:
    ticker: str = os.environ.get("TICKER", "AAPL")

    @property
    def csv_path(self) -> str:
        return f"data/raw/{self.ticker}_daily_10y_yahoo.csv"

    @property
    def merged_csv_path(self) -> str:
        return f"outputs/{self.ticker}_price_sentiment_merged.csv"

    date_format: str = "%Y-%m-%d"
    horizons: Tuple[int, ...] = (1, 5, 10)
    test_years: Tuple[int, ...] = (2021, 2022, 2023, 2024, 2025)
    val_ratio_of_train: float = 0.10
    quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9)
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
    retrain_mape_threshold: float = 7.0
    out_dir_models: str = "outputs/models"
    out_dir_scalers: str = "outputs/scalers"
    out_dir_metrics: str = "outputs/metrics"
    out_dir_plots: str = "outputs/plots"
    out_dir_preds: str = "outputs/predictions"

def load_merged_features(cfg: QuantileConfig) -> pd.DataFrame:
    """
    Loads merged price + sentiment data and builds final model features.
    """
    if not os.path.exists(cfg.merged_csv_path):
        raise FileNotFoundError(f"Merged dataset not found: {cfg.merged_csv_path}")

    df = pd.read_csv(cfg.merged_csv_path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()
    df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    df = df.set_index("Date")

    # make sure price columns are numeric
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # build price-derived features
    df = add_features(df)

    # ensure sentiment columns exist
    for col in SENTIMENT_FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0

    df[SENTIMENT_FEATURE_COLS] = df[SENTIMENT_FEATURE_COLS].apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0.0)

    df = df.dropna(subset=["Close"]).copy()
    return df


def _load_features_with_fallback(cfg: QuantileConfig) -> pd.DataFrame:
    """
    Tries to load merged CSV (price + sentiment).
    If file is missing or corrupted, falls back to price-only data
    and fills sentiment columns with zero.
    """
    # Case 1: merged CSV exists -> load it
    if os.path.exists(cfg.merged_csv_path):
        try:
            df = load_merged_features(cfg)
            n_sent = sum(1 for c in SENTIMENT_FEATURE_COLS if c in df.columns and df[c].any())
            if n_sent > 0:
                print(f"  Loaded merged features ({len(df)} rows, {n_sent} active sentiment cols)")
            else:
                print(f"  Loaded merged features ({len(df)} rows, sentiment cols all zero - no news data yet)")
            return df
        except Exception as e:
            print(f"  Warning: merged CSV load failed ({e}). Falling back to price-only.")

    # Case 2: fallback - price-only data
    print(f"  Merged CSV not found at '{cfg.merged_csv_path}'.")
    print("  Using price-only features. Sentiment columns will be zero.")
    print("  To add sentiment: run the full pipeline (collector → sentiment_model → merge).")

    if not os.path.exists(cfg.csv_path):
        msg = (
            "Neither merged CSV nor price CSV found.\n"
            f"  merged : {cfg.merged_csv_path}\n"
            f"  price  : {cfg.csv_path}\n"
            "Run data_collection.py first."
        )
        raise FileNotFoundError(msg)

    df = load_prices(cfg.csv_path, cfg.date_format)
    df = add_features(df)

    # Add sentiment columns with zero values (neutral baseline)
    for col in SENTIMENT_FEATURE_COLS:
        df[col] = 0.0

    # Save as merged CSV so next run does not need fallback
    os.makedirs(os.path.dirname(cfg.merged_csv_path), exist_ok=True)
    price_cols = ["Open", "High", "Low", "Close", "Volume"] + SENTIMENT_FEATURE_COLS
    save_cols  = [c for c in price_cols if c in df.columns]
    df[save_cols].to_csv(cfg.merged_csv_path)
    print(f"  Saved price-only merged CSV -> {cfg.merged_csv_path}")

    return df


# ─────────────────────────────────────────────
# Pinball Loss
# ─────────────────────────────────────────────
def pinball_loss(quantiles):
    quantiles_tf = tf.constant(quantiles, dtype=tf.float32)

    def loss(y_true, y_pred):
        y_true = tf.repeat(y_true, len(quantiles), axis=-1)
        errors = y_true - y_pred
        losses = tf.maximum(quantiles_tf * errors, (quantiles_tf - 1.0) * errors)
        return tf.reduce_mean(losses)

    return loss


# ─────────────────────────────────────────────
# Model Architecture
# ─────────────────────────────────────────────
def build_model(window: int, n_features: int, cfg: QuantileConfig) -> tf.keras.Model:
    n_q = len(cfg.quantiles)
    model = Sequential([
        Input(shape=(window, n_features)),
        LSTM(cfg.units1, return_sequences=True),
        Dropout(cfg.dropout),
        LSTM(cfg.units2),
        Dropout(cfg.dropout),
        Dense(32, activation="relu"),
        Dense(n_q),  # [q10, q50, q90]
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.lr, clipnorm=1.0),
        loss=pinball_loss(cfg.quantiles),
    )
    return model


# ─────────────────────────────────────────────
# Step 1 — Update data from Yahoo Finance
# ─────────────────────────────────────────────
def update_price_data(cfg: QuantileConfig) -> pd.DataFrame:
    print(f"\n[1/4] Updating price data for {cfg.ticker}...")
    os.makedirs(os.path.dirname(cfg.csv_path), exist_ok=True)

    if os.path.exists(cfg.csv_path):
        df_existing = load_prices(cfg.csv_path, cfg.date_format)
        last_date   = df_existing.index[-1]
        fetch_start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"  Existing data up to: {last_date.date()}")
    else:
        df_existing = pd.DataFrame()
        fetch_start = "2015-01-01"
        print(f"  No existing data. Downloading full history...")

    today = date.today().isoformat()
    if fetch_start > today:
        print(f"  Already up to date.")
        return add_features(load_prices(cfg.csv_path, cfg.date_format))

    new_df = yf.download(cfg.ticker, start=fetch_start, end=today,
                         interval="1d", auto_adjust=False, progress=False)

    if new_df is None or new_df.empty:
        print(f"  No new data from Yahoo. Using existing data.")
        return add_features(load_prices(cfg.csv_path, cfg.date_format))

    if isinstance(new_df.columns, pd.MultiIndex):
        new_df.columns = [col[0] for col in new_df.columns]

    new_df.index = pd.to_datetime(new_df.index)
    new_df = new_df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    print(f"  Downloaded {len(new_df)} new rows "
          f"({new_df.index[0].date()} -> {new_df.index[-1].date()})")

    if not df_existing.empty:
        merged = pd.concat([df_existing, new_df])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    else:
        merged = new_df

    merged.to_csv(cfg.csv_path)
    print(f"  Saved {len(merged)} total rows -> {cfg.csv_path}")
    return add_features(merged)


# ─────────────────────────────────────────────
# Step 2 — Check if model needs training
# ─────────────────────────────────────────────
def model_needs_training(cfg: QuantileConfig, H: int) -> bool:
    model_path    = os.path.join(cfg.out_dir_models,  f"{cfg.ticker}_quantile_H{H}.keras")
    scaler_X_path = os.path.join(cfg.out_dir_scalers, f"{cfg.ticker}_quantile_H{H}_scaler_X.joblib")
    scaler_y_path = os.path.join(cfg.out_dir_scalers, f"{cfg.ticker}_quantile_H{H}_scaler_y.joblib")

    if not os.path.exists(model_path) or not os.path.exists(scaler_X_path) or not os.path.exists(scaler_y_path):
        print(f"  H={H}: No saved model or scalers. Training required.")
        return True

    meta_path = os.path.join(cfg.out_dir_models, f"{cfg.ticker}_quantile_H{H}_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        last_trained = datetime.strptime(meta["trained_on"], "%Y-%m-%d").date()
        days_since   = (date.today() - last_trained).days
        print(f"  H={H}: Model trained {days_since} days ago ({last_trained})")

    return False


# ─────────────────────────────────────────────
# Step 3 — Train and save model
# ─────────────────────────────────────────────
def train_and_save(df_feat: pd.DataFrame, cfg: QuantileConfig, H: int):
    print(f"\n  Training Quantile LSTM for H={H}d...")
    ensure_dirs(cfg.out_dir_models, cfg.out_dir_scalers)

    target    = make_target(df_feat, H)
    df_work   = df_feat.loc[target.dropna().index].copy()
    y_raw     = target.loc[df_work.index].values
    X_raw     = df_work[FEATURE_COLS].values
    n         = len(y_raw)
    val_start = int(n * (1 - cfg.val_ratio_of_train))

    # FIX: scale both X and y on train portion only (leakage-safe)
    from sklearn.preprocessing import StandardScaler
    scaler_X = RobustScaler().fit(X_raw[:val_start])
    scaler_y = StandardScaler().fit(y_raw[:val_start].reshape(-1, 1))

    X_s = scaler_X.transform(X_raw)
    y_s = scaler_y.transform(y_raw.reshape(-1, 1)).flatten()

    X_tr, y_tr, _ = make_sequences(X_s, y_s, 0,         val_start, cfg.window)
    X_va, y_va, _ = make_sequences(X_s, y_s, val_start, n,         cfg.window)
    print(f"  Train: {len(X_tr)} sequences | Val: {len(X_va)}")

    tf.keras.backend.clear_session()
    model = build_model(cfg.window, X_tr.shape[-1], cfg)
    model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=cfg.es_patience,
                          min_delta=cfg.es_min_delta, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor="val_loss", patience=cfg.lr_patience,
                              factor=cfg.lr_factor, min_lr=cfg.lr_min, verbose=0),
        ],
        verbose=0, shuffle=False,
    )

    model_path    = os.path.join(cfg.out_dir_models,  f"{cfg.ticker}_quantile_H{H}.keras")
    scaler_X_path = os.path.join(cfg.out_dir_scalers, f"{cfg.ticker}_quantile_H{H}_scaler_X.joblib")
    scaler_y_path = os.path.join(cfg.out_dir_scalers, f"{cfg.ticker}_quantile_H{H}_scaler_y.joblib")
    meta_path     = os.path.join(cfg.out_dir_models,  f"{cfg.ticker}_quantile_H{H}_meta.json")

    model.save(model_path)
    joblib.dump(scaler_X, scaler_X_path)
    joblib.dump(scaler_y, scaler_y_path)
    save_json({"trained_on": date.today().isoformat(),
               "n_train": len(X_tr), "ticker": cfg.ticker, "horizon": H}, meta_path)

    print(f"  Model saved -> {model_path}")
    return model, scaler_X, scaler_y


# ─────────────────────────────────────────────
# Step 4 — Evaluate on recent 30 days
# ─────────────────────────────────────────────
def evaluate_recent(df_feat: pd.DataFrame, cfg: QuantileConfig,
                    H: int, model, scaler_X, scaler_y) -> float:
    target  = make_target(df_feat, H)
    df_work = df_feat.loc[target.dropna().index].copy()
    y_raw   = target.loc[df_work.index].values
    X_raw   = df_work[FEATURE_COLS].values
    closes  = df_work["Close"].values
    n       = len(y_raw)

    # FIX: scale y before building sequences, then inverse after predict
    y_s        = scaler_y.transform(y_raw.reshape(-1, 1)).flatten()
    eval_start = max(0, n - 30 - cfg.window)
    X_s        = scaler_X.transform(X_raw)
    X_ev, y_ev_s, ev_idx = make_sequences(X_s, y_s, eval_start, n, cfg.window)

    if len(X_ev) == 0:
        return 0.0

    preds_s     = model.predict(X_ev, verbose=0)  # shape (n, 3) — scaled log-returns
    # inverse transform each quantile column
    preds_log   = scaler_y.inverse_transform(preds_s)
    y_ev_log    = scaler_y.inverse_transform(y_ev_s.reshape(-1, 1)).flatten()

    base_closes = closes[ev_idx]
    q50_price   = base_closes * np.exp(preds_log[:, 1])
    true_price  = base_closes * np.exp(y_ev_log)
    return mape(true_price, q50_price)


# ─────────────────────────────────────────────
# Walk-Forward Evaluation (--eval mode)
# ─────────────────────────────────────────────
def run_walk_forward(cfg: QuantileConfig):
    print(f"\n{'='*55}")
    print(f"  Walk-Forward Evaluation — {cfg.ticker}")
    print(f"{'='*55}")

    ensure_dirs(cfg.out_dir_metrics, cfg.out_dir_plots)
    df_feat = load_merged_features(cfg)
    print(f"Dataset: {len(df_feat)} rows "
          f"({df_feat.index[0].date()} -> {df_feat.index[-1].date()})")
    summary = {}

    for H in cfg.horizons:
        print(f"\n  Quantile LSTM | H={H}d\n  {'-'*50}")
        target  = make_target(df_feat, H)
        df_work = df_feat.loc[target.dropna().index].copy()
        y_raw   = target.loc[df_work.index].values
        X_raw   = df_work[FEATURE_COLS].values
        dates   = df_work.index
        closes  = df_work["Close"].values

        all_dates = []
        all_true, all_low, all_mid, all_high = [], [], [], []
        fold_metrics = []

        for year in cfg.test_years:
            test_mask  = dates.year == year
            train_mask = dates.year < year
            train_idx  = np.where(train_mask)[0]
            test_idx   = np.where(test_mask)[0]

            if len(train_idx) < cfg.min_train_rows or len(test_idx) == 0:
                continue

            val_cut   = int(len(train_idx) * (1 - cfg.val_ratio_of_train))
            train_end = train_idx[val_cut]
            val_end   = train_idx[-1] + 1

            from sklearn.preprocessing import StandardScaler
            scaler_X = RobustScaler().fit(X_raw[train_idx[:val_cut]])
            scaler_y = StandardScaler().fit(y_raw[train_idx[:val_cut]].reshape(-1, 1))
            X_s = scaler_X.transform(X_raw)
            y_s = scaler_y.transform(y_raw.reshape(-1, 1)).flatten()

            X_tr, y_tr, _        = make_sequences(X_s, y_s, 0,           train_end,         cfg.window)
            X_va, y_va, _        = make_sequences(X_s, y_s, train_end,   val_end,           cfg.window)
            X_te, y_te_s, te_idx = make_sequences(X_s, y_s, test_idx[0], test_idx[-1]+1,   cfg.window)

            if len(X_tr) == 0 or len(X_te) == 0:
                continue

            print(f"\n  Fold {year}: train={len(X_tr)} | val={len(X_va)} | test={len(X_te)}")
            tf.keras.backend.clear_session()
            model = build_model(cfg.window, X_tr.shape[-1], cfg)
            model.fit(X_tr, y_tr, validation_data=(X_va, y_va),
                      epochs=cfg.epochs, batch_size=cfg.batch_size,
                      callbacks=[
                          EarlyStopping(monitor="val_loss", patience=cfg.es_patience,
                                        min_delta=cfg.es_min_delta, restore_best_weights=True, verbose=0),
                          ReduceLROnPlateau(monitor="val_loss", patience=cfg.lr_patience,
                                           factor=cfg.lr_factor, min_lr=cfg.lr_min, verbose=0),
                      ], verbose=0, shuffle=False)

            # FIX: inverse transform predictions and true values back to log-return space
            preds_s     = model.predict(X_te, verbose=0)
            preds_log   = scaler_y.inverse_transform(preds_s)
            y_te_log    = scaler_y.inverse_transform(y_te_s.reshape(-1, 1)).flatten()

            base_closes = closes[te_idx]
            q10_price   = base_closes * np.exp(preds_log[:, 0])
            q50_price   = base_closes * np.exp(preds_log[:, 1])
            q90_price   = base_closes * np.exp(preds_log[:, 2])
            true_price  = base_closes * np.exp(y_te_log)

            fold_mape     = mape(true_price, q50_price)
            fold_coverage = coverage(true_price, q10_price, q90_price)
            fold_width    = interval_width(q10_price, q90_price)
            fold_dir      = dir_acc(base_closes, true_price, q50_price)

            fold_metrics.append({"year": year, "mape_pct": round(fold_mape, 4),
                                  "coverage_pct": round(fold_coverage, 2),
                                  "interval_width": round(fold_width, 4),
                                  "dir_acc_pct": round(fold_dir, 2), "n_test": len(X_te)})
            print(f"  MAPE={fold_mape:.2f}%  Coverage={fold_coverage:.1f}%  "
                  f"Width=${fold_width:.2f}  DirAcc={fold_dir:.1f}%")

            all_dates.extend(dates[te_idx])
            all_true.extend(true_price)
            all_low.extend(q10_price)
            all_mid.extend(q50_price)
            all_high.extend(q90_price)

        if not fold_metrics:
            continue

        avg_mape     = np.mean([f["mape_pct"]       for f in fold_metrics])
        avg_coverage = np.mean([f["coverage_pct"]   for f in fold_metrics])
        avg_width    = np.mean([f["interval_width"]  for f in fold_metrics])
        avg_dir      = np.mean([f["dir_acc_pct"]    for f in fold_metrics])

        print(f"\n  Overall: MAPE={avg_mape:.2f}%  Coverage={avg_coverage:.1f}%  "
              f"Width=${avg_width:.2f}  DirAcc={avg_dir:.1f}%")

        summary[f"H{H}"] = {"avg_mape_pct": round(avg_mape, 4),
                             "avg_coverage_pct": round(avg_coverage, 2),
                             "avg_width": round(avg_width, 4),
                             "avg_dir_acc_pct": round(avg_dir, 2),
                             "folds": fold_metrics}

        fig, ax = plt.subplots(figsize=(14, 6))
        plot_dates = pd.DatetimeIndex(all_dates)
        ax.fill_between(plot_dates, np.array(all_low), np.array(all_high),
                        alpha=0.25, color="steelblue", label="Interval (q10-q90)")
        ax.plot(plot_dates, np.array(all_mid),  color="steelblue", linewidth=1.5,
                label="Predicted median (q50)", zorder=3)
        ax.plot(plot_dates, np.array(all_true), color="tomato", linewidth=1.2,
                label="Actual price", zorder=4)
        ax.set_title(f"{cfg.ticker} — Quantile LSTM | H={H}d | Walk-Forward")
        ax.set_xlabel("Date"); ax.set_ylabel("Close Price ($)")
        ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
        plot_path = os.path.join(cfg.out_dir_plots, f"{cfg.ticker}_quantile_eval_H{H}.png")
        plt.savefig(plot_path, dpi=180); plt.close()
        print(f"  Plot -> {plot_path}")
        save_json(summary[f"H{H}"],
                  os.path.join(cfg.out_dir_metrics, f"{cfg.ticker}_quantile_H{H}.json"))

    save_json(summary, os.path.join(cfg.out_dir_metrics,
                                    f"{cfg.ticker}_quantile_summary.json"))
    print("\nWalk-Forward evaluation complete.")
    return summary


# ─────────────────────────────────────────────
# Main — Smart daily runner
# ─────────────────────────────────────────────
def run(cfg: QuantileConfig, force_train: bool = False):
    ensure_dirs(cfg.out_dir_models, cfg.out_dir_scalers,
                cfg.out_dir_metrics, cfg.out_dir_plots, cfg.out_dir_preds)

    # Step 1 — Update price data
    update_price_data(cfg)

    # Step 1b — Update sentiment (collect + score + merge)
    _lora_dir = "outputs/models/finbert_lora_best"
    _lora_ok  = (
        os.path.isdir(_lora_dir) and
        any(f.endswith((".bin", ".safetensors", "adapter_config.json"))
            for f in os.listdir(_lora_dir))
    ) if os.path.isdir(_lora_dir) else False

    if not _lora_ok:
        print("\n[1b/4] LoRA model not found at '%s'." % _lora_dir)
        print("       Skipping sentiment update — will use price-only features.")
        print("       To enable sentiment: run train_finbert_phrasebank.py first.")
    else:
        try:
            from src.sentiment.collector import collect_and_label
            from src.sentiment.sentiment_model import (
                load_news, load_finbert_lora, score_news, build_daily_sentiment,
            )
            from src.merge_sentiment_price import (
                load_price_data, load_sentiment_data, merge_price_and_sentiment,
            )
            import src.merge_sentiment_price as _msp

            print("\n[1b/4] Updating sentiment data...")
            collect_and_label()

            df_news               = load_news("data/processed/sentiment/news_raw.csv")
            tokenizer, sent_model = load_finbert_lora()
            df_scored             = score_news(df_news, tokenizer, sent_model)
            daily_df              = build_daily_sentiment(df_scored)
            daily_path = f"outputs/{cfg.ticker}_news_daily.csv"
            daily_df_ticker = daily_df[daily_df["ticker"] == cfg.ticker].copy() if "ticker" in daily_df.columns else daily_df.copy()
            daily_df_ticker.to_csv(daily_path, index=False, encoding="utf-8-sig")

            price_df = load_price_data(cfg.csv_path)
            sent_df = load_sentiment_data(daily_path)
            merged = merge_price_and_sentiment(price_df, sent_df)

            os.makedirs("outputs", exist_ok=True)
            merged.to_csv(cfg.merged_csv_path, index=False, encoding="utf-8-sig")

            print(f"  Sentiment merged successfully -> {cfg.merged_csv_path}")           
            

        except Exception as e:
            print(f"  Sentiment update failed: {e}")
            print("  Continuing with existing data (or price-only fallback).")

    # Step 2 — Load features (with fallback to price-only if merged CSV missing)
    df_feat = _load_features_with_fallback(cfg)
    last_date  = df_feat.index[-1]
    last_close = df_feat["Close"].iloc[-1]
    X_last     = df_feat[FEATURE_COLS].values

    print(f"\n[2/4] Checking models...")
    results = {}

    for H in cfg.horizons:
        model_path    = os.path.join(cfg.out_dir_models,  f"{cfg.ticker}_quantile_H{H}.keras")
        scaler_X_path = os.path.join(cfg.out_dir_scalers, f"{cfg.ticker}_quantile_H{H}_scaler_X.joblib")
        scaler_y_path = os.path.join(cfg.out_dir_scalers, f"{cfg.ticker}_quantile_H{H}_scaler_y.joblib")

        needs_train = force_train or model_needs_training(cfg, H)

        if needs_train:
            model, scaler_X, scaler_y = train_and_save(df_feat, cfg, H)
        else:
            model = tf.keras.models.load_model(
                model_path,
                custom_objects={"loss": pinball_loss(cfg.quantiles)}
            )
            scaler_X = joblib.load(scaler_X_path)
            scaler_y = joblib.load(scaler_y_path)
            print(f"  H={H}: Model loaded.")

        # Step 3 — Check recent performance
        print(f"\n[3/4] Evaluating recent performance (H={H})...")
        recent_mape = evaluate_recent(df_feat, cfg, H, model, scaler_X, scaler_y)
        print(f"  Recent MAPE (last 30 days): {recent_mape:.2f}%")

        if recent_mape > cfg.retrain_mape_threshold and not needs_train:
            print(f"  MAPE above threshold ({cfg.retrain_mape_threshold}%). Retraining...")
            model, scaler_X, scaler_y = train_and_save(df_feat, cfg, H)
            recent_mape = evaluate_recent(df_feat, cfg, H, model, scaler_X, scaler_y)
            print(f"  MAPE after retrain: {recent_mape:.2f}%")

        # Step 4 — Predict (FIX: inverse transform scaled predictions)
        X_last_scaled = scaler_X.transform(X_last[-cfg.window:])
        X_input       = X_last_scaled.reshape(1, cfg.window, -1)
        pred_s        = model.predict(X_input, verbose=0)           # scaled, shape (1, 3)
        pred_log      = scaler_y.inverse_transform(pred_s)[0]       # actual log-returns

        q10_price  = last_close * np.exp(pred_log[0])
        q50_price  = last_close * np.exp(pred_log[1])
        q90_price  = last_close * np.exp(pred_log[2])
        change_pct = (q50_price - last_close) / last_close * 100
        direction  = "UP" if q50_price > last_close else "DOWN"

        results[H] = {"last_close": round(last_close, 2),
                      "q10": round(q10_price, 2), "q50": round(q50_price, 2),
                      "q90": round(q90_price, 2), "change_pct": round(change_pct, 2),
                      "direction": direction, "recent_mape": round(recent_mape, 2)}

    # Print final results
    print(f"\n{'='*55}")
    print(f"  PRICE RANGE FORECAST — {cfg.ticker}")
    print(f"  Data up to : {last_date.date()}")
    print(f"  Last close : ${last_close:.2f}")
    print(f"{'='*55}")

    for H, r in results.items():
        label = "Tomorrow (H=1)" if H == 1 else f"{H} days ahead"
        print(f"\n  {label}:")
        print(f"    Lower bound (q10) : ${r['q10']:.2f}")
        print(f"    Median      (q50) : ${r['q50']:.2f}  ({r['change_pct']:+.2f}%) {r['direction']}")
        print(f"    Upper bound (q90) : ${r['q90']:.2f}")
        print(f"    Interval width    : ${r['q90'] - r['q10']:.2f}")
        print(f"    Recent MAPE       : {r['recent_mape']:.2f}%")

    save_json({"date": date.today().isoformat(), "ticker": cfg.ticker,
               "last_close": last_close,
               "predictions": {str(k): v for k, v in results.items()}},
              os.path.join(cfg.out_dir_preds,
                           f"{cfg.ticker}_forecast_{date.today()}.json"))
    return results


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    ticker = os.environ.get("TICKER", "AAPL")

    if "--ticker" in sys.argv:
        idx = sys.argv.index("--ticker")
        if idx + 1 < len(sys.argv):
            ticker = sys.argv[idx + 1]

    os.environ["TICKER"] = ticker
    cfg = QuantileConfig()

    if "--eval" in sys.argv:
        run_walk_forward(cfg)
    elif "--train" in sys.argv:
        run(cfg, force_train=True)
    else:
        run(cfg, force_train=False)